import os
import json
import time
import uuid
import sqlite3
import threading
import subprocess
from datetime import datetime, timezone

from flask import Flask, request, jsonify, render_template, send_from_directory
from google.cloud import storage
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

PROJECT_ID = "cs-poc-zmijby7puat1wkj99wbs6hr"
BUCKET_NAME = f"{PROJECT_ID}-audio-uploads"
# NOTE: "chirp_2" batch recognition is blocked in us-central1 for this project
# ("no longer generally available" — likely an allowlist/deprecation quirk of
# that specific region). europe-west4 supports full BatchRecognize for both
# "chirp" (multi-language) and "chirp_2" (denoiser/profanity/vocab-boost), so
# the whole app runs against europe-west4. Verified empirically 2026-08-24.
LOCATION = "europe-west4"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
AUDIO_STORE_DIR = os.path.join(BASE_DIR, "audio_store")
TRANSCRIPT_DIR = os.path.join(BASE_DIR, "transcripts")
DB_PATH = os.path.join(BASE_DIR, "history.db")
SETTINGS_PATH = os.path.join(BASE_DIR, "settings.json")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AUDIO_STORE_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma", ".mp4", ".webm"}

# Capability matrix for this GCP project/region (us-central1), verified empirically:
#   - "chirp"   supports multiple simultaneous language_codes (code-switching, e.g. an
#     Indian meeting mixing en-IN/hi-IN/mr-IN) but does NOT support denoiser,
#     profanity_filter, vocabulary boosting (adaptation), or diarization.
#   - "chirp_2" supports denoiser, profanity_filter, vocabulary boosting and
#     multi-channel mode, but only ONE language_code at a time in this region.
#   - Speaker diarization is not available for either model in us-central1.
AVAILABLE_MODELS = ["chirp", "chirp_2"]
AVAILABLE_LANGUAGES = [
    "en-IN", "hi-IN", "mr-IN", "gu-IN", "ta-IN", "te-IN",
    "kn-IN", "bn-IN", "pa-IN", "ml-IN", "en-US", "en-GB",
]

DEFAULT_SETTINGS = {
    "model": "chirp",
    "language_codes": ["en-IN", "hi-IN", "mr-IN"],
    "punctuation": True,
    "profanity_filter": False,
    "denoise": False,
    "snr_threshold": 10.0,
    "multi_channel": False,
    "phrase_hints": [],
    "phrase_boost": 10.0,
}

SETTINGS_LOCK = threading.Lock()


def load_settings():
    with SETTINGS_LOCK:
        if os.path.exists(SETTINGS_PATH):
            with open(SETTINGS_PATH) as f:
                saved = json.load(f)
            merged = {**DEFAULT_SETTINGS, **saved}
            return merged
        return dict(DEFAULT_SETTINGS)


def save_settings(settings):
    with SETTINGS_LOCK:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=2)


def normalize_settings(raw, base=None):
    """Merge raw (partial, possibly stringly-typed) settings onto a base config,
    coercing types and enforcing the model capability matrix above."""
    s = dict(base or DEFAULT_SETTINGS)

    if "model" in raw and raw["model"] in AVAILABLE_MODELS:
        s["model"] = raw["model"]

    if "language_codes" in raw:
        langs = raw["language_codes"]
        if isinstance(langs, str):
            langs = [l.strip() for l in langs.split(",") if l.strip()]
        langs = [l for l in langs if l in AVAILABLE_LANGUAGES]
        if langs:
            s["language_codes"] = langs

    for bool_field in ("punctuation", "profanity_filter", "denoise", "multi_channel"):
        if bool_field in raw:
            v = raw[bool_field]
            s[bool_field] = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")

    if "snr_threshold" in raw:
        try:
            s["snr_threshold"] = float(raw["snr_threshold"])
        except (TypeError, ValueError):
            pass

    if "phrase_boost" in raw:
        try:
            s["phrase_boost"] = float(raw["phrase_boost"])
        except (TypeError, ValueError):
            pass

    if "phrase_hints" in raw:
        hints = raw["phrase_hints"]
        if isinstance(hints, str):
            hints = [h.strip() for h in hints.replace("\n", ",").split(",") if h.strip()]
        s["phrase_hints"] = list(hints)

    # Enforce the capability matrix: "chirp" cannot do multi-lang alongside these
    # features, and only "chirp_2" supports them at all in this region.
    if s["model"] == "chirp":
        s["profanity_filter"] = False
        s["denoise"] = False
        s["multi_channel"] = False
        s["phrase_hints"] = []
    else:
        # chirp_2 only supports a single language_code in this region.
        s["language_codes"] = s["language_codes"][:1] or [DEFAULT_SETTINGS["language_codes"][0]]

    return s

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB

# In-memory job store for live progress: job_id -> {status, progress, error, transcript, filename}
JOBS = {}
JOBS_LOCK = threading.Lock()

DB_LOCK = threading.Lock()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with DB_LOCK, get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                transcript_file TEXT,
                audio_file TEXT,
                keep_audio INTEGER DEFAULT 0,
                settings_json TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            )
            """
        )
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        if "settings_json" not in existing_cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN settings_json TEXT")


def db_insert_job(job_id, filename, keep_audio, settings):
    with DB_LOCK, get_db() as conn:
        conn.execute(
            "INSERT INTO jobs (id, filename, status, keep_audio, settings_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, filename, "queued", int(keep_audio), json.dumps(settings), now_iso()),
        )


def db_update_job(job_id, **fields):
    if not fields:
        return
    cols = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [job_id]
    with DB_LOCK, get_db() as conn:
        conn.execute(f"UPDATE jobs SET {cols} WHERE id = ?", values)


def db_list_jobs(search=None):
    with DB_LOCK, get_db() as conn:
        if search:
            rows = conn.execute(
                "SELECT * FROM jobs WHERE filename LIKE ? ORDER BY created_at DESC",
                (f"%{search}%",),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def db_get_job(job_id):
    with DB_LOCK, get_db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None


def db_delete_job(job_id):
    with DB_LOCK, get_db() as conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def now_iso():
    return datetime.now(timezone.utc).isoformat()


init_db()


def set_job(job_id, **kwargs):
    with JOBS_LOCK:
        JOBS[job_id].update(kwargs)


def convert_to_flac(src_path, dst_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", dst_path],
        check=True,
        capture_output=True,
    )


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    return f"gs://{bucket_name}/{destination_blob_name}"


def build_recognition_config(settings):
    feature_kwargs = {"enable_automatic_punctuation": settings.get("punctuation", True)}

    denoiser_config = None
    adaptation = None

    if settings["model"] == "chirp_2":
        feature_kwargs["profanity_filter"] = settings.get("profanity_filter", False)
        if settings.get("multi_channel"):
            feature_kwargs["multi_channel_mode"] = (
                cloud_speech.RecognitionFeatures.MultiChannelMode.SEPARATE_RECOGNITION_PER_CHANNEL
            )
        if settings.get("denoise"):
            denoiser_config = cloud_speech.DenoiserConfig(
                denoise_audio=True,
                snr_threshold=settings.get("snr_threshold", 10.0),
            )
        hints = settings.get("phrase_hints") or []
        if hints:
            boost = settings.get("phrase_boost", 10.0)
            phrases = [cloud_speech.PhraseSet.Phrase(value=h, boost=boost) for h in hints]
            adaptation = cloud_speech.SpeechAdaptation(
                phrase_sets=[
                    cloud_speech.SpeechAdaptation.AdaptationPhraseSet(
                        inline_phrase_set=cloud_speech.PhraseSet(phrases=phrases)
                    )
                ]
            )

    config_kwargs = {
        "auto_decoding_config": cloud_speech.AutoDetectDecodingConfig(),
        "language_codes": settings["language_codes"],
        "model": settings["model"],
        "features": cloud_speech.RecognitionFeatures(**feature_kwargs),
    }
    if denoiser_config is not None:
        config_kwargs["denoiser_config"] = denoiser_config
    if adaptation is not None:
        config_kwargs["adaptation"] = adaptation

    return cloud_speech.RecognitionConfig(**config_kwargs)


def transcribe_chirp(gcs_uri, gcs_output_uri, settings):
    client = SpeechClient(
        client_options=ClientOptions(api_endpoint=f"{LOCATION}-speech.googleapis.com")
    )

    config = build_recognition_config(settings)

    files_metadata = [cloud_speech.BatchRecognizeFileMetadata(uri=gcs_uri)]

    request_ = cloud_speech.BatchRecognizeRequest(
        recognizer=f"projects/{PROJECT_ID}/locations/{LOCATION}/recognizers/_",
        config=config,
        files=files_metadata,
        recognition_output_config=cloud_speech.RecognitionOutputConfig(
            gcs_output_config=cloud_speech.GcsOutputConfig(uri=gcs_output_uri)
        ),
    )

    operation = client.batch_recognize(request=request_)
    response = operation.result(timeout=7200)
    return response


def download_and_parse_result(bucket_name, prefix):
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    full_transcript = ""
    for blob in blobs:
        if blob.name.endswith(".json"):
            content = blob.download_as_bytes()
            data = json.loads(content)
            if "results" in data:
                for result in data["results"]:
                    alts = result.get("alternatives")
                    if alts:
                        full_transcript += alts[0].get("transcript", "") + "\n"
    return full_transcript


def process_job(job_id, upload_path, original_filename, keep_audio, settings):
    stored_audio_name = None
    converted_flac = False
    ext = os.path.splitext(upload_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(upload_path))[0]
    flac_path = upload_path if ext == ".flac" else os.path.join(UPLOAD_DIR, f"{base_name}.flac")

    try:
        set_job(job_id, status="converting", progress=10)
        db_update_job(job_id, status="converting")

        if ext != ".flac":
            convert_to_flac(upload_path, flac_path)
            converted_flac = True

        set_job(job_id, status="uploading", progress=25)
        db_update_job(job_id, status="uploading")
        gcs_blob_name = f"transcript-app/{job_id}/{os.path.basename(flac_path)}"
        gcs_uri = upload_blob(BUCKET_NAME, flac_path, gcs_blob_name)

        set_job(job_id, status="transcribing", progress=40)
        db_update_job(job_id, status="transcribing")
        output_prefix = f"transcript-app/{job_id}/output/"
        gcs_output_uri = f"gs://{BUCKET_NAME}/{output_prefix}"
        transcribe_chirp(gcs_uri, gcs_output_uri, settings)

        set_job(job_id, status="downloading", progress=90)
        db_update_job(job_id, status="downloading")
        transcript = download_and_parse_result(BUCKET_NAME, output_prefix)

        transcript_file = os.path.join(TRANSCRIPT_DIR, f"{job_id}.txt")
        with open(transcript_file, "w") as f:
            f.write(transcript)

        if keep_audio:
            stored_audio_name = f"{job_id}{ext}"
            stored_audio_path = os.path.join(AUDIO_STORE_DIR, stored_audio_name)
            os.replace(upload_path, stored_audio_path)

        set_job(
            job_id,
            status="done",
            progress=100,
            transcript=transcript,
            transcript_file=f"{job_id}.txt",
        )
        db_update_job(
            job_id,
            status="done",
            transcript_file=f"{job_id}.txt",
            audio_file=stored_audio_name,
            completed_at=now_iso(),
        )
    except Exception as e:
        set_job(job_id, status="error", error=str(e))
        db_update_job(job_id, status="error", error=str(e), completed_at=now_iso())
    finally:
        if converted_flac:
            try:
                if os.path.exists(flac_path):
                    os.remove(flac_path)
            except OSError:
                pass
        if not keep_audio:
            try:
                if os.path.exists(upload_path):
                    os.remove(upload_path)
            except OSError:
                pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    keep_audio = request.form.get("keep_audio", "false").lower() in ("1", "true", "yes", "on")

    raw_settings = {}
    if "settings" in request.form:
        try:
            raw_settings = json.loads(request.form["settings"])
        except (TypeError, ValueError):
            raw_settings = {}
    settings = normalize_settings(raw_settings, base=load_settings())

    job_id = uuid.uuid4().hex[:12]
    saved_name = f"{job_id}{ext}"
    upload_path = os.path.join(UPLOAD_DIR, saved_name)
    file.save(upload_path)

    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "filename": file.filename,
            "error": None,
            "transcript": None,
        }
    db_insert_job(job_id, file.filename, keep_audio, settings)

    thread = threading.Thread(
        target=process_job, args=(job_id, upload_path, file.filename, keep_audio, settings), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id, "settings": settings})


@app.route("/api/status/<job_id>")
def status(job_id):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job:
        return jsonify(job)
    row = db_get_job(job_id)
    if not row:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(row)


@app.route("/api/download/<job_id>")
def download(job_id):
    row = db_get_job(job_id)
    if not row or row.get("status") != "done" or not row.get("transcript_file"):
        return jsonify({"error": "Transcript not ready"}), 404
    return send_from_directory(
        TRANSCRIPT_DIR, row["transcript_file"], as_attachment=True,
        download_name=f"{os.path.splitext(row['filename'])[0]}_transcript.txt",
    )


@app.route("/api/audio/<job_id>")
def audio(job_id):
    row = db_get_job(job_id)
    if not row or not row.get("audio_file"):
        return jsonify({"error": "Audio not stored for this job"}), 404
    return send_from_directory(
        AUDIO_STORE_DIR, row["audio_file"], as_attachment=True,
        download_name=row["filename"],
    )


@app.route("/api/config")
def config():
    return jsonify({
        "models": AVAILABLE_MODELS,
        "languages": AVAILABLE_LANGUAGES,
        "capabilities": {
            "chirp": {
                "multi_language": True,
                "denoise": False,
                "profanity_filter": False,
                "phrase_hints": False,
                "multi_channel": False,
            },
            "chirp_2": {
                "multi_language": False,
                "denoise": True,
                "profanity_filter": True,
                "phrase_hints": True,
                "multi_channel": True,
            },
        },
        "presets": {
            "meeting_room": {
                "label": "Meeting room (multi-speaker, mixed language)",
                "settings": {
                    "model": "chirp",
                    "language_codes": ["en-IN", "hi-IN", "mr-IN"],
                    "punctuation": True,
                },
            },
            "single_speaker": {
                "label": "Single speaker / dictation",
                "settings": {
                    "model": "chirp_2",
                    "language_codes": ["en-IN"],
                    "punctuation": True,
                    "denoise": False,
                },
            },
            "noisy_mixed": {
                "label": "Noisy / mixed recording",
                "settings": {
                    "model": "chirp_2",
                    "language_codes": ["en-IN"],
                    "punctuation": True,
                    "denoise": True,
                    "snr_threshold": 10.0,
                },
            },
            "stereo_channels": {
                "label": "Stereo / separate mic channels",
                "settings": {
                    "model": "chirp_2",
                    "language_codes": ["en-IN"],
                    "punctuation": True,
                    "denoise": True,
                    "multi_channel": True,
                },
            },
        },
    })


@app.route("/api/settings", methods=["GET"])
def get_settings():
    return jsonify(load_settings())


@app.route("/api/settings", methods=["POST"])
def update_settings():
    raw = request.get_json(force=True, silent=True) or {}
    settings = normalize_settings(raw, base=load_settings())
    save_settings(settings)
    return jsonify(settings)


@app.route("/api/history")
def history():
    search = request.args.get("q", "").strip() or None
    rows = db_list_jobs(search=search)
    return jsonify(rows)


@app.route("/api/transcript/<job_id>")
def transcript_text(job_id):
    row = db_get_job(job_id)
    if not row or row.get("status") != "done" or not row.get("transcript_file"):
        return jsonify({"error": "Transcript not available"}), 404
    path = os.path.join(TRANSCRIPT_DIR, row["transcript_file"])
    if not os.path.exists(path):
        return jsonify({"error": "Transcript file missing"}), 404
    with open(path) as f:
        text = f.read()
    return jsonify({"transcript": text, "filename": row["filename"]})


@app.route("/api/job/<job_id>", methods=["DELETE"])
def delete_job(job_id):
    row = db_get_job(job_id)
    if not row:
        return jsonify({"error": "Job not found"}), 404

    if row.get("transcript_file"):
        try:
            os.remove(os.path.join(TRANSCRIPT_DIR, row["transcript_file"]))
        except OSError:
            pass
    if row.get("audio_file"):
        try:
            os.remove(os.path.join(AUDIO_STORE_DIR, row["audio_file"]))
        except OSError:
            pass

    db_delete_job(job_id)
    with JOBS_LOCK:
        JOBS.pop(job_id, None)

    return jsonify({"ok": True})


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host="127.0.0.1", port=5050, debug=debug, threaded=True)
