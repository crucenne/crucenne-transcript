import os
import json
import uuid
import tempfile
import threading
import subprocess
from datetime import datetime, timezone

from flask import Flask, request, jsonify, render_template, Response
from google.cloud import storage
from google.cloud import firestore
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

AKASH_API_KEY = os.environ.get("AKASH_API_KEY")
AKASH_BASE_URL = os.environ.get("AKASH_BASE_URL", "https://api.akashml.com/v1")
AKASH_MODEL = os.environ.get("AKASH_MODEL", "openai/gpt-oss-120b")

PROJECT_ID = "cs-poc-zmijby7puat1wkj99wbs6hr"
BUCKET_NAME = f"{PROJECT_ID}-audio-uploads"
GCS_PREFIX = "transcript-app"
# NOTE: "chirp_2" batch recognition is blocked in us-central1 for this project
# ("no longer generally available" — likely an allowlist/deprecation quirk of
# that specific region). europe-west4 supports full BatchRecognize for both
# "chirp" (multi-language) and "chirp_2" (denoiser/profanity/vocab-boost), so
# the whole app runs against europe-west4. Verified empirically 2026-08-24.
LOCATION = "europe-west4"

# Cloud Run gives each instance an ephemeral, memory-backed /tmp — used only as
# scratch space while converting/uploading a file. Nothing under here is relied
# on to persist: job metadata lives in Firestore, transcripts/kept audio live
# in GCS, so any instance can serve any request.
SCRATCH_DIR = os.path.join(tempfile.gettempdir(), "transcript-app")
os.makedirs(SCRATCH_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma", ".mp4", ".webm"}

# Capability matrix for this GCP project/region (europe-west4), verified empirically:
#   - "chirp"   supports multiple simultaneous language_codes (code-switching, e.g. an
#     Indian meeting mixing en-IN/hi-IN/mr-IN) but does NOT support denoiser,
#     profanity_filter, vocabulary boosting (adaptation), or diarization.
#   - "chirp_2" supports denoiser, profanity_filter, vocabulary boosting and
#     multi-channel mode, but only ONE language_code at a time in this region.
#   - Speaker diarization is not available for either model in this region.
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

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB

fs_client = firestore.Client(project=PROJECT_ID)
JOBS_COLLECTION = "jobs"
CONFIG_DOC = fs_client.collection("config").document("settings")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


_akash_client = None


def get_akash_client():
    global _akash_client
    if not AKASH_API_KEY:
        raise RuntimeError("AKASH_API_KEY is not configured on the server")
    if _akash_client is None:
        _akash_client = OpenAI(api_key=AKASH_API_KEY, base_url=AKASH_BASE_URL)
    return _akash_client


ASK_SYSTEM_PROMPT = """You are a transcript Q&A assistant. You will be given one or more \
transcripts, each split into numbered lines, and a question from the user. \
Answer using ONLY the information present in the transcripts below — never guess or use \
outside knowledge. These transcripts may mix Hindi/English (Hinglish); interpret them as-is.

Speaker identity is generally NOT available (no diarization), so unless a line names a \
person, answer "who says X" questions by pointing to the transcript(s)/line(s) where that \
content appears rather than inventing a speaker name.

Respond with ONLY a single JSON object, no markdown fences, matching this exact shape:
{"answer": "<concise answer, 1-4 sentences>", "citations": [{"job_id": "<job id>", "line": <1-based line number>}]}

Include a citation for every line you relied on. If the answer isn't found in the \
transcripts, say so plainly in "answer" and return an empty "citations" list."""


def build_ask_context(transcripts):
    blocks = []
    for t in transcripts:
        header = f'### Transcript "{t["filename"]}" (job_id: {t["job_id"]})'
        numbered = "\n".join(f"L{i + 1}: {line}" for i, line in enumerate(t["lines"]))
        blocks.append(f"{header}\n{numbered}")
    return "\n\n".join(blocks)


def load_settings():
    snap = CONFIG_DOC.get()
    if snap.exists:
        return {**DEFAULT_SETTINGS, **snap.to_dict()}
    return dict(DEFAULT_SETTINGS)


def save_settings(settings):
    CONFIG_DOC.set(settings)


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


# ---------------------------------------------------------------
# Firestore-backed job store (source of truth — no in-memory state,
# so any Cloud Run instance can serve any request)
# ---------------------------------------------------------------

def job_ref(job_id):
    return fs_client.collection(JOBS_COLLECTION).document(job_id)


def db_insert_job(job_id, filename, keep_audio, settings):
    job_ref(job_id).set({
        "filename": filename,
        "status": "queued",
        "progress": 0,
        "error": None,
        "transcript_blob": None,
        "audio_blob": None,
        "keep_audio": bool(keep_audio),
        "settings": settings,
        "created_at": now_iso(),
        "completed_at": None,
    })


def db_update_job(job_id, **fields):
    if not fields:
        return
    job_ref(job_id).update(fields)


def db_list_jobs(search=None):
    docs = fs_client.collection(JOBS_COLLECTION).order_by(
        "created_at", direction=firestore.Query.DESCENDING
    ).stream()
    rows = [{"id": d.id, **d.to_dict()} for d in docs]
    if search:
        needle = search.lower()
        rows = [r for r in rows if needle in (r.get("filename") or "").lower()]
    return rows


def db_get_job(job_id):
    snap = job_ref(job_id).get()
    if not snap.exists:
        return None
    return {"id": snap.id, **snap.to_dict()}


def db_delete_job(job_id):
    job_ref(job_id).delete()


# ---------------------------------------------------------------
# GCS-backed file storage (transcripts + optionally kept original audio)
# ---------------------------------------------------------------

def gcs_bucket():
    return storage.Client(project=PROJECT_ID).bucket(BUCKET_NAME)


def gcs_upload_file(local_path, blob_name):
    blob = gcs_bucket().blob(blob_name)
    blob.upload_from_filename(local_path)
    return blob_name


def gcs_upload_text(text, blob_name):
    blob = gcs_bucket().blob(blob_name)
    blob.upload_from_string(text, content_type="text/plain; charset=utf-8")
    return blob_name


def gcs_download_bytes(blob_name):
    return gcs_bucket().blob(blob_name).download_as_bytes()


def gcs_delete(blob_name):
    try:
        gcs_bucket().blob(blob_name).delete()
    except Exception:
        pass


def convert_to_flac(src_path, dst_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", src_path, "-ac", "1", "-ar", "16000", dst_path],
        check=True,
        capture_output=True,
    )


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
    bucket = storage.Client(project=PROJECT_ID).bucket(bucket_name)
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
    stored_audio_blob = None
    converted_flac = False
    ext = os.path.splitext(upload_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(upload_path))[0]
    flac_path = upload_path if ext == ".flac" else os.path.join(SCRATCH_DIR, f"{base_name}.flac")

    try:
        db_update_job(job_id, status="converting", progress=10)

        if ext != ".flac":
            convert_to_flac(upload_path, flac_path)
            converted_flac = True

        db_update_job(job_id, status="uploading", progress=25)
        gcs_blob_name = f"{GCS_PREFIX}/{job_id}/{os.path.basename(flac_path)}"
        gcs_upload_file(flac_path, gcs_blob_name)
        gcs_uri = f"gs://{BUCKET_NAME}/{gcs_blob_name}"

        db_update_job(job_id, status="transcribing", progress=40)
        output_prefix = f"{GCS_PREFIX}/{job_id}/output/"
        gcs_output_uri = f"gs://{BUCKET_NAME}/{output_prefix}"
        transcribe_chirp(gcs_uri, gcs_output_uri, settings)

        db_update_job(job_id, status="downloading", progress=90)
        transcript = download_and_parse_result(BUCKET_NAME, output_prefix)

        transcript_blob = f"{GCS_PREFIX}/{job_id}/transcript.txt"
        gcs_upload_text(transcript, transcript_blob)

        if keep_audio:
            stored_audio_blob = f"{GCS_PREFIX}/{job_id}/original{ext}"
            gcs_upload_file(upload_path, stored_audio_blob)

        db_update_job(
            job_id,
            status="done",
            progress=100,
            transcript_blob=transcript_blob,
            audio_blob=stored_audio_blob,
            completed_at=now_iso(),
        )
    except Exception as e:
        db_update_job(job_id, status="error", error=str(e), completed_at=now_iso())
    finally:
        for p in ({flac_path, upload_path} if converted_flac else {upload_path}):
            try:
                if os.path.exists(p):
                    os.remove(p)
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
    upload_path = os.path.join(SCRATCH_DIR, saved_name)
    file.save(upload_path)

    db_insert_job(job_id, file.filename, keep_audio, settings)

    thread = threading.Thread(
        target=process_job, args=(job_id, upload_path, file.filename, keep_audio, settings), daemon=True
    )
    thread.start()

    return jsonify({"job_id": job_id, "settings": settings})


@app.route("/api/status/<job_id>")
def status(job_id):
    row = db_get_job(job_id)
    if not row:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(row)


@app.route("/api/download/<job_id>")
def download(job_id):
    row = db_get_job(job_id)
    if not row or row.get("status") != "done" or not row.get("transcript_blob"):
        return jsonify({"error": "Transcript not ready"}), 404
    data = gcs_download_bytes(row["transcript_blob"])
    download_name = f"{os.path.splitext(row['filename'])[0]}_transcript.txt"
    return Response(
        data, mimetype="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
    )


@app.route("/api/audio/<job_id>")
def audio(job_id):
    row = db_get_job(job_id)
    if not row or not row.get("audio_blob"):
        return jsonify({"error": "Audio not stored for this job"}), 404
    data = gcs_download_bytes(row["audio_blob"])
    return Response(
        data, mimetype="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{row["filename"]}"'},
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
    if not row or row.get("status") != "done" or not row.get("transcript_blob"):
        return jsonify({"error": "Transcript not available"}), 404
    text = gcs_download_bytes(row["transcript_blob"]).decode("utf-8")
    return jsonify({"transcript": text, "filename": row["filename"]})


@app.route("/api/ask", methods=["POST"])
def ask():
    body = request.get_json(force=True, silent=True) or {}
    job_ids = body.get("job_ids")
    question = (body.get("question") or "").strip()

    if not isinstance(job_ids, list) or not job_ids:
        return jsonify({"error": "job_ids must be a non-empty list"}), 400
    if not question:
        return jsonify({"error": "question is required"}), 400

    transcripts = []
    for job_id in job_ids:
        row = db_get_job(job_id)
        if not row or row.get("status") != "done" or not row.get("transcript_blob"):
            return jsonify({"error": f"Transcript not available for job {job_id}"}), 404
        text = gcs_download_bytes(row["transcript_blob"]).decode("utf-8")
        lines = text.split("\n")
        transcripts.append({"job_id": job_id, "filename": row["filename"], "lines": lines})

    try:
        client = get_akash_client()
        completion = client.chat.completions.create(
            model=AKASH_MODEL,
            messages=[
                {"role": "system", "content": ASK_SYSTEM_PROMPT},
                {"role": "user", "content": f"{build_ask_context(transcripts)}\n\nQuestion: {question}"},
            ],
            temperature=0,
        )
        raw = completion.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": f"AI request failed: {e}"}), 502

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        answer = parsed.get("answer", "").strip()
        raw_citations = parsed.get("citations", [])
        if not isinstance(raw_citations, list):
            raw_citations = []
    except (json.JSONDecodeError, AttributeError):
        answer = raw
        raw_citations = []

    transcripts_by_id = {t["job_id"]: t for t in transcripts}
    citations = []
    for c in raw_citations:
        if not isinstance(c, dict):
            continue
        job_id = c.get("job_id")
        line_no = c.get("line")
        t = transcripts_by_id.get(job_id)
        if not t or not isinstance(line_no, int) or line_no < 1 or line_no > len(t["lines"]):
            continue
        citations.append({
            "job_id": job_id,
            "filename": t["filename"],
            "line": line_no,
            "text": t["lines"][line_no - 1],
        })

    return jsonify({"answer": answer, "citations": citations})


@app.route("/api/job/<job_id>", methods=["DELETE"])
def delete_job(job_id):
    row = db_get_job(job_id)
    if not row:
        return jsonify({"error": "Job not found"}), 404

    if row.get("transcript_blob"):
        gcs_delete(row["transcript_blob"])
    if row.get("audio_blob"):
        gcs_delete(row["audio_blob"])

    db_delete_job(job_id)
    return jsonify({"ok": True})


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG") == "1"
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)
