# Chirp Transcriber

Drop an audio file, transcribe it with Google Cloud Speech-to-Text (Chirp / Chirp 2),
browse history, and tune model settings (language, denoiser, vocabulary boosting) per
recording condition (meeting room, phone call, noisy/mixed audio).

## Local development

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/python app.py
```

Requires Google Cloud Application Default Credentials with access to the
`cs-poc-zmijby7puat1wkj99wbs6hr` project (Speech-to-Text + Cloud Storage):

```bash
gcloud auth application-default login
```

## Production (Cloud Run)

Job metadata lives in Firestore and transcripts/kept audio live in GCS, so the app is
stateless per-instance and safe to run on Cloud Run. Every push to `main` auto-builds
and redeploys via `.github/workflows/deploy.yml` (Cloud Build + Cloud Run, authenticated
with Workload Identity Federation — no static keys).

The service runs with `--min-instances=1 --max-instances=1 --no-cpu-throttling` so the
background thread that drives each transcription job (conversion → GCS upload → Chirp
BatchRecognize → result parsing) always survives on the one instance handling it.

To deploy manually:

```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/cs-poc-zmijby7puat1wkj99wbs6hr/transcript-app-repo/transcript-app:manual
gcloud run deploy transcript-app \
  --image us-central1-docker.pkg.dev/cs-poc-zmijby7puat1wkj99wbs6hr/transcript-app-repo/transcript-app:manual \
  --region us-central1 \
  --service-account transcript-app-run@cs-poc-zmijby7puat1wkj99wbs6hr.iam.gserviceaccount.com \
  --allow-unauthenticated --no-cpu-throttling --min-instances=1 --max-instances=1 \
  --memory=2Gi --cpu=2 --timeout=3600
```
