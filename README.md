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

## Production

Served via `gunicorn` (see `deploy/transcript-app.service` for the systemd unit used
on the GCP Compute Engine deployment). Run with a single worker to keep the in-memory
job-progress tracking consistent:

```bash
gunicorn -w 1 --threads 8 -b 0.0.0.0:8080 app:app
```
