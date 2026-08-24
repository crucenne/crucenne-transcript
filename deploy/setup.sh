#!/bin/bash
# Provisions a fresh Debian/Ubuntu Compute Engine VM to run the Chirp Transcriber app.
# Run as root (e.g. via `gcloud compute ssh ... --command "sudo bash setup.sh"`).
set -euo pipefail

REPO_URL="https://github.com/crucenne/crucenne-transcript.git"
APP_DIR="/opt/transcript-app"
APP_USER="transcript"

apt-get update
apt-get install -y python3 python3-venv python3-pip ffmpeg nginx git

id -u "$APP_USER" &>/dev/null || useradd -r -m -d "$APP_DIR" -s /usr/sbin/nologin "$APP_USER"

if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull
else
  rm -rf "$APP_DIR"
  git clone "$REPO_URL" "$APP_DIR"
fi

python3 -m venv "$APP_DIR/venv"
"$APP_DIR/venv/bin/pip" install --upgrade pip -q
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/requirements.txt" -q

mkdir -p "$APP_DIR/uploads" "$APP_DIR/audio_store" "$APP_DIR/transcripts"
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

cp "$APP_DIR/deploy/transcript-app.service" /etc/systemd/system/transcript-app.service
systemctl daemon-reload
systemctl enable transcript-app
systemctl restart transcript-app

cp "$APP_DIR/deploy/nginx.conf" /etc/nginx/sites-available/transcript-app
ln -sf /etc/nginx/sites-available/transcript-app /etc/nginx/sites-enabled/transcript-app
rm -f /etc/nginx/sites-enabled/default
nginx -t
systemctl restart nginx

echo "Deployed. Check: systemctl status transcript-app nginx"
