FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY templates/ templates/

ENV PORT=8080
EXPOSE 8080

# Single worker keeps the background-thread job model simple; combined with
# --min-instances=1 --max-instances=1 at deploy time so all requests for a
# given job land on the instance that's actually processing it.
CMD exec gunicorn -w 1 --threads 8 --timeout 120 -b 0.0.0.0:${PORT} app:app
