# HW5 model service: my HW4 Ridge pipeline behind FastAPI (Ananya Das).
# Build from this folder:  docker build -t mfe-hw5:local .
# Run:                      docker run --rm --name mfe-hw5 -p 127.0.0.1:8000:8000 mfe-hw5:local
FROM python:3.12-slim

# No .pyc files in the image; log lines reach `docker logs` immediately.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so code changes reuse this cached layer. The versions are pinned in
# requirements.txt to match the HW4 training environment (scikit-learn 1.8.0, numpy 2.3.5,
# pandas 2.3.3, scipy 1.16.3, joblib 1.5.3), because sklearn artifacts are not portable
# across versions.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only what the service needs at runtime: the app, the inference layer, and the saved HW4
# model with its metadata. No hw4.py, DuckDB, matplotlib, export script, tests, or data;
# the model is loaded from the artifact at startup and never retrained here.
COPY app.py model.py ./
COPY artifacts/model.joblib artifacts/model_metadata.json ./artifacts/

# Run as an unprivileged user; the app only reads its files.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

# python:3.12-slim has no curl, so the health check uses the standard library.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
