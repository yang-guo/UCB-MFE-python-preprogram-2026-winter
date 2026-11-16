FROM python:3.12-slim

# Pinned versions = my HW4 training environment (sklearn 1.8.0, numpy 2.3.5, ...).
# Loading a joblib/sklearn artifact across versions is unsupported.
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Service code, the importable custom transformers, and the trusted HW4 artifact.
# The export script, tests, databases and local environments are not copied.
COPY app.py hw4_model.py ./
COPY artifacts/hw4_model.joblib artifacts/model_metadata.json artifacts/

# Run as an unprivileged user; the image only needs to read its files.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
