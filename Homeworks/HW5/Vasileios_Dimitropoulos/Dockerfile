FROM python:3.12-slim

# No .pyc files in the image; logs are flushed immediately (visible in `docker logs`).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first so this layer is cached when only the code changes. Versions are
# pinned in requirements.txt to my HW4 training environment (scikit-learn 1.9.1, ...).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code, the custom transformer module that the pickled pipeline imports
# (model_pipeline.RankFeatureDesign), and the fitted HW4 artifact + its metadata/hash.
COPY app.py model_pipeline.py ./
COPY artifacts/model.joblib artifacts/model_metadata.json artifacts/

# Run as an unprivileged user; the service only needs to read /app.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
