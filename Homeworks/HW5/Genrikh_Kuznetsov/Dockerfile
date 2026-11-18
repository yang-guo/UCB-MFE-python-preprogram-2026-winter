FROM python:3.12-slim

# Don't write .pyc files; send logs straight to `docker logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first so this layer is cached when only the code changes.
# Versions match the HW4 training environment (scikit-learn 1.8.0, numpy 2.3.5, ...).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Service code, the custom transformer module needed to unpickle the pipeline,
# and the fitted HW4 artifact. No training data, databases, or credentials.
COPY app.py model_service.py hw4_pipeline.py ./
COPY artifacts/model.joblib artifacts/model_metadata.json artifacts/

# Run as an unprivileged user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
