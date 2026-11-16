FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so code changes do not invalidate this (slow) layer.
# Versions are pinned to the HW4 training environment (scikit-learn 1.8.0 etc.).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only what the service needs: the app, the custom transformer module that the
# pickled pipeline imports, and the fitted HW4 artifact (no data, no venv, no secrets).
COPY app.py hw4_model.py ./
COPY artifacts/ ./artifacts/

# Do not run the service as root.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
