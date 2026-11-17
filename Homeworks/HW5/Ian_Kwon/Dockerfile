FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so code changes do not reinstall them.  The versions match the
# HW4 training environment (scikit-learn 1.8.0, numpy 2.3.5, joblib 1.5.3).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Service code, the HW4 module the pickle imports, and the fitted artifact.
COPY app.py model_service.py ./
COPY quantlab/ quantlab/
COPY artifacts/ artifacts/

# Run as an unprivileged user; the service only reads its files.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
