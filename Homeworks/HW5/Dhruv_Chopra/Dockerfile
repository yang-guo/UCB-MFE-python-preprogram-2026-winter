FROM python:3.12-slim

WORKDIR /app

# Dependencies first so code changes do not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Service code, the custom module the pickle references, and the fitted HW4 artifact.
COPY app.py ranks_model.py ./
COPY artifacts/hw4_ridge_v1.joblib artifacts/

# Run as an unprivileged user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
