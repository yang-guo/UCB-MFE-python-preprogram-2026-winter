FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY app.py model_service.py ./
COPY artifacts/hw4_ridge.joblib artifacts/hw4_ridge.json ./artifacts/

USER 10001:10001
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
