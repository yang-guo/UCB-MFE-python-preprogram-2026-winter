FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY artifacts/model.joblib artifacts/model.joblib
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
