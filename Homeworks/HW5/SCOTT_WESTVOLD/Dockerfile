FROM python:3.12-slim
# PLEASE NOTE DOCKER SETUP SOURCED FROM Lecture 9 Demo.
WORKDIR /app 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY model.joblib .
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
