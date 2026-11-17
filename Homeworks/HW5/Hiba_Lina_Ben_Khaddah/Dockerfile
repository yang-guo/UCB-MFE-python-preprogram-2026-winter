FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app.py course_data.py ./
COPY artifacts/ ./artifacts/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
