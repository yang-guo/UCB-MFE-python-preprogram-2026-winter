FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY app.py hw4_model.py model.joblib example_request.json ./

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
