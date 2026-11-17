FROM python:3.12-slim

WORKDIR /app

# requirements first: this layer is rebuilt only when the versions change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# the API, the model code and the saved HW4 model (no database, no training code)
COPY app.py model_service.py ./
COPY artifacts/ artifacts/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
