FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The service, the HW4 modules needed to load the model, and the fitted model.
COPY app.py serving.py ./
COPY hw4/ hw4/
COPY artifacts/model.joblib artifacts/model_metadata.json artifacts/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
