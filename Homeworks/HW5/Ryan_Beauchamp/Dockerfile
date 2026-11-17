FROM python:3.12-slim

# Work inside /app so the service finds its files next to app.py.
WORKDIR /app

# Install dependencies first: Docker caches this layer until requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Service code, the module the pickle imports (quantile_clipper), and the HW4 artifact + metadata.
COPY app.py model_service.py quantile_clipper.py model.joblib model_metadata.json ./

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
