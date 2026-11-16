FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Install the same model-library versions used to fit the HW4 artifact.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# The service needs the custom transformer class to unpickle the fitted pipeline.
COPY app.py hw4_pipeline.py model.joblib model_reference.json ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
