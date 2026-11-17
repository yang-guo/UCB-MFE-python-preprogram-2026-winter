FROM python:3.12-slim

WORKDIR /app

# Same pinned versions as the HW4 training env (sklearn 1.8.0 etc.); see requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Service code, the custom transformer needed to unpickle, and the fitted HW4 artifact.
COPY app.py hw5_transformers.py ./
COPY model/ ./model/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
