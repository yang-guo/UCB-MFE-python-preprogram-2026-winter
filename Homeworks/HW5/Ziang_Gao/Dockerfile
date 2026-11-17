FROM python:3.12-slim

WORKDIR /app

# Dependencies first. Docker then reuses this layer when only the code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The service, the model logic, the custom transformer module, and the fitted HW4 artifact.
COPY app.py inference.py preprocessing.py ./
COPY artifacts/ artifacts/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
