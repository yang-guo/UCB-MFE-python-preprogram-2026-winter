FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so editing the app does not reinstall them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The service, its model code, and the saved HW4 model.
COPY app.py model_service.py ./
COPY artifacts/ artifacts/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
