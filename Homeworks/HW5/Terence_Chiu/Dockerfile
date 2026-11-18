FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The service, the custom transformer the pickled pipeline needs, and the fitted artifact.
COPY app.py course_data.py ./
COPY artifacts/ artifacts/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
