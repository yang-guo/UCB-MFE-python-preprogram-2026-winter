FROM python:3.12-slim

# Set the working directory to /app.

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Copy requirements.txt and install its dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app.py, fitted HW4 artifact.
COPY app.py hw4_model.py ./
COPY artifacts/ artifacts/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
