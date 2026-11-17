FROM python:3.12-slim

# Logs go straight to `docker logs`; no .pyc files are written into the image.
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first: this layer is rebuilt only when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The service, its model logic and the fitted HW4 artifact (the HW4 model needs no custom
# transformer module). .dockerignore keeps everything else out of the build context.
COPY app.py inference.py ./
COPY artifacts/ artifacts/

# The service only reads its files, so it does not need root.
RUN useradd --create-home --uid 10001 service
USER service

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
