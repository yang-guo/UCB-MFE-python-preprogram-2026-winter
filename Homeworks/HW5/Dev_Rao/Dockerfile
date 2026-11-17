FROM python:3.12-slim

# Adapted from Lectures/Lecture 9/demo/Dockerfile.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Pinned versions match the HW4 training environment (repo uv.lock); install first
# so code changes don't invalidate the dependency layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Service code, the custom estimator module the pickle needs, and the trusted artifact.
COPY app.py model_service.py composite.py ./
COPY artifacts/ artifacts/

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
