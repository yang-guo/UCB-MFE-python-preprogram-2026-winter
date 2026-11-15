FROM python:3.12-slim

# LightGBM's Linux wheel links against the GNU OpenMP runtime, which the slim image lacks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so code-only changes reuse the cached install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The service code, its model module, and the trusted fitted HW4 artifact.
# (No custom transformer classes: the artifact holds a plain LGBMRegressor.)
COPY app.py model_service.py ./
COPY artifacts/ artifacts/

# Run as an unprivileged user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
