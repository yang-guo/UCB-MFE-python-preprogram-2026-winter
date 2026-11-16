FROM python:3.12-slim
WORKDIR /app

# Avoid nested BLAS threads for this small, single-row prediction service.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY app.py schemas.py serving.py ./
COPY hw4/__init__.py hw4/preprocessing.py hw4/
COPY artifacts/fixed_anchor_2012.joblib artifacts/model_metadata.json artifacts/

# Prediction only reads the packaged artifact and needs no privileged user.
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
