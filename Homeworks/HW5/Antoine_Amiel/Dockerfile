FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Same pinned versions as the HW4 training environment (see requirements.txt).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Service code and the exported HW4 model. The model uses only scikit-learn
# classes, so no custom transformer module is needed.
COPY app.py model_service.py ./
COPY artifacts/ artifacts/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
