FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install pinned dependencies first so this layer is cached when only code changes.
# The model stack matches the HW4 training environment (see requirements.txt).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application, the custom-transformer module the pickle imports (hw4_model.QuantileClipper),
# and the fitted HW4 artifact. No database, credentials or virtual environment is copied.
COPY app.py hw4_model.py ./
COPY model/ ./model/

# Run as an unprivileged user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
