FROM python:3.12-slim

WORKDIR /app

# Dependencies first, so this layer is reused when only the code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The service code and the fitted HW4 artifact. The artifact holds only
# scikit-learn objects, so there are no custom transformer modules to copy.
COPY app.py model_service.py ./
COPY artifacts/ artifacts/

# Do not run the service as root.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
