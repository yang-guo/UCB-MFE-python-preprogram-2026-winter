FROM python:3.12-slim
# Set the working directory to /app.
WORKDIR /app

# Copy requirements.txt and install its dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#  Copy app.py, your fitted HW4 artifact, and any custom transformer modules
COPY app.py factor_model.py ./
COPY artifacts/ artifacts/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
