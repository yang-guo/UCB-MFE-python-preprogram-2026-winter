FROM python:3.12-slim

# TODO: Set the working directory to /app.
WORKDIR /app

# TODO: Copy requirements.txt and install its dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# TODO: Copy app.py, your fitted HW4 artifact, and any custom transformer modules.
COPY app.py fm_model.py model.joblib ./

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
