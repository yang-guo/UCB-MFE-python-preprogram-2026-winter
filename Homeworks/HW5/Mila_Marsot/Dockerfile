FROM python:3.12-slim

# All the files go in the folder /app inside the image.
WORKDIR /app

# Install the libraries first, with the same versions as in HW4 (scikit-learn 1.8.0, ...).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API, the model code (with the custom transformer) and the saved model.
# .dockerignore keeps .venv, databases, passwords and caches out of the image.
COPY app.py hw4_model.py ./
COPY artifacts/ artifacts/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
