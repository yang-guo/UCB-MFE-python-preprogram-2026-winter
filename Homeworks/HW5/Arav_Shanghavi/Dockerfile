FROM python:3.12-slim

WORKDIR /app

# dependencies first, so editing app code doesn't redo the pip layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# blend_model.py must sit next to app.py: the pickled pipeline imports it by that name
COPY app.py blend_model.py ./
COPY artifacts/ ./artifacts/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
