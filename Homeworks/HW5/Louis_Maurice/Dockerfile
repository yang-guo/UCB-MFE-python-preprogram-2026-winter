FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Service code, the custom transformer module, and the fitted HW4 artifact.
COPY app.py model.py course_data.py ./
COPY artifacts/ artifacts/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
