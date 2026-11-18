FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY app.py model_tools.py ./
COPY artifacts/ ./artifacts/
COPY example_request.json ./
USER 10001:10001
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
