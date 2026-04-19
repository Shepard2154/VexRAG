FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Keep this port in sync with "port" in config.json.
EXPOSE 9003

CMD ["python", "small_rag.py"]
