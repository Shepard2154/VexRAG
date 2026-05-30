FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Keep this port in sync with "port" in config.json.
EXPOSE 8080

CMD ["python", "nq_rag.py"]
