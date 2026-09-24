FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ /app/server/
COPY scripts/ /app/scripts/

EXPOSE 8080

ENTRYPOINT ["python", "scripts/run_server.py"]
