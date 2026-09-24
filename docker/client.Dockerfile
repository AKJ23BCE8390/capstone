FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY client/ /app/client/
COPY scripts/ /app/scripts/

ENTRYPOINT ["python", "scripts/run_client.py"]
