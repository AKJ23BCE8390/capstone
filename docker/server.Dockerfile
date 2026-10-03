# Micro-footprint Python container engine distribution build layer
FROM python:3.10-slim

WORKDIR /app

# Ensure security updates are run and system dependencies are cleared out
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cache optimization package mapping injection steps
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir flwr==1.5.0 requests numpy==1.23.5

# Expose internal execution channels
COPY server/ strategy.py aggregator.py straggler.py server.py /app/

EXPOSE 8080

ENTRYPOINT ["python", "server.py"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
