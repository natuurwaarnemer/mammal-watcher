FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/clips/confirmed /app/clips/uncertain

COPY mammal_watcher.py classifier.py rtsp_consumer.py mqtt_publisher.py species_mammals_nl.csv ./

CMD ["python", "-u", "mammal_watcher.py", "--config", "/app/config.yaml"]
