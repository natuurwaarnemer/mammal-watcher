#!/bin/bash
set -e
cd /home/natuurwaarnemer/mammal-watcher

# Zorg dat .env bestaat met minimale defaults
if [ ! -f .env ]; then
    echo "ESP32_RTSP_URL=rtsp://192.168.2.20:8554/audio" > .env
    echo "[startup] .env aangemaakt met standaard ESP32 URL"
fi

# Stack opstarten
docker compose up -d
echo "[startup] mammal-watcher stack gestart"
