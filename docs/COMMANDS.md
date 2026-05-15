# 🦊 MammalRadar — Commands cheatsheet

> Iteratief bijgehouden. Elke nieuwe handige command komt hier bij.  
> Laatste update: 2026-05-15

---

## 🐳 Stack beheer

```bash
# Start alles
docker compose up -d

# Start alles (verwijdert oude orphan containers)
docker compose up -d --remove-orphans

# Stop alles
docker compose down

# Stop alles én verwijder volumes
docker compose down -v

# Herstart één service
docker compose restart mammal-watcher
docker compose restart mammal-review-api
docker compose restart mammalradar-web

# Rebuild + herstart na code wijziging
docker compose up -d --build --remove-orphans

# Update images + herstart
docker compose pull && docker compose up -d --remove-orphans
```

---

## 📋 Logs

```bash
# Live logs per container
docker logs -f mammal-watcher
docker logs -f mammal-review-api
docker logs -f mammal-mediamtx
docker logs -f mammalradar-web

# Alle logs samen (laatste 50 regels)
docker compose logs -f --tail=50

# Logs van de laatste 10 minuten
docker compose logs --since=10m
```

---

## 📊 Status & monitoring

```bash
# Overzicht draaiende containers
docker compose ps

# CPU/RAM gebruik (snapshot)
docker stats --no-stream

# CPU/RAM gebruik (live)
docker stats
```

---

## 🎙️ RTSP / audio testen

```bash
# Test of de RTSP stream beschikbaar is via MediaMTX
ffprobe -rtsp_transport tcp \
  -i rtsp://localhost:8554/mic \
  -show_streams -select_streams a \
  -of json 2>&1 | grep -E '"codec|sample_rate|channels"'

# Luister 5 seconden audio via ffmpeg (headless check)
ffmpeg -rtsp_transport tcp -i rtsp://localhost:8554/mic \
  -t 5 -f null - 2>&1 | tail -5
```

---

## 🖥️ Screen (lange processen — voorkomt wegvallen bij SSH disconnect)

```bash
# Nieuwe screen sessie starten
screen -S training

# Detach (sessie blijft draaien op achtergrond)
Ctrl+A, D

# Weer terugkoppelen
screen -r training

# Overzicht actieve sessies
screen -ls

# Sessie forceren te sluiten
screen -X -S training quit
```

---

## 🔁 Training

```bash
# Training starten (altijd in screen!)
screen -S training

docker run --rm \
  -v /mnt/usb:/mnt/usb:ro \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/species_config.json:/app/species_config.json:ro \
  -v $(pwd)/training:/app/training:ro \
  mammal-watcher-mammal-watcher \
  python3 training/train.py \
    --data /mnt/usb/prepared/index.csv \
    --output /app/models \
    --epochs 30 \
    --batch-size 16 \
    --num-workers 2 \
    --max-per-species 500 \
    --augment

# Handmatig controleren welke feedback clips beschikbaar zijn
find feedback/needs_review -name "*.wav" | wc -l
find feedback/confirmed -name "*.wav" | wc -l

# Retraining pipeline (na feedback collectie)
bash retrain.sh
```

---

## 🔍 Model inspecteren

```bash
# Controleer of model geldig is (niet corrupt)
docker run --rm \
  -v $(pwd)/models:/app/models \
  mammal-watcher-mammal-watcher \
  python3 -c "
import torch
cp = torch.load('/app/models/mammal_cnn.pt', map_location='cpu', weights_only=False)
print('Keys:', list(cp.keys()))
print('Soorten:', cp.get('class_mapping', {}))
print('Datum:', cp.get('training_info', {}).get('trained_at', 'onbekend'))
print('Val acc:', cp.get('val_accuracy', 'onbekend'))
"

# USB data inspecteren via container
docker run --rm \
  -v /mnt/usb:/mnt/usb:ro \
  mammal-watcher-mammal-watcher \
  python3 -c "
import collections
import os
for soort in sorted(os.listdir('/mnt/usb/prepared/')):
    pad = f'/mnt/usb/prepared/{soort}'
    if os.path.isdir(pad):
        n = len([f for f in os.listdir(pad) if f.endswith('.wav')])
        print(f'{n:5d} {soort}')
"
```

---

## 🗂️ Git & deployment

```bash
# Laatste wijzigingen ophalen + rebuild
git pull && docker compose up -d --build --remove-orphans

# Huidige status repo
git log --oneline -10

# Bekijk welke bestanden gewijzigd zijn
git status
git diff --stat HEAD~1
```

---

## 🧹 Opruimen

```bash
# Verwijder gestopte / orphan containers
docker compose up -d --remove-orphans

# Verwijder alle ongebruikte Docker images
docker image prune -f

# Verwijder alle ongebruikte volumes (LET OP: data gaat verloren)
docker volume prune -f

# Schijfruimte Docker bekijken
docker system df
```

---

## 🔎 Troubleshooting

```bash
# Bekijk exit code van gestopte container
docker inspect mammal-watcher --format='{{.State.ExitCode}}'

# Shell in draaiende container
docker exec -it mammal-watcher bash
docker exec -it mammal-review-api bash

# Test review API rechtstreeks
curl http://localhost:8081/api/stats | python3 -m json.tool
curl http://localhost:8081/api/detections?limit=5 | python3 -m json.tool

# Model laadt niet? Controleer of het echt een PyTorch bestand is
xxd models/mammal_cnn.pt | head -3
# Eerste bytes moeten zijn: 80 02 of 80 04 (pickle/torch magic)
# NIET: 73 6b 6c (= 'skl' = sklearn!) 
```

---

> 💡 **Tip:** voeg nieuwe commands toe via een commit of meld ze in de Space — dan wordt dit bestand bijgewerkt.
