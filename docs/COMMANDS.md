# 🦊 MammalRadar — Commands cheatsheet

> Iteratief bijgehouden. Elke nieuwe handige command komt hier bij.  
> Laatste update: 2026-05-15

---

## 🐳 Stack beheer

```bash
# Start alles (clean, verwijdert oude orphan containers)
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

## 🔁 Retraining

```bash
# Hertraining pipeline starten
bash retrain.sh

# Handmatig controleren welke feedback clips beschikbaar zijn
find feedback/needs_review -name "*.wav" | wc -l
find feedback/confirmed -name "*.wav" | wc -l
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
```

---

> 💡 **Tip:** voeg nieuwe commands toe via een commit of meld ze in de Space — dan wordt dit bestand bijgewerkt.
