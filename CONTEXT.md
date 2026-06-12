# 🦊 MammalRadar — Projectcontext

## Wat is dit project?
MammalRadar (repository: `mammal-watcher`) is een volledig **lokaal draaiend systeem** dat Nederlandse zoogdieren herkent op basis van geluid.

Eigenaar: @natuurwaarnemer | Taal: Python
Server: HP T630 (192.168.2.35) — ook n8n host
Veldapparaat (huidig): ESP32 microfoon op 192.168.2.20
Veldapparaat (nieuw): **Arduino Nicla Voice** (NDP120 Neural Decision Processor)

---

## 🎯 Visie

Een €45 kastje in de Biesbosch dat om 2:47 's nachts een otter detecteert, het gedrag herkent als "sociale roep", valideert dat otters hier voorkomen, en binnen 10 seconden een Telegram stuurt naar een bioloog.

Vijf lagen:
1. **Edge detectie** — Nicla Voice herkent "klinkt als zoogdier" lokaal
2. **Soortherkenning** — BirdNET embeddings + PyTorch MLP op T630 (15 doelsoorten + background)
3. **Gedragsanalyse** — NatureLM
4. **Plausibiliteit** — GBIF/NDFF
5. **Citizen science** — Zoogdiervereniging/NDFF

---

## 🔧 Huidige status (2026-06-12)

### ✅ Wat werkt
- Docker stack: mediamtx, mammal-watcher, review-api, mammalradar-web
- RTSP pipeline: ESP32 → MediaMTX → mammal-watcher
- Review interface: mammalradar.net/review
- BirdNET-Go op NUC2 (192.168.2.23) — 267k+ vogel-detecties, HTTP API bereikbaar

### ✅ Gefixed in sessie 2026-06-12
- **Tensor bug (Copilot PR #32):** OUTPUT_TENSOR_INDEX 546→545, EMBEDDING_DIM 6522→1024
- **Background-klasse:** 18 niet-doelsoorten uit prepared/ mappen naar `background` label
- **Docker:** torch CPU-only, tensorflow-cpu==2.15.1 (2.16+ blokkeert tensor 545 met XNNPACK)
- **BirdNET-Go download script:** dataset/download_birdnet_clips.py — vogel-WAVs via NUC2 HTTP API
- **248 achtergrond-clips** gedownload (65 vogelsoorten, 16kHz WAV, zelfde microfoon)
- **min_confidence:** 0.95 → 0.70

### 🔄 Nu bezig (2026-06-12 nacht)
**Embedding herextractie loopt overnight** (~13 uur, klaar ~09:00 op 2026-06-13):
```bash
screen -r mammal-watcher   # voortgang bekijken
```

---

## 🚀 Volgende sessie — begin hier

### Stap 1: controleer of extractie klaar is
```bash
screen -r mammal-watcher
# Als klaar: "Klaar! X embeddings opgeslagen"
```

### Stap 2: hertrainen (~10 min)
```bash
docker run --rm \
  -v ~/mammal-watcher:/app \
  -v /mnt/usb:/mnt/usb \
  mammal-watcher-mammal-watcher \
  python3 training/train_mlp.py \
    --embeddings-dir /mnt/usb/embeddings/embeddings_index.csv \
    --output /app/models \
    --max-per-species 500
```

### Stap 3: container herstarten
```bash
cd ~/mammal-watcher && docker compose restart mammal-watcher
```

### Stap 4: eerste detecties valideren
```bash
docker logs -f mammal-watcher
```
Browser: mammalradar.net/review

### Daarna: Nicla Voice fase
Trainingsdata (16kHz WAV) staat klaar in /mnt/usb/prepared/ — direct bruikbaar voor Edge Impulse.

---

## 🔑 Technische details (geleerd 2026-06-12)

### BirdNET FP32 model tensor indices
- **Tensor 545:** GLOBAL_AVG_POOL/Mean — shape (1, 1024) — **de embedding**
- **Tensor 546:** Identity — shape (1, 6522) — soortklassificatie output
- TF >= 2.16 maakt tensor 545 ontoegankbaar met XNNPACK → gepind op tensorflow-cpu==2.15.1
- `resize_tensor_input(0, [1, 144000])` + `allocate_tensors()` vereist vóór gebruik

### Docker training commando (altijd met repo mount!)
```bash
docker run --rm -v ~/mammal-watcher:/app -v /mnt/usb:/mnt/usb mammal-watcher-mammal-watcher python3 training/...
```
Training scripts zitten NIET in de image — repo mount is verplicht.

### BirdNET-Go NUC2 API
- Clips: `GET http://192.168.2.23:8080/api/v2/audio/{id}` → audio/mp4 (AAC)
- Detecties: `GET http://192.168.2.23:8080/api/v2/detections?limit=1000&offset=N`
- Max 1000 per request, geen auth op lokaal netwerk
- SSH naar NUC2 niet ingesteld vanaf T630

### Background-klasse samenstelling
- 18 niet-doelsoorten in BACKGROUND_SPECIES (train_mlp.py) → automatisch hernoemd
- 248 vogelclips van NUC2, gelabeld als `background` in prepared/index.csv
- Cap: 500 clips via --max-per-species

### Sample rates
- WAV-bestanden prepared/: 16kHz (GBIF) of 22kHz (Tierstimmen)
- BirdNET: librosa resampelt intern naar 48kHz
- Nicla Voice / Edge Impulse: 16kHz — zelfde WAVs direct bruikbaar

---

## 📋 Roadmap

| Stap | Wat | Status |
|------|-----|--------|
| 1 | Fix tensor + dim-mismatch | ✅ |
| 2 | Background-klasse | ✅ |
| 3 | Embeddings herextraheren | 🔄 loopt |
| 4 | Hertrainen MLP | ⏳ |
| 5 | Eerste detecties valideren | ⏳ |
| 6 | Nicla Voice edge model (Edge Impulse) | ⏳ |
| 7 | n8n Telegram alerts tier-1 soorten | ⏳ |
| 8 | NatureLM gedragsanalyse | ⏳ |
| 9 | GBIF/NDFF plausibiliteitslaag | ⏳ |
| 10 | Citizen science NDFF | ⏳ |

---

## ⚙️ Werkregels (strikt — geleerd uit incident 15-05-2026)

| Type wijziging | Aanpak |
|---|---|
| Content (CONTEXT.md, docs, HTML) | Direct commit op main ✅ |
| Code (.py, Dockerfile, docker-compose.yml) | **PR verplicht** 🔄 |
| Docker build/rebuild | **Altijd door gebruiker zelf uitvoeren** |
| Infrastructuur | **Nooit zonder expliciete opdracht** |

### Regels voor Claude
- Lees CONTEXT.md aan het begin van elke sessie
- Één stap tegelijk
- Geen autonome rebuilds
- Docker run voor training altijd met `-v ~/mammal-watcher:/app`

---

## 🗂️ Bestandsstructuur

```
mammal-watcher/
├── mammal_watcher.py
├── classifier.py
├── config.yaml                    # min_confidence: 0.70
├── species_config.json            # 15 soorten + felis_catus + homo_sapiens + background
├── dataset/
│   └── download_birdnet_clips.py  # vogel-clips van NUC2 API
├── training/
│   ├── train_mlp.py               # EMBEDDING_DIM=1024, BACKGROUND_SPECIES mapping
│   └── extract_embeddings.py      # OUTPUT_TENSOR_INDEX=545, EMBEDDING_DIM=1024
├── models/mammal_mlp.pt           # vervangen na hertraining
└── Dockerfile                     # torch CPU-only, tensorflow-cpu==2.15.1
```

---

## 🖥️ Trainingsdata (USB /mnt/usb op T630)

| Map | Inhoud |
|-----|--------|
| /mnt/usb/prepared/ | 35 soorten WAV (23.856 entries in index.csv) |
| /mnt/usb/prepared/background/ | 248 vogel-WAVs van NUC2 |
| /mnt/usb/embeddings/ | .npy embeddings (worden nu herextraheerd) |

---

## 🔗 Integraties

| Service | Adres | Status |
|---------|-------|--------|
| MQTT broker (HA) | homeassistant:1883 | ✅ |
| n8n | localhost:5678 | ✅ draait |
| BirdNET-Go (NUC2) | 192.168.2.23:8080 | ✅ HTTP API |
| mammalradar.net | Cloudflare tunnel | ✅ live |

---

## 🛒 Hardware — Nicla Voice prototype (~€82)

| Component | Doel | Prijs |
|---|---|---|
| Arduino Nicla Voice | NDP120 + mic | ~€86 |
| ESP32-WROOM-32 | WiFi bridge | ~€8 |
| SD module + 32GB | Audio buffer | ~€10 |
| LiPo 3.7V 3000mAh | Stroom | ~€10 |
| TP4056 charger | Laden | ~€2 |
| Kabels | Verbindingen | ~€5 |

Fase 2 (veld, GSM): +€76 met LilyGo T-SIM7600E, zonnepaneel, IP67 behuizing.
