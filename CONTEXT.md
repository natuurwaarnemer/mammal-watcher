# 🦊 MammalRadar — Projectcontext voor Copilot

## Wat is dit project?
MammalRadar (repository: `mammal-watcher`) is een volledig **lokaal draaiend systeem** dat Nederlandse zoogdieren herkent op basis van geluid. Het systeem luistert 24/7 via een ESP32-microfoon, classificeert geluiden met een eigen PyTorch CNN-model en stuurt alerts via MQTT → n8n → Telegram.

Eigenaar: @natuurwaarnemer  
Taal: Python  
Hardware: ESP32, HP630/T630 (n8nserver), NUC2 (BirdNET-Pi), HP640 (InfluxDB/Grafana)

---

## Architectuur (7 stappen)

| Stap | Omschrijving | Status |
|------|-------------|--------|
| 1 | ESP32 mic → RTSP stream → MediaMTX → ffmpeg bridge | ✅ WERKEND |
| 2 | YAMNet pre-filter (bird/mammal/human/vehicle) | ✅ WERKEND (vervangen door MammalCNN in stap 4) |
| 3 | EcoSound subcategorisatie | ⏳ UITGESTELD |
| 4 | Eigen MammalCNN soortherkenning (23 NL soorten) | 🔄 IN ONTWIKKELING |
| 5 | NatureLM zero-shot geavanceerde AI laag | ⏳ TOEKOMST |
| 6 | InfluxDB + Grafana dashboards | ⏳ TOEKOMST |
| 7 | n8n workflows (Telegram, Mastodon, dagrapport) | ⏳ TOEKOMST |

---

## Huidige status (per 2026-05-15)

### ✅ Wat werkt
- RTSP audio pipeline: ESP32 → MediaMTX → ffmpeg-bridge → `rtsp://localhost:8554/mic`
- Reboot-safe startup via `startup.sh` + systemd unit
- **MammalCNN** draait lokaal (PyTorch, CPU-only op HP630/T630)
- 23 doelsoorten in training dataset
- Clip opslag: `clips/confirmed` en `clips/uncertain`
- Review workflow: `review_api.py` (Flask API) + web UI
- Feedback loop: `feedback_collector.py` voor active learning
- MQTT publisher: detecties → n8n
- Docker stack: `docker-compose.yml` met alle services
- nginx landing page onder `web/`

### 🔄 In ontwikkeling (Stap 4)
- MammalCNN model verbeteren met meer trainingsdata
- Balanced dataset: 10s clips per soort
- Retraining pipeline via `retrain.sh`

### ⏳ Nog niet begonnen
- InfluxDB/Grafana (Stap 6)
- n8n automations (Stap 7)

---

## Technische beslissingen & waarom

| Beslissing | Reden |
|---|---|
| RTSP via MediaMTX i.p.v. directe audio | Stabielere stream, herbruikbaar door meerdere consumers |
| ffmpeg bridge | Herlevert RTSP naar formaat dat Python consumers aankunnen |
| YAMNet als pre-filter vervangen door MammalCNN | Eigen model geeft betere soortspecifieke herkenning voor NL fauna |
| CPU-only PyTorch (geen GPU) | Hardware beperking: HP630/T630 heeft geen GPU |
| Classifier input altijd normaliseren (PR #23) | Lage RTSP amplitudes werden gemist zonder normalisatie |
| Data augmentatie + class-weighted loss (PR #14) | Class imbalance tussen soorten in trainingsdata |
| Zware CPU-augmentaties verwijderd (PR #15) | Te traag op CPU hardware |
| NatureLM (HuggingFace) + Freesound downloaders (PR #8) | Xeno-Canto had onvoldoende zoogdiergeluiden |
| GBIF + iNaturalist als databronnen (PR #10, #11) | Extra trainingsdata voor zeldzame soorten |
| Parquet column load + index checkpoint (PR #9) | NatureLM dataset te groot voor streaming scan |
| Pending-review feedback loop (PR #17) | Low-data soorten kunnen via menselijke review verbeteren |
| Review API beveiligd (PR #22) | Review surface was onbeschermd toegankelijk |
| 23 soorten balanced 10s clips (PR #22) | Van 12 naar 23 soorten uitgebreid |

---

## Doelsoorten (23 NL zoogdieren)

| Soort | Wetenschappelijk | Prioriteit | Alert |
|---|---|---|---|
| Vos | Vulpes vulpes | Hoog | ✅ |
| Das | Meles meles | Hoog | ✅ |
| Otter | Lutra lutra | Zeer hoog | ✅ |
| Wolf | Canis lupus | Zeer hoog | ✅ |
| Wezel | Mustela nivalis | Hoog | ✅ |
| Hermelijn | Mustela erminea | Hoog | ✅ |
| Steenmarter | Martes foina | Middel | ✅ |
| Boommarter | Martes martes | Hoog | ✅ |
| Ree | Capreolus capreolus | Middel | ❌ |
| Edelhert | Cervus elaphus | Middel | ❌ |
| Wild zwijn | Sus scrofa | Laag | ❌ |
| Bever | Castor fiber | Hoog | ✅ |

---

## Bestandsstructuur

```
mammal-watcher/
├── mammal_watcher.py      # Hoofdproces: RTSP consumer + classificatie loop
├── classifier.py          # MammalCNN model + inferentie
├── rtsp_consumer.py       # RTSP audio inlezen
├── mqtt_publisher.py      # Detecties publiceren via MQTT
├── feedback_collector.py  # Active learning feedback loop
├── review_api.py          # Flask API voor clip review
├── retrain.sh             # Retraining pipeline script
├── config.yaml            # Configuratie (drempelwaarden, MQTT, etc.)
├── species_config.json    # Per-soort configuratie
├── species_mammals_nl.csv # NL zoogdierenlijst
├── docker-compose.yml     # Volledige Docker stack
├── Dockerfile             # Mammal-watcher container
├── Dockerfile.api         # Review API container
├── startup.sh             # Reboot-safe startup script
├── mediamtx.yml           # MediaMTX configuratie
├── dataset/               # Trainingsdata scripts en downloaders
├── training/              # CNN training code
├── web/                   # nginx landing page
├── n8n/                   # n8n workflow exports
├── systemd/               # systemd unit files
├── tests/                 # Pytest tests
├── docs/                  # Extra documentatie
└── feedback/              # Opgeslagen feedback clips
```

---

## Trainingsdata bronnen
- **NatureLM** (HuggingFace) — grote audio dataset
- **Freesound** — aanvullende geluidsopnames
- **iNaturalist** — observaties met geluid
- **GBIF** — taxon-gekoppelde audio
- **Eigen veldopnames** via `clips/` map (active learning)

---

## Volgende logische stappen (Stap 4 afronden)
1. MammalCNN model valideren: accuracy per soort meten
2. Retraining pipeline testen met nieuwe feedback clips
3. MQTT → n8n koppeling testen met echte detecties
4. Dan: InfluxDB/Grafana opzetten (Stap 6)
5. Dan: n8n Telegram alerts configureren (Stap 7)

---

## PR geschiedenis (samenvatting)
- PR #1: Initiële scaffold
- PR #2-3: RTSP/MediaMTX/MQTT architecturele pivot
- PR #4: Werkende stack vastgezet (v0.3)
- PR #5: YAMNet inferentie + clip capture
- PR #6: Reboot-safe systemd startup
- PR #7: ROADMAP + dataset pipeline
- PR #8-11: Downloaders (NatureLM, Freesound, iNaturalist, GBIF)
- PR #12: Dataset pipeline fixes
- PR #13: PyTorch CNN training pipeline (12 soorten)
- PR #14: Data augmentatie + class-weighted loss
- PR #15: Zware augmentaties verwijderd + healthcheck
- PR #16: Hernoemd naar WildEar
- PR #17: Pending-review feedback loop
- PR #18: Hernoemd naar MammalRadar + landing page
- PR #19: Logo SVG (niet gemerged)
- PR #20: Dockerfile fix feedback_collector
- PR #21: YAMNet vervangen door lokale MammalCNN + review workflow
- PR #22: Review beveiligd + 23 soorten + balanced 10s clips
- PR #23: Classifier input normalisatie fix (lage RTSP amplitudes)
