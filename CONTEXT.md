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
| 1 | ESP32 mic → RTSP stream → MediaMTX directe fan-out (ffmpeg bridge verwijderd) | ✅ WERKEND |
| 2 | YAMNet pre-filter (bird/mammal/human/vehicle) | ✅ WERKEND (vervangen door MammalCNN in stap 4) |
| 3 | EcoSound subcategorisatie | ⏳ UITGESTELD |
| 4 | Eigen MammalCNN soortherkenning (15 soorten op USB, training loopt) | 🔄 IN ONTWIKKELING |
| 5 | NatureLM zero-shot geavanceerde AI laag / BirdNET embeddings | ⏳ TOEKOMST |
| 6 | InfluxDB + Grafana dashboards | ⏳ TOEKOMST |
| 7 | n8n workflows (Telegram, Mastodon, dagrapport) | ⏳ TOEKOMST |

---

## Huidige status (per 2026-05-15)

### ✅ Wat werkt
- RTSP audio pipeline: ESP32 → MediaMTX directe fan-out → `rtsp://localhost:8554/mic`
- MediaMTX directe fan-out (ffmpeg-bridge verwijderd wegens instabiliteit)
- Reboot-safe startup via `startup.sh` + systemd unit
- Docker stack draait: `mammal-mediamtx`, `mammal-watcher`, `mammal-review-api`, `mammalradar-web`
- Clip opslag: `clips/confirmed` en `clips/uncertain`
- Review workflow: `web/review.html` + `review_api.py` (Flask→FastAPI)
- Feedback loop: `feedback_collector.py` voor active learning
- MQTT publisher: detecties → n8n
- mammalradar.net landing page via Cloudflare tunnel
- RTSP amplitude normalisatie fix (PR #23/#24) — meer detecties

### 🔄 In ontwikkeling (Stap 4)
- MammalCNN opnieuw trainen — vorig model was corrupt (sklearn Pipeline opgeslagen als .pt)
- Trainingsdata staat op USB `/mnt/usb/prepared/` — 15 soorten, 20.192 WAV clips
- Training draait via: `docker run --rm -v /mnt/usb:/mnt/usb:ro ...` in `screen -S training`
- Training script: `training/train.py` — CPU-only, class-weighted loss, augmentatie voor zeldzame soorten
- `--max-per-species 500` — voorkomt dominantie van vulpes_vulpes (10.618 clips!)

### ⚠️ Bekende problemen (opgelost tijdens sessie 2026-05-15)
- `mammal_cnn.pt` was corrupt: sklearn Pipeline opgeslagen onder PyTorch bestandsnaam
- Oorzaak: GPT/Sonnet sessie die alles herinstalleerde zonder expliciete opdracht
- Oplossing: opnieuw trainen met `training/train.py` op USB-data
- Stub classifier draaide → random output (wolf, bever, wezel binnen 1 minuut = nep)

### ⏳ Nog niet begonnen
- InfluxDB/Grafana (Stap 6)
- n8n automations (Stap 7)
- BirdNET embeddings als transfer learning basis (Stap 5, voor veldkastjes)

---

## Trainingsdata (USB `/mnt/usb`)

| Locatie | Inhoud |
|---------|--------|
| `/mnt/usb/prepared/` | WAV clips per soort (index.csv aanwezig) |
| `/mnt/usb/audio/` | Ruwe audio per soort |
| `/mnt/usb/features/` | 1024-dim .npy features (sklearn formaat, niet gebruikt door CNN) |

### Verdeling clips per soort (prepared/)
| Soort | Clips | Status |
|-------|-------|--------|
| vulpes_vulpes | 10.618 | ⚠️ dominant → gecapped op 500 |
| sus_scrofa | 4.673 | ⚠️ veel |
| sciurus_vulgaris | 1.542 | ok |
| capreolus_capreolus | 1.542 | ok |
| canis_lupus | 523 | ok |
| canis_aureus | 403 | ok |
| cervus_elaphus | 392 | ok |
| martes_foina | 93 | ⚠️ weinig → augmentatie |
| lutra_lutra | 90 | ⚠️ weinig → augmentatie |
| lynx_lynx | 83 | ⚠️ weinig → augmentatie |
| eliomys_quercinus | 67 | 🚨 kritiek → altijd augmentatie |
| felis_silvestris | 53 | 🚨 kritiek |
| castor_fiber | 48 | 🚨 kritiek |
| martes_martes | 33 | 🚨 kritiek |
| meles_meles | 32 | 🚨 kritiek |

---

## Technische beslissingen & waarom

| Beslissing | Reden |
|---|---|
| RTSP via MediaMTX i.p.v. directe audio | Stabielere stream, herbruikbaar door meerdere consumers |
| ffmpeg bridge verwijderd → MediaMTX direct | Bridge was instabiel met veel uitval, MediaMTX fan-out is robuuster |
| YAMNet als pre-filter vervangen door MammalCNN | Eigen model geeft betere soortspecifieke herkenning voor NL fauna |
| CPU-only PyTorch (geen GPU) | Hardware beperking: HP630/T630 heeft geen GPU |
| Classifier input altijd normaliseren (PR #23) | Lage RTSP amplitudes werden gemist zonder normalisatie |
| RTSP amplitude normalisatie ook in consumer | MediaMTX levert soms stille float32 frames, peak-norm vóór classificatie nodig |
| Data augmentatie + class-weighted loss (PR #14) | Class imbalance tussen soorten in trainingsdata |
| Zware CPU-augmentaties verwijderd (PR #15) | Te traag op CPU hardware |
| NatureLM (HuggingFace) + Freesound downloaders (PR #8) | Xeno-Canto had onvoldoende zoogdiergeluiden |
| GBIF + iNaturalist als databronnen (PR #10, #11) | Extra trainingsdata voor zeldzame soorten |
| Parquet column load + index checkpoint (PR #9) | NatureLM dataset te groot voor streaming scan |
| Pending-review feedback loop (PR #17) | Low-data soorten kunnen via menselijke review verbeteren |
| Review API beveiligd (PR #22) | Review surface was onbeschermd toegankelijk |
| --max-per-species 500 in training | Voorkomt dat vos (10.618 clips) het model domineert |
| Model opslaan met class_mapping + mel_params | classifier.py verwacht deze keys — zonder dit laadt model niet |
| Nooit AI (GPT/Sonnet) zonder toezicht laten herinstalleren | Sessie 15-05-2026: alles kapot gemaakt, sklearn model als .pt opgeslagen |

---

## Doelsoorten (15 soorten in huidige training)

| Soort | Wetenschappelijk | Prioriteit | Alert |
|---|---|---|---|
| Vos | Vulpes vulpes | Hoog | ✅ |
| Wolf | Canis lupus | Zeer hoog | ✅ |
| Goudjakhals | Canis aureus | Hoog | ✅ |
| Boommarter | Martes martes | Hoog | ✅ |
| Steenmarter | Martes foina | Middel | ✅ |
| Das | Meles meles | Hoog | ✅ |
| Otter | Lutra lutra | Zeer hoog | ✅ |
| Ree | Capreolus capreolus | Middel | ❌ |
| Edelhert | Cervus elaphus | Middel | ❌ |
| Wild zwijn | Sus scrofa | Laag | ❌ |
| Bever | Castor fiber | Hoog | ✅ |
| Lynx | Lynx lynx | Zeer hoog | ✅ |
| Eikelmuis | Eliomys quercinus | Middel | ❌ |
| Wilde kat | Felis silvestris | Hoog | ✅ |
| Rode eekhoorn | Sciurus vulgaris | Laag | ❌ |

---

## Bestandsstructuur

```
mammal-watcher/
├── mammal_watcher.py      # Hoofdproces: RTSP consumer + classificatie loop
├── classifier.py          # MammalCNN model + inferentie
├── rtsp_consumer.py       # RTSP audio inlezen (direct vanaf MediaMTX fan-out)
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
├── models/                # Model bestanden (mammal_cnn.pt hier opslaan!)
├── dataset/               # Trainingsdata scripts en downloaders
├── training/              # CNN training code (train.py)
├── web/index.html         # MammalRadar landing page
├── web/review.html        # Review UI voor needs_review clips
├── web/                   # Overige web assets (nginx)
├── n8n/                   # n8n workflow exports
├── systemd/               # systemd unit files
├── tests/                 # Pytest tests
├── docs/                  # Extra documentatie (COMMANDS.md)
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

## Training commando (referentie)

```bash
# Start altijd in screen!
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
```

---

## Volgende logische stappen (Stap 4 afronden)
1. ⏳ Wacht op training resultaat (duurt ~1-2 uur CPU)
2. Valideer model: accuracy per soort bekijken in training output
3. Docker stack herstarten zodat nieuw model geladen wordt
4. Controleer logs: `docker logs -f mammal-watcher` — echte detecties?
5. Dan: MQTT → n8n koppeling testen met echte detecties
6. Dan: InfluxDB/Grafana opzetten (Stap 6)
7. Dan: n8n Telegram alerts configureren (Stap 7)
8. Toekomst: BirdNET embeddings als transfer learning basis (veldkastjes)

---

## Toekomstige ideeën (nog niet opgepakt)

### Review page verbeteringen
- **"Opnieuw trainen" knop** op `web/review.html` → triggert `retrain.sh` via nieuwe API endpoint
- **Teller** hoeveel keer opnieuw getraind is (bijhouden in `models/retrain_log.json` of sidecar)
  - Bijv: `retrain #3 — 2026-05-16 — val_acc: 0.74 — 312 nieuwe feedback clips`
  - Zichtbaar op review page zodat je voortgang ziet
- Idee: drempelwaarde instellen → automatisch retrain starten als X nieuwe confirmed clips beschikbaar zijn

### Veldkastjes (toekomst)
- BirdNET embeddings als transfer learning basis → minder data nodig per soort
- Raspberry Pi / ESP32-CAM als veldkastje hardware
- Lokale opslag + periodieke sync naar n8nserver

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
- PR #24: RTSP amplitude fix + review UI bugfixes + CONTEXT.md update
- Commit: COMMANDS.md toegevoegd (iteratief cheatsheet)
- Commit: CONTEXT.md + COMMANDS.md update na debug sessie 15-05-2026
