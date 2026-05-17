# 🦊 MammalRadar — Projectcontext voor Copilot

## Wat is dit project?
MammalRadar (repository: `mammal-watcher`) is een volledig **lokaal draaiend systeem** dat Nederlandse zoogdieren herkent op basis van geluid. Het systeem luistert 24/7 via een ESP32-microfoon, classificeert geluiden met een eigen PyTorch MLP-model (BirdNET embeddings) en stuurt alerts via MQTT → n8n → Telegram.

Eigenaar: @natuurwaarnemer  
Taal: Python  
Hardware: ESP32, HP630/T630 (n8nserver), NUC2 (BirdNET-Pi), HP640 (InfluxDB/Grafana)

---

## 🎯 Visie

MammalRadar is meer dan een detector. Het doel is een **volledig ecologisch monitoringssysteem** voor Nederlandse zoogdieren:

1. **Detectie** — welk zoogdier is er?
2. **Plausibiliteitslaag** — klopt deze detectie gezien de locatie en het seizoen?
3. **Gedragsanalyse** — wat doet het dier? (alarm, territorium, voortplanting)
4. **Tijdspatronen** — wanneer is welke soort actief? (nacht, seizoen, weer)
5. **Citizen science** — onverwachte detecties melden aan Zoogdiervereniging/NDFF

### 🗺️ Plausibiliteitslaag (toekomstige stap)
Elke detectie wordt getoetst aan verspreidingsdata:
- **Bronnen**: NDFF, Zoogdiervereniging atlasdata, waarneming.nl API, GBIF (al in stack)
- **Logica**:
  - Normale detectie + bekende locatie → gewone alert
  - Onverwachte locatie + lage confidence → waarschijnlijk fout-positief → wegfilteren
  - Onverwachte locatie + hoge confidence (>0.95) → 🚨 **mogelijk wetenschappelijk interessant** → melding Zoogdiervereniging
- **Voorbeeld**: eikelmuis in Drenthe (alleen Zuid-Limburg bekend) met conf=0.91 → bijzondere melding

---

## Werkregels (belangrijk voor Copilot)

| Type wijziging | Aanpak |
|---|---|
| Content (README, CONTEXT.md, docs, HTML tekst) | Direct commit op main ✅ |
| Code (`.py`, `Dockerfile`, `docker-compose.yml`, scripts) | PR verplicht 🔄 |
| Na `git pull` met codewijzigingen | Altijd `docker compose build --no-cache mammal-watcher` |
| Nooit | AI zonder toezicht laten herinstalleren/rebuilden |

---

## Architectuur (7 stappen)

| Stap | Omschrijving | Status |
|------|-------------|--------|
| 1 | ESP32 mic → RTSP stream → MediaMTX directe fan-out (ffmpeg bridge verwijderd) | ✅ WERKEND |
| 2 | YAMNet pre-filter (bird/mammal/human/vehicle) | ✅ WERKEND (vervangen door MammalCNN in stap 4) |
| 3 | EcoSound subcategorisatie | ⏳ UITGESTELD |
| 4 | BirdNET embeddings + PyTorch MLP soortherkenning (15 soorten) | ✅ WERKEND (val_acc 0.49) |
| 5 | NatureLM zero-shot geavanceerde AI laag / BirdNET embeddings | ⏳ TOEKOMST |
| 6 | InfluxDB + Grafana dashboards | ⏳ TOEKOMST |
| 7 | n8n workflows (Telegram, Mastodon, dagrapport) | ⏳ TOEKOMST |

---

## Huidige status (per 2026-05-17)

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
- **BirdNET MLP model getraind** (PR #25): val_acc 0.49, 20k embeddings, 15 soorten

### 📊 MLP model resultaten (2026-05-17, --max-per-species 500)
| Soort | Accuracy | Opmerking |
|---|---|---|
| castor_fiber | 1.00 | 🏆 |
| lynx_lynx | 0.92 | ✅ |
| felis_silvestris | 0.86 | ✅ |
| eliomys_quercinus | 0.86 | ✅ |
| lutra_lutra | 0.83 | ✅ |
| martes_foina | 0.81 | ✅ |
| martes_martes | 0.71 | ✅ |
| meles_meles | 0.75 | ✅ |
| canis_aureus | 0.58 | ⚠️ verward met wolf/vos |
| canis_lupus | 0.58 | ⚠️ verward met vos/aureus |
| capreolus_capreolus | 0.50 | ⚠️ |
| cervus_elaphus | 0.55 | ⚠️ |
| sciurus_vulgaris | 0.52 | ⚠️ |
| vulpes_vulpes | 0.36 | ⚠️ oorzaak onbekend — veel variatie in geluiden? |
| sus_scrofa | 0.09 | 🚨 verward met capreolus |
| **Overall** | **0.49** | Van 0.20 → 0.49 na --max-per-species fix |

### 🔄 In ontwikkeling
- Docker image rebuild na PR #25 (train_mlp.py --max-per-species fix)
- Container laadt nog stub fallback — rebuild in progress
- MQTT → n8n koppeling testen met echte detecties

### ⚠️ Bekende aandachtspunten
- `mammal_cnn.pt` was corrupt (sessie 15-05-2026): sklearn model als .pt opgeslagen door AI zonder toezicht
- Vos accuracy laag (0.36): oorzaak onbekend, **niet speculeren over databron** — nog uit te zoeken
- Wild zwijn accuracy laag (0.09): verward met ree — mogelijk akoestische overlap
- Tierstimmen Archiv: toestemming aangevraagd per mail, **niet gebruiken tot bevestiging**

### ⏳ Nog niet begonnen
- InfluxDB/Grafana (Stap 6)
- n8n automations (Stap 7)
- Plausibiliteitslaag NDFF/GBIF (Stap 5)

---

## Trainingsdata (USB `/mnt/usb`)

| Locatie | Inhoud |
|---------|--------|
| `/mnt/usb/prepared/` | WAV clips per soort (index.csv aanwezig) |
| `/mnt/usb/audio/` | Ruwe audio per soort |
| `/mnt/usb/features/` | 1024-dim .npy features (sklearn formaat, niet gebruikt door MLP) |
| `/mnt/usb/embeddings/` | BirdNET .npy embeddings (20.208 bestanden) + embeddings_index.csv |

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
| sklearn vervangen door pure PyTorch/numpy | Minder dependencies, consistenter met rest van stack |
| BirdNET embeddings als feature extractor | Transfer learning: rijke audio representaties, weinig trainingsdata nodig per soort |
| Tierstimmen Archiv niet gebruiken | Toestemming aangevraagd maar nog niet verkregen — in afwachting |

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
├── classifier.py          # MammalCNN/MLP model + inferentie
├── rtsp_consumer.py       # RTSP audio inlezen (direct vanaf MediaMTX fan-out)
├── mqtt_publisher.py      # Detecties publiceren via MQTT
├── feedback_collector.py  # Active learning feedback loop
├── review_api.py          # FastAPI voor clip review
├── retrain.sh             # Retraining pipeline script
├── config.yaml            # Configuratie (drempelwaarden, MQTT, etc.)
├── species_config.json    # Per-soort configuratie
├── species_mammals_nl.csv # NL zoogdierenlijst
├── docker-compose.yml     # Volledige Docker stack
├── Dockerfile             # Mammal-watcher container
├── Dockerfile.api         # Review API container
├── startup.sh             # Reboot-safe startup script
├── mediamtx.yml           # MediaMTX configuratie
├── models/                # Model bestanden (mammal_mlp.pt, mammal_cnn.pt fallback)
├── dataset/               # Trainingsdata scripts en downloaders
├── training/              # MLP + CNN training code
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
- **Tierstimmen Archiv** — toestemming aangevraagd, nog in afwachting

---

## Training commando (referentie)

```bash
# BirdNET embeddings aanpak (aanbevolen — sneller en robuuster)

# Stap 1: embeddings uitrekenen (eenmalig, ~30 min voor 20k clips)
screen -S embeddings
docker run --rm \
  -v /mnt/usb:/mnt/usb \
  -v $(pwd)/training:/app/training:ro \
  mammal-watcher-mammal-watcher \
  python3 training/extract_embeddings.py \
    --data /mnt/usb/prepared/index.csv \
    --embeddings-dir /mnt/usb/embeddings

# Stap 2: MLP trainen op embeddings (~5-10 min op CPU!)
screen -S training
docker run --rm \
  -v /mnt/usb:/mnt/usb:ro \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/species_config.json:/app/species_config.json:ro \
  -v $(pwd)/training:/app/training:ro \
  mammal-watcher-mammal-watcher \
  python3 training/train_mlp.py \
    --embeddings-dir /mnt/usb/embeddings/embeddings_index.csv \
    --output /app/models \
    --epochs 100 \
    --patience 10 \
    --max-per-species 500

# CNN aanpak (fallback, nog steeds ondersteund)
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

## Volgende logische stappen
1. ✅ BirdNET embeddings uitgerekend (20.208 clips)
2. ✅ MLP getraind (val_acc 0.49, --max-per-species 500)
3. 🔄 Docker image rebuilden zodat nieuw MLP model geladen wordt
4. ⏳ Controleer logs: `docker logs -f mammal-watcher` — echte detecties?
5. ⏳ MQTT → n8n koppeling testen met echte detecties
6. ⏳ InfluxDB/Grafana opzetten (Stap 6)
7. ⏳ n8n Telegram alerts configureren (Stap 7)
8. ⏳ Plausibiliteitslaag NDFF/GBIF bouwen (Stap 5)

---

## Toekomstige ideeën (nog niet opgepakt)

### Plausibiliteitslaag
- NDFF + Zoogdiervereniging atlasdata per soort per km-hok
- waarneming.nl API voor recente meldingen in regio
- Logica: onverwachte locatie + hoge confidence → citizen science melding
- Scenario: eikelmuis in Drenthe met conf>0.95 → melding Zoogdiervereniging

### Gedragsanalyse & tijdspatronen
- Tijdstip activiteit per soort bijhouden (nachts, seizoen)
- Weersdata koppelen (meer activiteit na regen?)
- Triangulatie via meerdere kastjes (waar precies?)
- Individu-herkenning (zelfde wolf terugzien?)

### Review page verbeteringen
- **"Opnieuw trainen" knop** op `web/review.html` → triggert `retrain.sh` via nieuwe API endpoint
- **Teller** hoeveel keer opnieuw getraind is (bijhouden in `models/retrain_log.json`)
- Automatisch retrain starten als X nieuwe confirmed clips beschikbaar zijn

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
- PR #25: train_mlp.py --max-per-species parameter toegevoegd
- Commit: COMMANDS.md toegevoegd (iteratief cheatsheet)
- Commit: CONTEXT.md updates (visie, plausibiliteitslaag, werkregels, status 17-05-2026)
