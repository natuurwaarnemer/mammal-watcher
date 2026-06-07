# 🦊 MammalRadar — Projectcontext

## Wat is dit project?
MammalRadar (repository: `mammal-watcher`) is een volledig **lokaal draaiend systeem** dat Nederlandse zoogdieren herkent op basis van geluid. Een goedkoop veldapparaat detecteert aan de rand, de server identificeert de soort, AI analyseert het gedrag, ecologische data valideert de detectie — en bij een bijzondere vondst gaat er een alert naar een bioloog.

Eigenaar: @natuurwaarnemer
Taal: Python
Server: HP T630 (192.168.2.35) — ook n8n host
Veldapparaat (huidig): ESP32 microfoon op 192.168.2.20
Veldapparaat (nieuw): **Arduino Nicla Voice** (NDP120 Neural Decision Processor)

---

## 🎯 Visie

Een €45 kastje in de Biesbosch dat om 2:47 's nachts een otter detecteert, het gedrag herkent als "sociale roep", valideert dat otters hier voorkomen, en binnen 10 seconden een Telegram stuurt naar een bioloog. Volledig autonoom, open source, repliceerbaar door iedereen.

Vijf lagen:
1. **Edge detectie** — Nicla Voice herkent "klinkt als zoogdier" lokaal op het apparaat
2. **Soortherkenning** — BirdNET embeddings + PyTorch MLP op de T630 (15 NL soorten)
3. **Gedragsanalyse** — NatureLM: alarm / territorium / voortplanting / foerageren
4. **Plausibiliteit** — GBIF/NDFF: klopt deze soort op deze locatie en dit seizoen?
5. **Citizen science** — onverwachte detecties (hoge confidence, onbekende locatie) → Zoogdiervereniging/NDFF

---

## 🏗️ Architectuur — de omslag

### Oud (huidig, werkt niet goed)
```
ESP32 mic → RTSP stream (24/7) → T630 → BirdNET → MLP classifier
                                              ↑
                          PROBLEEM: geen background klasse →
                          model classificeert wind/regen/vogels
                          ook als zoogdier → veel false positives
```

### Nieuw (doel)
```
[Nicla Voice — in het veld]
  NDP120 model: "klinkt dit als een zoogdier?"
  NEE → niks doen (99% van de tijd)
  JA  → stuur audioclip + metadata naar T630

[T630 — server pipeline]
  BirdNET embeddings → MLP → welke soort?
  NatureLM → wat doet het dier?
  GBIF/NDFF → klopt dit ecologisch?
  n8n → Telegram / NDFF melding
```

**Waarom dit de omslag is:** het background-klasse probleem wordt opgelost op hardwareniveau.
De server ziet alleen audio die al een eerste filter heeft doorstaan.
Meerdere veldkastjes mogelijk voor ~€45 per stuk. Batterijgevoede inzet in het veld.

---

## 🔧 Huidige status (2026-06-07)

### ✅ Wat werkt
- Docker stack draait: mediamtx, mammal-watcher, review-api, mammalradar-web
- RTSP pipeline: ESP32 → MediaMTX → mammal-watcher
- Review workflow: web/review.html + review_api.py
- Feedback loop: feedback_collector.py
- MQTT publisher (verbonden, maar geen detecties)
- mammalradar.net via Cloudflare tunnel
- BirdNET MLP model getraind (val_acc 0.49, 20k embeddings, 15 soorten)

### 🚨 Huidig kapot — PRIORITEIT 1
**Fout:** `mat1 and mat2 shapes cannot be multiplied (1x1024 and 6522x512)`

- `EMBEDDING_DIM = 6522` hardcoded in `training/train_mlp.py`
- BirdNET produceert op runtime **1024-dim** embeddings
- Model is getraind op verkeerde dimensie (waarschijnlijk BirdNET versie-mismatch)
- Fix: controleer dimensie van embeddings op /mnt/usb/embeddings/, pas EMBEDDING_DIM aan, hertrainen

### ⏳ Nog niet begonnen
- Nicla Voice integratie (nieuwe edge architectuur)
- NatureLM gedragsanalyse (Stap 5)
- InfluxDB/Grafana (Stap 6)
- n8n Telegram alerts (Stap 7)
- Plausibiliteitslaag NDFF/GBIF (Stap 5)

---

## 📋 Roadmap — in volgorde

| Stap | Wat | Waarom eerst |
|------|-----|-------------|
| **1** | Fix dim-mismatch: controleer embedding-dimensie op disk, pas EMBEDDING_DIM aan, hertrainen | Model produceert nu nul detecties |
| **2** | Bewijs dat model werkt: check docker logs op echte detecties | Baseline vaststellen vóór architectuurwijziging |
| **3** | Nicla Voice: edge model trainen (binair: zoogdier / niet) via Edge Impulse | Vervangt ESP32, lost background-probleem op |
| **4** | Server pipeline aanpassen op Nicla Voice input | Geen RTSP meer, clip-gebaseerde verwerking |
| **5** | n8n Telegram alerts voor tier-1 soorten (wolf, otter, das, lynx) | Eerste bruikbaar eindresultaat |
| **6** | NatureLM gedragsanalyse | Tweede AI-laag |
| **7** | GBIF/NDFF plausibiliteitslaag | Derde AI-laag |
| **8** | InfluxDB/Grafana dashboards | Historische data |
| **9** | Citizen science koppeling NDFF | Einddoel |

---

## ⚙️ Werkregels (strikt — geleerd uit incident 15-05-2026)

| Type wijziging | Aanpak |
|---|---|
| Content (CONTEXT.md, README, docs, HTML tekst) | Direct commit op main ✅ |
| Code (`.py`, `Dockerfile`, `docker-compose.yml`, scripts) | **PR verplicht** 🔄 |
| Na `git pull` met codewijzigingen | Altijd `docker compose build --no-cache mammal-watcher` |
| Docker build/rebuild | **Altijd door gebruiker zelf uitvoeren** |
| Infrastructuur wijzigen | **Nooit zonder expliciete opdracht** |

### Regels voor Claude specifiek
- **Lees CONTEXT.md aan het begin van elke sessie** — dit is de enige bron van waarheid
- **Één stap tegelijk** — niet meerdere problemen tegelijk aanpakken
- **Zeg het als je iets niet weet** — niet speculeren en code schrijven
- **Geen autonome rebuilds** — code aanpassen ja, docker uitvoeren nee
- **PR beschrijving bevat altijd:** wat doet het, waarom, hoe te testen

### Wat er mis ging op 15-05-2026
AI heeft zonder toezicht containers herbouwd en hergeïnstalleerd:
- sklearn model opgeslagen als `.pt` bestand (verkeerd formaat)
- Stack kapot gemaakt die daarna handmatig gerepareerd moest worden
- **Conclusie:** Claude schrijft code en PRs, gebruiker beslist over deployment

---

## 🗂️ Bestandsstructuur (relevant)

```
mammal-watcher/
├── mammal_watcher.py       # Hoofdproces: RTSP consumer + classificatie loop
├── classifier.py           # MammalCNN/MLP model + inferentie
├── rtsp_consumer.py        # RTSP audio inlezen
├── mqtt_publisher.py       # Detecties via MQTT
├── feedback_collector.py   # Active learning
├── review_api.py           # FastAPI clip review
├── config.yaml             # Drempelwaarden, MQTT, paden
├── species_config.json     # Per-soort configuratie
├── training/
│   ├── train_mlp.py        # ← EMBEDDING_DIM=6522 BUG HIER
│   ├── train.py            # CNN training (fallback)
│   └── extract_embeddings.py
├── models/
│   ├── mammal_mlp.pt       # Huidig model (kapot door dim-mismatch)
│   └── mammal_cnn.pt       # Fallback
├── web/                    # mammalradar.net frontend
├── n8n/                    # Workflow exports
└── tests/                  # Pytest tests
```

---

## 📊 Model accuraatheid (laatste training, 2026-05-17)

| Soort | Accuracy | Status |
|---|---|---|
| castor_fiber | 1.00 | ✅ |
| lynx_lynx | 0.92 | ✅ |
| felis_silvestris | 0.86 | ✅ |
| lutra_lutra | 0.83 | ✅ |
| martes_foina | 0.81 | ✅ |
| meles_meles | 0.75 | ✅ |
| martes_martes | 0.71 | ✅ |
| capreolus_capreolus | 0.50 | ⚠️ |
| canis_lupus | 0.58 | ⚠️ |
| vulpes_vulpes | 0.36 | ⚠️ oorzaak onbekend |
| sus_scrofa | 0.09 | 🚨 verward met ree |
| **Overall** | **0.49** | dim-mismatch maakt dit nu irrelevant |

---

## 🖥️ Trainingsdata (USB /mnt/usb op T630)

| Map | Inhoud | Gebruik |
|-----|--------|---------|
| `/mnt/usb/prepared/` | WAV clips per soort + index.csv | Bron voor embeddings |
| `/mnt/usb/embeddings/` | BirdNET .npy + embeddings_index.csv | MLP training input |
| `/mnt/usb/features/` | 1024-dim sklearn .npy | **Niet gebruiken** |

Grote klasse-imbalans: vulpes_vulpes 10.618 clips → gecapped op 500 via `--max-per-species`

---

## 🔗 Integraties

| Service | Adres | Status |
|---------|-------|--------|
| MQTT broker (HA) | homeassistant:1883 | ✅ verbonden |
| n8n | localhost:5678 | ✅ draait, workflow nog niet actief |
| BirdNET-Go (NUC2) | 192.168.2.23 | ✅ vogels apart |
| mammalradar.net | Cloudflare tunnel | ✅ live |
| Tierstimmen Archiv | — | ⏳ toestemming aangevraagd, niet gebruiken |

---

## 🚀 Volgende sessie — begin hier

1. Controleer embedding-dimensie op disk:
```bash
python3 -c "import numpy as np; a=np.load('/mnt/usb/embeddings/embeddings_index.csv'.replace('embeddings_index.csv','') + 'vulpes_vulpes_0001.npy'); print(a.shape)"
```
Of via docker:
```bash
docker run --rm -v /mnt/usb:/mnt/usb mammal-watcher-mammal-watcher \
  python3 -c "import numpy as np, pathlib; f=next(pathlib.Path('/mnt/usb/embeddings').rglob('*.npy')); print(np.load(f).shape)"
```

2. Pas `EMBEDDING_DIM` aan in `training/train_mlp.py`
3. PR aanmaken → gebruiker merget → gebruiker hertraint → logs checken op detecties

---

## 🛒 Hardware — wat bestellen

### Architectuur veldkastje
```
[Omgeving / geluid]
        ↓
[Nicla Voice]
  - NDP120 + mic
  - Edge Impulse model A (soort + confidence)
        ↓ UART: {"species":"meles_meles","conf":0.87}
[MCU / Node]
  - ESP32 of LilyGo T-SIM7600E
  - GSM/4G modem
  - SD (audio-buffer)
        ↓
[GSM netwerk]
        ↓
[T630 / Server]
  - Ontvangst API / MQTT
  - Opslag audio-snippets
  - NatureLM (model B: gedragsanalyse)
        ↓
[HA / n8n]
  - Alerts, logging, visualisatie
```

### Fase 1 — Prototype (WiFi, thuis/tuin testen)

| Component | Doel | Prijs |
|---|---|---|
| Arduino Nicla Voice | NDP120 + mic, edge model | ~€47 |
| ESP32-WROOM-32 devboard | WiFi bridge + audio buffer | ~€8 |
| Micro SD module + 32GB kaart | Audio snippets opslaan | ~€10 |
| LiPo 3.7V 3000mAh | Stroom | ~€10 |
| TP4056 charger module | Laden via USB | ~€2 |
| Dupont/JST kabels | Verbindingen | ~€5 |
| **Totaal fase 1** | | **~€82** |

### Fase 2 — Veld (GSM, waterproof, autonoom)

Bovenop fase 1 (vervang losse ESP32 door LilyGo):

| Component | Doel | Prijs |
|---|---|---|
| LilyGo T-SIM7600E | ESP32 + 4G in één board — vervangt losse ESP32 | ~€35 |
| Simbase SIM | Pay-per-MB, NL/EU dekking | ~€5 + gebruik |
| Zonnepaneel 5W + laadregelaar | Onbeperkt autonoom in veld | ~€15 |
| IP67 behuizing 150x100x75mm | Weatherproof kastje | ~€12 |
| Kabelwartels M12 | Waterdicht kabelinvoer | ~€5 |
| Windkap voor microfoon | Windruis onderdrukken | ~€4 |
| **Extra fase 2** | | **~€76** |

**Let op:** LilyGo T-SIM7600E vervangt de losse ESP32 uit fase 1 — niet dubbel bestellen.

### Verbinding Nicla Voice ↔ ESP32
- Protocol: **UART** (TX/RX) — betrouwbaarder dan BLE voor continue werking
- Nicla Voice stuurt bij detectie: `{"species":"meles_meles","conf":0.87,"trigger":true}`
- ESP32 slaat bijbehorend audioclip op SD op + stuurt pakket door via WiFi/4G

### Edge Impulse
Toolchain voor Nicla Voice model training. Gratis tier is ruim voldoende.
Workflow: trainingsdata uploaden → model trainen → deployen naar Nicla Voice.
Aparte toolchain van de Python/PyTorch server-pipeline.
