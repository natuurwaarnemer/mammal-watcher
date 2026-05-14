# 🗺️ MammalRadar — Architectuuroverzicht

Dit document geeft een volledig overzicht van alle componenten, datastromen en
integraties van het MammalRadar-systeem. Het wordt bijgehouden naast de code.

---

## Huidige opstelling (v1 — Centraal op HP T630)

```mermaid
graph TD
    subgraph Hardware
        ESP32["🎙️ ESP32-microfoon\n192.168.2.20:8554/audio"]
    end

    subgraph T630["🖥️ HP T630 — n8nserver (192.168.2.35)"]
        subgraph Stack["Docker Compose stack"]
            MTX["mediamtx\nRTSP relay\n:8554/mic"]
            BRIDGE["rtsp-bridge\nffmpeg\nESP32 → MediaMTX"]
            MW["mammal-watcher\nPython classifier\nMammalCNN model"]
            WEB["mammalradar-web\nnginx :8080\nmammalradar.net"]
            API["review-api\nFastAPI :8081"]
        end

        subgraph Volumes["Volumes op schijf"]
            MODEL["models/mammal_cnn.pt"]
            CLIPS["clips/\nalle detecties"]
            FEEDBACK["feedback/\nneeds_review / confirmed / rejected"]
        end

        N8N["n8n\nlocalhost:5678"]
    end

    subgraph NUC["💻 NUC (192.168.2.23)"]
        BIRDNET["BirdNET-Go\nvogeldetectie"]
    end

    subgraph Integraties
        HA["🏠 Home Assistant\nMQTT sensor"]
        TG["📱 Telegram\nalerts"]
        INFLUX["📊 InfluxDB\nhistoriek"]
    end

    subgraph Internet
        CF["☁️ Cloudflare Tunnel"]
        MAMMALRADAR["🌐 mammalradar.net"]
    end

    ESP32 -->|RTSP stream| MTX
    MTX -->|RTSP relay| BRIDGE
    MTX -->|RTSP relay| MW
    MTX -->|RTSP relay| BIRDNET
    MW -->|detectie opslaan| CLIPS
    MW -->|onzekere detectie| FEEDBACK
    MW -->|MQTT publish| HA
    MW -->|MQTT publish| N8N
    N8N --> TG
    N8N --> INFLUX
    MODEL -->|geladen door| MW
    API -->|leest en schrijft| FEEDBACK
    API -->|leest| CLIPS
    WEB -->|proxy /api/| API
    WEB --> CF
    CF --> MAMMALRADAR
```

---

## Feedback & hertraining loop

```mermaid
sequenceDiagram
    participant MW as mammal-watcher
    participant FS as feedback/needs_review
    participant UI as review.html
    participant API as review-api
    participant TR as training/train.py
    participant MODEL as models/mammal_cnn.pt

    MW->>FS: Opslaan onzekere detectie (.wav + .json)
    UI->>API: GET /api/detections
    API->>UI: Lijst met clips + audio URLs
    UI->>UI: Gebruiker beluistert clip
    alt Correct
        UI->>API: POST /api/confirm
        API->>FS: Verplaats naar confirmed/
    else False positive
        UI->>API: POST /api/reject
        API->>FS: Verplaats naar rejected/
    end
    Note over TR: Periodiek handmatig uitvoeren
    TR->>MODEL: Hertraining op confirmed/ data
    MODEL->>MW: Kopieer naar server → docker compose up --build
```

---

## Toekomstige opstelling (v2 — Veldkastje) 🌿

> **Status: in ontwerp** — Het veldkastje is een autonome, batterijaangedreven
> eenheid die lokaal classificeert en detecties via WiFi of 4G doorstuurt
> naar de centrale server.

```mermaid
graph TD
    subgraph Veld["🌿 Veldkastje (buiten, autonoom)"]
        MIC_VELD["🎙️ Microfoon\nUSB of I2S MEMS"]
        PI["🍓 Raspberry Pi / SBC"]
        MODEL_VELD["models/mammal_cnn.pt\nlokale kopie"]
        BUFFER["Audio buffer\n5s vensters"]
        LOCAL_CLS["MammalCNNClassifier\nlokale inferentie"]
        UPLINK["📡 Uplink\nWiFi / 4G / LoRa"]
    end

    subgraph T630["🖥️ HP T630 — centrale server"]
        INGEST["Ingest API\n(toekomstig endpoint)"]
        MW_CENTRAL["mammal-watcher\ncentraal"]
        FEEDBACK_CENTRAL["feedback/\nconfirmed / rejected"]
        MODEL_CENTRAL["models/mammal_cnn.pt\nmaster model"]
    end

    subgraph Internet2["Internet"]
        MAMMALRADAR2["🌐 mammalradar.net\nreview + dashboard"]
    end

    MIC_VELD --> PI
    PI --> BUFFER
    MODEL_VELD --> LOCAL_CLS
    BUFFER --> LOCAL_CLS
    LOCAL_CLS -->|detectie + audio clip| UPLINK
    UPLINK -->|HTTPS POST| INGEST
    INGEST --> MW_CENTRAL
    MW_CENTRAL --> FEEDBACK_CENTRAL
    FEEDBACK_CENTRAL -->|hertraining| MODEL_CENTRAL
    MODEL_CENTRAL -->|OTA model update| MODEL_VELD
    MAMMALRADAR2 -->|review UI| FEEDBACK_CENTRAL
```

### Veldkastje — hardwareopties (concept)

| Component | Optie A (WiFi-bereik) | Optie B (autonoom buiten bereik) |
|---|---|---|
| **SBC** | Raspberry Pi Zero 2W | Raspberry Pi 4 + 4G HAT |
| **Microfoon** | USB-microfoon of I2S MEMS | Zelfde |
| **Opslag** | microSD (lokale buffer) | microSD |
| **Connectiviteit** | WiFi naar thuisnetwerk | 4G SIM / LoRaWAN |
| **Stroom** | 5V adapter of powerbank | Zonnepaneel + LiPo accu |
| **Behuizing** | Weerbestendige IP65 box | Zelfde |
| **Model update** | Git pull + restart | OTA via HTTPS |

### Veldkastje — dataflow

```mermaid
sequenceDiagram
    participant MIC as Microfoon
    participant PI as Veldkastje (Pi)
    participant SERVER as Centrale server (T630)
    participant UI as mammalradar.net

    loop Elke 5 seconden
        MIC->>PI: Audio chunk (5s, 16kHz)
        PI->>PI: MammalCNNClassifier.classify()
        alt Detectie boven drempel
            PI->>SERVER: POST /api/ingest clip + metadata
            SERVER->>UI: Nieuwe detectie zichtbaar
        else Geen detectie
            PI->>PI: Weggooien, doorgaan
        end
    end

    Note over SERVER,PI: Periodiek model-update
    SERVER->>PI: Nieuw mammal_cnn.pt via OTA
```

---

## Componenten — snel overzicht

| Component | Type | Host | Poort | Doel |
|---|---|---|---|---|
| ESP32-microfoon | Hardware | 192.168.2.20 | 8554 | Audio bron |
| MediaMTX | Docker | T630 | 8554 | RTSP relay |
| rtsp-bridge | Docker | T630 | — | ESP32 → relay doorsturen |
| mammal-watcher | Docker | T630 | — | AI classificatie + MQTT |
| review-api | Docker | T630 | 8081 | Feedback REST API |
| mammalradar-web | Docker | T630 | 8080 | Website + review UI |
| n8n | Systeem | T630 | 5678 | Workflow automatisering |
| BirdNET-Go | Systeem | NUC (192.168.2.23) | 8080 | Vogeldetectie |
| Home Assistant | Systeem | HA server | 8123 | Domotica integratie |
| Mosquitto MQTT | Systeem | HA server | 1883 | Berichtenbus |
| Cloudflare Tunnel | Cloud | — | — | mammalradar.net → T630 |
| Veldkastje (v2) | Hardware | Buiten | — | Lokale detectie |

---

## Mappenstructuur op de server

```
/home/natuurwaarnemer/mammal-watcher/
├── config.yaml              # Configuratie (model, MQTT, drempels)
├── docker-compose.yml       # Stack definitie
├── Dockerfile               # mammal-watcher image
├── Dockerfile.api           # review-api image
├── mammal_watcher.py        # Hoofdproces + RTSP loop
├── classifier.py            # AI model klassen (Stub / YAMNet / MammalCNN)
├── review_api.py            # Feedback REST API (FastAPI)
├── feedback_collector.py    # Feedback opslag helper
├── models/
│   └── mammal_cnn.pt        # Getraind CNN model (~333 KB)
├── clips/
│   ├── confirmed/           # Bevestigde detecties (WAV)
│   ├── uncertain/           # Onzekere detecties (WAV)
│   └── index.jsonl          # Index van alle clips
├── feedback/
│   ├── needs_review/        # Wacht op beoordeling via UI
│   ├── confirmed/           # Goedgekeurd via review.html
│   └── rejected/            # Afgekeurd als false positive
├── training/
│   └── train.py             # CNN trainingsscript (PyTorch)
├── dataset/                 # Trainingsdata (GBIF, iNaturalist, NatureLM)
└── web/
    ├── index.html           # Landingspagina (mammalradar.net)
    ├── review.html          # Review interface (bevestigen / afwijzen)
    ├── nginx.conf           # Nginx configuratie + /api/ proxy
    └── assets/              # Logo en statische bestanden
```

---

## Netwerkoverzicht thuis

```mermaid
graph LR
    ESP32["🎙️ ESP32\n192.168.2.20"] -->|RTSP| T630
    T630["🖥️ T630\n192.168.2.35"] -->|MQTT| HA["🏠 Home Assistant"]
    T630 -->|RTSP relay| NUC["💻 NUC\n192.168.2.23"]
    T630 -->|Cloudflare| WEB["🌐 mammalradar.net"]
    VELD["🌿 Veldkastje (v2)"] -->|WiFi/4G| T630
```

---

*Laatst bijgewerkt: 2026-05-14 — gegenereerd op basis van de actieve codebase.*
*Voor vragen of aanpassingen: open een Issue op GitHub.*
