# mammal-watcher 🦡

**mammal-watcher** is een Python-service die zoogdiergeluiden detecteert via
een live RTSP-audiostream. Hij luistert naar een ESP32-I²S-microfoon via een
MediaMTX relay, analyseert 5-seconden vensters op **zoogdiergeluiden** en
stuurt detecties via MQTT naar Home Assistant, n8n en InfluxDB.

Zie **[INSTALL.md](INSTALL.md)** voor de volledige stap-voor-stap handleiding.

---

## Hardware

| Component | Details |
|---|---|
| 🎙️ Microfoon | ESP32-C6 met I²S mic — `rtsp://192.168.2.20:8554/audio` @ 54 kHz mono |
| 🖥️ Server | HP T630 (`n8nserver`, 192.168.2.35) — Ubuntu 24.04, Docker 29.4.2 |
| 🐦 Vogel-detector | BirdNET-Go op NUC (192.168.2.23) |
| 📡 MQTT broker | Home Assistant (`homeassistant:1883`) |
| 🤖 Automatisering | n8n 2.8.4 (native systemd op T630) |

---

## Architectuur

```
ESP32-C6 (192.168.2.20:8554)
        │
        ▼ RTSP (1 publisher slot)
┌───────────────────────────────────────┐
│  HP T630 (n8nserver, 192.168.2.35)    │
│  ┌─────────────┐    ┌──────────────┐  │
│  │  MediaMTX   │───▶│ mammal-      │  │
│  │  :8554      │    │ watcher      │  │
│  │  (Docker)   │    │ (Docker)     │  │
│  └──────┬──────┘    └──────┬───────┘  │
└─────────┼──────────────────┼──────────┘
          │ RTSP relay       │ MQTT publish
          ▼                  ▼
   NUC: BirdNET-Go     homeassistant:1883
   (straks ompluggen        │
    naar T630:8554/mic)     ▼
                       n8n / Telegraf / HA
```

**Waarom MediaMTX?** De ESP32 accepteert maar één RTSP-client tegelijk.
MediaMTX relay lost dit op: BirdNET-Go én mammal-watcher kunnen beiden
de stream ontvangen.

---

## Snelle start

Zie **[INSTALL.md](INSTALL.md)** voor de volledige handleiding. Kort samengevat:

```bash
# Repo clonen
git clone https://github.com/natuurwaarnemer/mammal-watcher.git
cd mammal-watcher

# Config aanpassen (MQTT-credentials)
nano config.yaml

# MediaMTX relay starten en testen
docker compose up -d mediamtx
ffprobe -rtsp_transport tcp rtsp://localhost:8554/mic

# mammal-watcher starten
docker compose up -d mammal-watcher
docker logs -f mammal-watcher
```

### Dry-run (lokaal testen zonder MQTT)

```bash
python mammal_watcher.py --no-rtsp --dry-run --config config.yaml
```

Prints één sample payload naar stdout en sluit af met exitcode 0. Handig
om te controleren of de pipeline werkt zonder echte hardware.

---

## Roadmap — toekomstige PRs

| PR | Wat |
|----|-----|
| **#1** | Projectgeraamte met stub-classifier — bewijst de plumbing |
| **#2** | Architectuur-pivot: RTSP + MediaMTX + MQTT (dit) |
| **#4** | Echt ML-model: YAMNet of fine-tuned variant op NL-zoogdieren |
| **#5** | Trainings-pipeline: data verzamelen, labels, model bouwen |
| **#6** | NatureLM-integratie voor gedragsanalyse (lokroep, alarm, juveniel) |

---

## Hoe werkt GitHub hier?

GitHub is de digitale schuur voor dit project. Een korte uitleg:

- **Issues** = ideeën, fouten en vragen. Klik op het tabblad "Issues" en
  maak een nieuwe aan als je iets tegenkomt. Geen drempel.
- **Pull Requests (PRs)** = voorstellen voor wijzigingen. Ik (Copilot) maak
  een PR aan met nieuwe code. Jij leest mee, stelt vragen als commentaar, en
  klikt op **"Merge pull request"** als het er goed uitziet.
- **main branch** = de werkende versie. Wat hier staat, draait op jouw
  thuisserver. Code in een PR staat apart en raakt `main` pas aan na merge.
- **Commits** = snapshots. Elke wijziging heeft een datum en een beschrijving,
  zodat je altijd terug kunt naar gisteren.

Kort samengevat: **Issues = kladblok, PR = voorstel, main = werkend.**

---

## Dankwoord

Dit project maakt gebruik van [BirdNET-Go](https://github.com/tphakala/birdnet-go)
voor vogeldetectie en [MediaMTX](https://github.com/bluenviron/mediamtx) voor
de RTSP relay. Zonder die open-source bouwstenen was mammal-watcher er niet
geweest.

---

## Licentie

MIT — zie `LICENSE` (volgt in een latere PR).
