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

## Architectuur (live as of v0.3)

```
ESP32-C6 (192.168.2.20:8554/audio)
        │
        ▼ RTSP/TCP pull
┌───────────────────────────────────────────────┐
│  HP T630 (n8nserver, 192.168.2.35)            │
│                                               │
│  ┌──────────────┐   push /mic                 │
│  │ rtsp-bridge  │──────────────┐              │
│  │ (ffmpeg)     │              ▼              │
│  └──────────────┘    ┌──────────────────┐     │
│                      │    MediaMTX      │     │
│                      │   :8554/mic      │     │
│                      │   (Docker)       │     │
│                      └────────┬─────────┘     │
│                 RTSP fan-out  │               │
│           ┌───────────────────┤               │
│           ▼                   ▼               │
│  ┌──────────────┐   ┌──────────────────┐      │
│  │ mammal-      │   │  BirdNET-Go      │      │
│  │ watcher      │   │  (NUC :8554/mic) │      │
│  │ (Docker)     │   └──────────────────┘      │
│  └──────┬───────┘                             │
└─────────┼──────────────────────────────────────┘
          │ MQTT publish
          ▼
   homeassistant:1883 → Home Assistant sensors
```

**Waarom ffmpeg-bridge + `source: publisher`?**
De ESP32-firmware accepteert maar één RTSP-client tegelijk. MediaMTX's
ingebouwde RTSP-client veroorzaakte `unexpected interleaved frame`-fouten
met de ESP32-firmware. ffmpeg als pull/push bridge lost dit op: hij trekt
de stream van de ESP32 en pusht naar MediaMTX (`source: publisher`), waarna
MediaMTX de stream naar meerdere consumers fan-out (BirdNET-Go én mammal-watcher).

---

## Snelle start

Zie **[INSTALL.md](INSTALL.md)** voor de volledige handleiding. Kort samengevat:

```bash
# 1. Repo clonen
git clone https://github.com/natuurwaarnemer/mammal-watcher.git
cd mammal-watcher

# 2. ESP32 IP instellen
cp .env.example .env
nano .env   # pas ESP32_RTSP_URL aan naar het adres van jouw ESP32

# 3. Config aanpassen (MQTT-credentials)
nano config.yaml

# 4. Stack starten
docker compose up -d

# 5. BirdNET-Go ompluggen naar de MediaMTX relay
#    Verander in BirdNET-Go config:
#    rtsp.url: rtsp://<T630-host>:8554/mic
#    (was: rtsp://192.168.2.20:8554/audio)
```

### Dry-run (lokaal testen zonder MQTT)

```bash
python mammal_watcher.py --no-rtsp --dry-run --config config.yaml
```

Prints één sample payload naar stdout en sluit af met exitcode 0. Handig
om te controleren of de pipeline werkt zonder echte hardware.

---

## Probleemoplossing

### ESP32 weigert verbinding / rtsp-bridge crash-loops

De ESP32-firmware accepteert **maar één RTSP-client tegelijk**. Als BirdNET-Go
nog direct op `rtsp://192.168.2.20:8554/audio` luistert op het moment dat
rtsp-bridge probeert te verbinden, verbreekt de ESP32 één van de verbindingen
en eindigt rtsp-bridge in een restart-loop.

**Eerste keer opstarten — volgorde is belangrijk:**

1. Schakel de BirdNET-Go stream **uit** (of verander de URL tijdelijk) voordat
   je ffmpeg-bridge start.
2. Start de stack: `docker compose up -d`
3. Controleer dat rtsp-bridge verbonden is:
   `docker logs mammal-rtsp-bridge | tail -5`
4. Wijs BirdNET-Go opnieuw naar `rtsp://<T630-host>:8554/mic` (de MediaMTX relay).
   Nu kunnen beide consumers parallel luisteren.

### RTP-pakket waarschuwing in MediaMTX logs

```
RTP packets are too big (1460 > 1440), remuxing them into smaller ones
```

Dit is **onschadelijk**. De ESP32 stuurt RTP-pakketten van 1460 bytes; MediaMTX
fragmenteert ze automatisch naar ≤ 1440 bytes zonder dataverlies. Geen actie nodig.

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
