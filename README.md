# 🐾 MammalRadar

**Every sound is a signal.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Open Source](https://img.shields.io/badge/open--source-%E2%9D%A4-brightgreen.svg)](https://github.com/natuurwaarnemer/mammal-watcher)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=flat&logo=buy-me-a-coffee&logoColor=black)](https://www.buymeacoffee.com/natuurwaarnemer)

---

It started as a personal challenge: train a model that recognises mammals by their sound. But why stop at detection? MammalRadar grew into a complete open-source ecosystem — from a cheap field device to a self-learning AI that understands *why* an animal is there.

---

## Why MammalRadar?

| Solution | Cost | AI | Species | Open? |
|---|---|---|---|---|
| Rainforest Connection | High | ✅ | Tropical | ❌ Commercial |
| BirdNET-Go | Low | ✅ | Birds only | ✅ |
| AudioMoth | Low | ❌ Record only | All | ✅ |
| **MammalRadar** | **~€25** | **✅ Self-learning** | **All wildlife 🌍** | **✅ Open source** |

**Low-cost philosophy:**

| Component | Cost |
|---|---|
| Field device (Pi Zero 2W + microphone) | ~€25 |
| SIM data (Simbase, pay-per-MB) | ~€0.50 / year |
| Software | Free (open source) |

---

## System Overview

MammalRadar is more than a detector. It is a layered ecosystem where cheap hardware, edge AI, cloud intelligence, and ecological knowledge reinforce each other.

### 🧱 Hardware (field device)

- Directional microphone + pre-amp
- Edge device: ESP32, Pi Zero 2W, or Pi 4
- GSM module for data transfer
- Local buffering of audio snippets
- Timestamp + GPS or fixed location
- Optional: weather sensors

### 🖥️ Backend

- API for receiving detections
- Storage of audio, metadata, and spectrograms
- Database for species, behaviour, and context
- Model inference engine
- Feedback engine
- Model version management

### 🧠 AI layer

- Base model (13 Dutch mammal species)
- Augmentation pipeline
- Class balancing
- Periodic retraining
- Context-aware scoring
- Model evolution (specialised models for rain, night, forest, etc.)

### 🌍 Ecological intelligence (GBIF)

- Species presence maps
- Seasonal activity curves
- Habitat matching
- Regional sensitivity tuning
- Detection validation

### 🎧 Behaviour interpretation (NatureLM)

- Behaviour classification (alarm, territorial, foraging, mating)
- Contextual explanation in natural language
- Multi-species analysis
- Uncertainty estimation

---

## How It Works

Every detection flows through a smart, data-efficient pipeline:

```
Field device detects sound
  → Generates spectrogram locally
  → Runs CNN inference on-device
  → confidence > 0.85 : sends spectrogram + audio via GSM  (~80 KB)
  → confidence 0.5–0.85: sends spectrogram only             (~10 KB)
  → confidence < 0.5  : discarded

Backend:
  → NatureLM analyses behaviour
  → GBIF validates ecological probability
  → Result → dashboard + user notification
```

This keeps monthly data costs at roughly **9 MB** for a typical deployment — well within any pay-per-MB SIM plan.

---

## Intelligence Layers

MammalRadar becomes smarter over time through four reinforcing loops:

1. **Data-driven** — every new audio clip feeds back into retraining
2. **Context-driven** — time of day, season, habitat, weather, and GBIF presence data all influence scoring
3. **Feedback loops** — false positives are flagged; rare species get extra training weight
4. **Model evolution** — specialised models are spawned for specific conditions (rain, night, dense forest)

---

## Roadmap

```
✅  1. Base detection + GSM
✅  2. Backend + storage
🔄  3. First AI model (13 species)  ← in progress
⬜  4. Augmentation + retraining
⬜  5. Context engine
⬜  6. NatureLM integration
⬜  7. GBIF integration
⬜  8. Feedback loops
⬜  9. Model evolution
⬜ 10. Dashboard
⬜ 11. Automatic retraining
```

---

## Current Status — Home Test Setup

The live development environment running at home:

- ESP32-C6 microphone → MediaMTX RTSP relay
- mammal-watcher (Docker) → mammal detection
- BirdNET-Go in parallel → bird detection
- MQTT → Home Assistant

This setup proves the full pipeline end-to-end and drives the first training dataset.

---

## Who Is This For?

- 🌿 Nature reserve managers
- 🔬 Researchers & universities
- 🏕️ Rangers & field workers
- 👨‍🌾 Farmers (predator detection)
- 🧒 Schools & education
- 🌍 Anyone, anywhere in the world

If you can solder a microphone and run a Docker container, you can deploy MammalRadar.

---

## Support This Project

MammalRadar is built in spare time, driven by a passion for wildlife and open technology. If you find it useful, a coffee is always appreciated! ☕

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png)](https://www.buymeacoffee.com/natuurwaarnemer)

---

## Contributing

MammalRadar is open and welcoming. Whether you improve the model, add a new species dataset, build a better field enclosure, or translate the dashboard — every contribution matters. Open an issue or pull request — all skill levels welcome.

No contribution is too small. The goal is to make wildlife monitoring accessible to everyone.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

MammalRadar stands on the shoulders of two projects that made it possible:

- [BirdNET-Go](https://github.com/tphakala/birdnet-go) by **@tphakala**
  — the project that started it all 🐦 A complete, polished open-source bird detection system that proved real-time audio AI on consumer hardware is possible. MammalRadar is essentially "what if we did this for mammals?"

- [Sukecz/esp32-birdnet-mic](https://github.com/Sukecz/esp32-birdnet-mic) by **@Sukecz**
  — built an ESP32 I2S microphone specifically to feed audio into BirdNET-Go 🎤 That same firmware is what MammalRadar uses as its field microphone today.

And the broader ecosystem it builds on:

- [MediaMTX](https://github.com/bluenviron/mediamtx)
  — RTSP relay that makes fan-out to multiple consumers possible
- [NatureLM](https://github.com/earthspecies/NatureLM-audio)
  — behaviour interpretation and audio AI research
- [GBIF](https://www.gbif.org/)
  — ecological presence data for plausibility validation

---

## Current Development Setup

> The sections below document the current home test architecture and are preserved for reference during active development.

### Hardware

| Component | Details |
|---|---|
| 🎙️ Microphone | ESP32-C6 with I²S mic — `rtsp://192.168.2.20:8554/audio` @ 54 kHz mono |
| 🖥️ Server | HP T630 (`n8nserver`, 192.168.2.35) — Ubuntu 24.04, Docker 29.4.2 |
| 🐦 Bird detector | BirdNET-Go on NUC (192.168.2.23) |
| 📡 MQTT broker | Home Assistant (`homeassistant:1883`) |
| 🤖 Automation | n8n 2.8.4 (native systemd on T630) |

### Architecture (live as of v0.3)

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

**Why ffmpeg-bridge + `source: publisher`?**
The ESP32 firmware accepts only one RTSP client at a time. MediaMTX's built-in RTSP client caused `unexpected interleaved frame` errors with the ESP32 firmware. Using ffmpeg as a pull/push bridge works around this limitation.

### Quick Start

See **[INSTALL.md](INSTALL.md)** for the full step-by-step guide. In short:

```bash
# 1. Clone the repo
git clone https://github.com/natuurwaarnemer/mammal-watcher.git
cd mammal-watcher

# 2. Set ESP32 IP
cp .env.example .env
nano .env   # set ESP32_RTSP_URL to your ESP32's address

# 3. Configure MQTT credentials
nano config.yaml

# 4. Start the stack
docker compose up -d

# 5. Re-point BirdNET-Go to the MediaMTX relay
#    Change in BirdNET-Go config:
#    rtsp.url: rtsp://<T630-host>:8554/mic
#    (was: rtsp://192.168.2.20:8554/audio)
```

#### Dry-run (local test without MQTT)

```bash
python mammal_watcher.py --no-rtsp --dry-run --config config.yaml
```

Prints one sample payload to stdout and exits with code 0. Useful for verifying the pipeline without real hardware.

### Auto-start after Reboot

```bash
# 1. Create .env (once)
cp .env.example .env

# 2. Make startup script executable
chmod +x startup.sh

# 3. Install systemd service
sudo cp systemd/mammal-watcher.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable mammal-watcher.service
sudo systemctl start mammal-watcher.service

# 4. Verify
sudo systemctl status mammal-watcher.service
docker compose ps
```

### Troubleshooting

#### ESP32 refuses connection / rtsp-bridge crash-loops

The ESP32 firmware accepts **only one RTSP client at a time**. If BirdNET-Go is still connected directly to `rtsp://192.168.2.20:8554/audio` when rtsp-bridge tries to connect, the ESP32 drops one of them.

**First-time startup — order matters:**

1. Disable the BirdNET-Go stream (or temporarily change the URL) before starting ffmpeg-bridge.
2. Start the stack: `docker compose up -d`
3. Verify rtsp-bridge is connected: `docker logs mammal-rtsp-bridge | tail -5`
4. Re-point BirdNET-Go to `rtsp://<T630-host>:8554/mic` (the MediaMTX relay). Both consumers can now listen in parallel.

#### RTP packet warning in MediaMTX logs

```
RTP packets are too big (1460 > 1440), remuxing them into smaller ones
```

This is **harmless**. The ESP32 sends 1460-byte RTP packets; MediaMTX automatically fragments them to ≤ 1440 bytes without data loss. No action needed.

---

## Voor Nederlandstalige gebruikers

MammalRadar is een volledig open-source ecosysteem voor akoestische wildlife monitoring — van een goedkoop veldkastje (±€25) tot zelflerend AI-model. Het project begon als een persoonlijke uitdaging en groeide uit tot een compleet platform voor iedereen die natuur wil monitoren.
