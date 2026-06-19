# MammalRadar — CLAUDE.md

Lees dit bestand aan het begin van ELKE sessie. Dit is de enige bron van waarheid.
CONTEXT.md bevat de projectvisie en roadmap; dit bestand bevat de technische werkregels.

---

## Werkregels (strikt — geleerd uit incidenten)

| Type wijziging | Aanpak |
|---|---|
| Content (CLAUDE.md, CONTEXT.md, README, docs, HTML tekst) | Direct commit op main |
| Code (.py, Dockerfile, shell scripts) | **PR verplicht** |
| Na PR mergen met codewijzigingen | Altijd `git pull` op T630 |
| Docker build/rebuild | **Altijd door gebruiker zelf uitvoeren** |
| Infrastructuur wijzigen | **Nooit zonder expliciete opdracht** |

**Eén stap tegelijk.** Niet meerdere problemen tegelijk aanpakken.

---

## Servers & toegang

| Server | IP | Rol | SSH |
|---|---|---|---|
| HP T630 | 192.168.2.35 | Hoofd (mammal-watcher + n8n) | `ssh natuurwaarnemer@192.168.2.35` |
| NUC2 | 192.168.2.23 | BirdNET-Go (vogels) | niet ingesteld vanaf T630 |
| ESP32 microfoon | 192.168.2.20 | RTSP bron | — |

**Repo:** https://github.com/natuurwaarnemer/mammal-watcher
**GitHub token:** zit in git remote URL op T630 — ophalen via `git remote get-url origin`
**gh CLI:** NIET geïnstalleerd op T630 — PR aanmaken via GitHub API (curl) of vanuit HA machine

---

## Projectmap op T630

```
/home/natuurwaarnemer/mammal-watcher/   ← repo root
├── classifier.py                        ← YAMNetMLPClassifier, BirdNetMLPClassifier, MammalCNNClassifier
├── mammal_watcher.py                    ← hoofdproces, RTSP → embed → classify → MQTT
├── training/
│   ├── extract_embeddings.py            ← YAMNet embedding extractie
│   └── train_mlp.py                     ← MLP trainer, leest embeddings_index.csv
├── dataset/
│   ├── download_birdnet_clips.py        ← NUC2 clips downloaden (--species filter beschikbaar)
│   ├── download_naturelm.py             ← NatureLM soorten + background via streaming
│   ├── species_targets.yaml             ← 33 NL/DE doelsoorten voor NatureLM download
│   └── extract_features.py             ← VEROUDERD, niet gebruiken
├── models/
│   └── mammal_mlp.pt                    ← huidig actief model
├── config.yaml                          ← model: yamnet_mlp, feedback: disabled
└── venv/                                ← Python venv met tensorflow-hub, torch, etc.

/mnt/usb/                                ← USB schijf, altijd gemount, geen sudo nodig
├── prepared/                            ← soort-submappen, 16kHz WAV clips
│   ├── vulpes_vulpes/ … sus_scrofa/     ← doelsoorten
│   ├── background/                      ← NUC2 vogelclips + corviden (groeit)
│   ├── homo_sapiens/ felis_catus/       ← aparte klassen
│   └── canis_lupus_familiaris/ …        ← BACKGROUND_SPECIES (→ background bij training)
├── embeddings/                          ← BirdNET embeddings (VEROUDERD na PR #38)
└── embeddings_yamnet/                   ← YAMNet embeddings (huidige trainingsbasis)
```

---

## Architectuur

```
ESP32 mic → RTSP (MediaMTX) → mammal_watcher.py
                                    ↓
                          YAMNet (tensorflow_hub)
                          1024-dim embedding
                                    ↓
                          MLP classifier (mammal_mlp.pt)
                          18 klassen: 15 soorten + felis_catus + homo_sapiens + background
                                    ↓
                    background / laag confidence → geen actie
                    soort gedetecteerd → MQTT → n8n → Telegram
```

**NUC2 (192.168.2.23:8080):** BirdNET-Go voor vogels — zelfde microfoon, volledig onafhankelijk. NOOIT aanraken.

---

## Klassen (18 totaal — zie species_config.json)

**Doelsoorten (15):** vulpes_vulpes, canis_lupus, canis_aureus, martes_martes, martes_foina,
meles_meles, lutra_lutra, capreolus_capreolus, cervus_elaphus, sus_scrofa, castor_fiber,
sciurus_vulgaris, eliomys_quercinus, lynx_lynx, felis_silvestris

**Aparte klassen:** felis_catus, homo_sapiens, background

**BACKGROUND_SPECIES** (worden naar `background` gemapt door train_mlp.py):
gallus_gallus, alces_alces, bos_taurus, canis_lupus_familiaris, capra_hircus, dama_dama,
equus_caballus, erinaceus_europaeus, gulo_gulo, marmota_marmota, mustela_putorius,
myocastor_coypus, nyctereutes_procyonoides, ondatra_zibethicus, oryctolagus_cuniculus,
ovis_aries, procyon_lotor, ursus_arctos

---

## Training pipeline (volgorde)

```bash
# Stap 1: YAMNet embeddings extraheren (alle soorten + background)
# Duurt ~uren. Draait in achtergrond: tail -f /tmp/yamnet_extract.log
source ~/mammal-watcher/venv/bin/activate
python training/extract_embeddings.py \
    --prepared-dir /mnt/usb/prepared \
    --embeddings-dir /mnt/usb/embeddings_yamnet

# Stap 2: MLP hertrainen (ALTIJD met -v mount, scripts zitten NIET in Docker image)
docker run --rm \
    -v ~/mammal-watcher:/app \
    -v /mnt/usb:/mnt/usb \
    mammal-watcher-mammal-watcher \
    python training/train_mlp.py \
        --embeddings-dir /mnt/usb/embeddings_yamnet/embeddings_index.csv \
        --output /app/models

# Stap 3: container herbouwen (door gebruiker)
docker compose build mammal-watcher && docker compose up -d mammal-watcher

# Stap 4: valideren
docker logs mammal-watcher --tail 50
```

---

## Data downloaden

### Corviden van NUC2 (background-klasse)

Kraai/roek/kauw zijn de grootste bron van false positives — hun roepen triggeren
martes/otter/bever. NUC2 heeft 78k+ roek, 13k kraai, 9k kauw clips van dezelfde microfoon.

```bash
source ~/mammal-watcher/venv/bin/activate
python dataset/download_birdnet_clips.py \
    --output /mnt/usb/prepared/background \
    --index /mnt/usb/prepared/index.csv \
    --clips 900 --min-confidence 0.80 \
    --species "Corvus frugilegus,Corvus corone,Corvus monedula"
```

### NatureLM — soorten + background (lang, checkpoint herstartbaar)

NatureLM-audio-training (EarthSpeciesProject, 26.4M samples) bevat:
- iNaturalist (320k) + Animal Sound Archive/Tierstimmen (16k) → soortspecifieke audio
- WavCaps (402k, 7568 uur) + UrbanSound + AudioCaps → omgevingsgeluiden voor background

Script downloadt EERST alleen metadata (snel), DAARNA alleen audio voor matches.
Draait in tmux sessie 'naturelm': `tmux attach -t naturelm`
Log: `tail -f /tmp/naturelm_download.log`

```bash
source ~/mammal-watcher/venv/bin/activate
# Alles tegelijk (soorten + background):
python dataset/download_naturelm.py \
    --max-per-species 500 --background-clips 2000

# Alleen background:
python dataset/download_naturelm.py --skip-species --background-clips 2000

# Checkpoint: herstart gewoon opnieuw, slaat bestaande bestanden over
```

### iNaturalist — soortspecifieke clips (NIEUW — taxon IDs gecorrigeerd PR #47)

Download ruwe audio per soort via open iNaturalist API (geen key nodig).
Draait in tmux sessie 'inaturalist': `tmux attach -t inaturalist`

```bash
# Stap 1: download ruwe MP3/OGG per soort
source ~/mammal-watcher/venv/bin/activate
python dataset/download_inaturalist.py \
    --species-file dataset/species_targets.yaml \
    --output /tmp/inat_raw \
    --max-per-species 500

# Stap 2: verwerk naar 16kHz WAV chunks → prepared dir
python dataset/prepare_dataset.py \
    --input /tmp/inat_raw \
    --output /mnt/usb/prepared \
    --species-file dataset/species_targets.yaml
```

Beschikbare clips (top): vos 1702, grijze eekhoorn 2586, ree 723, edelhert 181,
goudjakhals 196, wolf 141, rode eekhoorn 117, wild zwijn 119, eikelmuis 85, relmuis 64.

### BirdNET-Go NUC2 API

```bash
# Clips downloaden (max 1000 per request, geen auth)
http://192.168.2.23:8080/api/v2/audio/{id}   # → audio/mp4
http://192.168.2.23:8080/api/v2/detections   # 276k+ detecties
```

---

## Docker stack

```bash
docker ps                          # running containers
docker compose up -d               # start alles
docker compose build mammal-watcher  # alleen watcher rebuilden
docker logs mammal-watcher --tail 100 -f
```

Containers: `mammal-watcher`, `mammalradar-web` (nginx), `mammal-review-api`, `mammal-mediamtx`, `gotenberg`

---

## config.yaml — huidige instellingen (2026-06-15)

```yaml
classifier:
  model: yamnet_mlp
  min_confidence: 0.70
  tier1_threshold: 0.85
  tier2_threshold: 0.65

clips:
  save_uncertain: false      # uitgeschakeld — ruis

feedback:
  enabled: false             # uitgeschakeld tijdens databouw
  min_pending_confidence: 0.70  # was 0.40 — te veel false positives
```

**Feedback/pending weer aanzetten na hertraining:** zet `enabled: true` in config.yaml en herstart container.

---

## CI / GitHub Actions

- Smoke test: `python mammal_watcher.py --no-rtsp --dry-run --config config.yaml`
  - Geen model beschikbaar in CI → valt terug op `_NoRtspFallbackClassifier`
  - Expliciete model branches (yamnet_mlp, birdnet_mlp) hebben try/except voor `--no-rtsp`
- Tests: `pytest tests/` — 76 tests, draait zonder GPU/model/USB

---

## Bekende valkuilen

| Valkuil | Uitleg |
|---|---|
| `dataset/extract_features.py` | Verouderd — verkeerd outputformaat (per-soort .npy). Gebruik `training/extract_embeddings.py` |
| Docker training zonder `-v /mnt/usb` | Training scripts zitten NIET in image; USB data ook niet |
| `embeddings/` (zonder `_yamnet`) | BirdNET embeddings — verouderd na PR #38, niet meer gebruiken |
| TF >= 2.16 | Maakt BirdNET tensor 545 ontoegankelijk via XNNPACK — gepind op 2.15.1 (YAMNet heeft dit niet) |
| `prepared/index.csv` | Bevat alleen 15 doelsoorten — NIET background/homo_sapiens/etc. Gebruik `--prepared-dir` |
| Copilot branches | 30+ open Copilot branches op GitHub — niet mergen zonder review |
| Corviden (kraai/roek/kauw) | Grootste bron van false positives — triggeren martes/otter klassen. Worden als background toegevoegd via PR #41 + NUC2 download |
| feedback/needs_review overloop | Bij min_pending_confidence 0.40 en 6 detecties/minuut: 8000+ bestanden/dag. Drempel nu 0.70, feedback disabled tijdens databouw |

---

## Recente PR historie

| PR | Inhoud | Status |
|---|---|---|
| #35 | fix: BirdNET tensor 545 + background-klasse structuur | ✅ main |
| #36 | feat: download NUC2 background-clips | ✅ main |
| #37 | fix: RMS stiltefilter in prepare_dataset.py | ✅ main |
| #38 | fix: YAMNet extractiepipeline (vervangt BirdNET) + `--prepared-dir` | ✅ main |
| #39 | feat: YAMNetMLPClassifier + config naar yamnet_mlp | ✅ main |
| #40 | fix: verwijder per-frame normalisatie RTSPConsumer | ✅ main |
| #41 | feat: `--species` filter in download_birdnet_clips.py (corviden) | ✅ main |
| #42 | feat: NatureLM output → /mnt/usb/prepared + background modus | ✅ main |
| #43 | fix: NatureLM metadata scan via streaming met vroege exit | ✅ main |
| #44 | refactor: single-pass streaming NatureLM (match + download in één loop) | ✅ main |
| #45 | feat: species_targets.yaml uitgebreid van 13 → 33 NL/DE soorten | ✅ main |
| #46 | feat: stream-checkpoint NatureLM — herstart vanaf opgeslagen positie | ✅ main |
| #47 | fix: 23 foute iNaturalist taxon IDs gecorrigeerd (vos stond op coyote etc.) | ✅ main |

---

## Volgende stappen (stand 2026-06-16)

**Nu bezig (stand 2026-06-19):**
- **NatureLM download** — tmux `naturelm`, log: `/tmp/naturelm_download.log`
  - Heeft nu **checkpoint** (PR #46): herstart vanaf opgeslagen positie in `/mnt/usb/naturelm_checkpoint.json`
  - Watchdog: `naturelm_watchdog.sh` via cron `0 * * * *` — herstart alleen als script dood is
- **iNaturalist download** — tmux `inaturalist` (gestart 2026-06-19 ~16:00)
  - Output: `/tmp/inat_raw/` → daarna `prepare_dataset.py` → `/mnt/usb/prepared/`
  - Na voltooiing: `venv/bin/python dataset/prepare_dataset.py --input /tmp/inat_raw --output /mnt/usb/prepared --species-file dataset/species_targets.yaml`
- mammal-watcher container: **GESTOPT** — wacht op hertraining

**Na beide downloads klaar — NIET meer:
1. Embeddings herextracten (alleen nieuwe bestanden, bestaande worden overgeslagen):
   ```bash
   source ~/mammal-watcher/venv/bin/activate
   python training/extract_embeddings.py \
       --prepared-dir /mnt/usb/prepared \
       --embeddings-dir /mnt/usb/embeddings_yamnet
   ```
2. MLP hertrainen (zie Training pipeline)
3. Container rebuilden (gebruiker)
4. Feedback/pending weer aanzetten in config.yaml
5. Valideren via mammalradar.net/review

**Daarna gepland:**
- n8n Telegram alerts tier-1 soorten (wolf, otter, bever) aanzetten
- Nicla Voice edge model — Edge Impulse data uploaden + binair model trainen
- Plausibiliteitslaag NDFF/GBIF (issue #33)

---

## Veldkastje — BOM & Architectuur

### Dataflow

```
[Omgeving / geluid]
        ↓
[Nicla Voice]
  - NDP120 neural processor + ingebouwde PDM-mic
  - Edge Impulse model (zie scope hieronder)
  - Stuurt via UART: label + confidence + trigger
        ↓
[ESP32-C6]                          ← ESPHome (geen custom firmware nodig)
  - Ontvangt UART van Nicla Voice
  - Optioneel: SD-kaart voor audio-buffer
  - Modem: WiFi (fase 1) of LilyGo T-SIM7600E 4G (fase 2)
  - Stuurt detectie → MQTT / HA API / webhook
        ↓
[Netwerk — WiFi of GSM/4G]
        ↓
[T630 / Server]
  - Ontvangst via MQTT of webhook
  - Opslag audio-snippets
  - Soortherkenning: mammal_mlp.pt (YAMNet embeddings)
        ↓
[Home Assistant / n8n / Dashboard]
  - Telegram alerts (tier-1 soorten)
  - mammalradar.net (review + visualisatie)
```

### Hardware BOM

#### Fase 1 — WiFi prototype (thuis/tuin)

| Component | Details | Prijs |
|---|---|---|
| Arduino Nicla Voice | NDP120 + PDM-mic, Edge Impulse deploy | ~€86 |
| ESP32-C6 devboard | WiFi 6, ESPHome, UART brug | ~€8 |
| Micro SD module + 32GB | Lokale audio-buffer | ~€10 |
| LiPo 3.7V 3000mAh | Stroom | ~€10 |
| TP4056 charger module | Laden via USB | ~€2 |
| Dupont/JST kabels | Verbindingen | ~€5 |
| **Totaal fase 1** | | **~€121** |

#### Fase 2 — Veld (GSM, waterproof, autonoom)

| Component | Details | Prijs |
|---|---|---|
| LilyGo T-SIM7600E | ESP32 + 4G modem in één board (vervangt ESP32-C6) | ~€35 |
| Simbase SIM | Pay-per-MB, NL/EU | ~€5 + gebruik |
| Zonnepaneel 5W + laadregelaar | Autonoom | ~€15 |
| IP67 behuizing 150×100×75mm | Weerbestendig | ~€12 |
| Kabelwartels M12 | Waterdicht | ~€5 |
| Windkap microfoon | Windruis | ~€4 |
| **Extra fase 2** | | **~€76** |

### Verbinding Nicla Voice ↔ ESP32-C6

```
Nicla Voice          ESP32-C6 (ESPHome)
───────────          ──────────────────
TX (UART)   ──────►  GPIO17 (RX)
RX (UART)   ◄──────  GPIO18 (TX)
GND         ──────►  GND
VIN (3.3V)  ◄──────  3V3
```

ESPHome UART config:
```yaml
uart:
  rx_pin: GPIO17
  tx_pin: GPIO18
  baud_rate: 9600
```

### Edge Impulse model scope (Nicla Voice NDP120)

Het NDP120 heeft beperkt RAM (~50KB voor activaties) en modelgeheugen (~200KB).

| Aanpak | Klassen | Haalbaarheid | Aanbevolen? |
|---|---|---|---|
| Binair: zoogdier / achtergrond | 2 | Zeker | Start hier |
| Klein: wolf + vos + bever + overig + achtergrond | 5 | Goed | Fase 2 |
| Alle soorten (12+) | 12+ | Risico | Niet voor NDP120 |

Trainingsdata staat klaar: `/mnt/usb/prepared/` (16kHz WAV, direct bruikbaar voor Edge Impulse).
