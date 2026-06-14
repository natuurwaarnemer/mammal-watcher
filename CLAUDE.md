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
│   ├── extract_embeddings.py            ← YAMNet embedding extractie (PR #38)
│   └── train_mlp.py                     ← MLP trainer, leest embeddings_index.csv
├── dataset/
│   └── extract_features.py              ← VEROUDERD, niet gebruiken (verkeerd formaat)
├── models/
│   └── mammal_mlp.pt                    ← huidig model (hertrainen na extractie)
├── config.yaml                          ← model: yamnet_mlp (gewijzigd PR #39)
└── venv/                                ← Python venv met tensorflow-hub, torch, etc.

/mnt/usb/                                ← USB schijf, altijd gemount, geen sudo nodig
├── prepared/                            ← 35+ soort-submappen, 16kHz WAV clips
│   ├── vulpes_vulpes/ … sus_scrofa/     ← doelsoorten
│   ├── background/                      ← 248 NUC2 vogelclips (= background klasse)
│   ├── homo_sapiens/ felis_catus/       ← aparte klassen in species_config.json
│   └── canis_lupus_familiaris/ …        ← BACKGROUND_SPECIES (→ background bij training)
├── embeddings/                          ← BirdNET embeddings (VEROUDERD na PR #38)
└── embeddings_yamnet/                   ← YAMNet embeddings (extractie loopt / klaar)
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
# En via browser: mammalradar.net/review
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

## BirdNET-Go NUC2 API (voor background-data)

```bash
# Clips downloaden (max 1000 per request, geen auth)
http://192.168.2.23:8080/api/v2/audio/{id}   # → audio/mp4
http://192.168.2.23:8080/api/v2/detections   # 267k+ detecties

# Script staat al klaar:
python dataset/download_birdnet_clips.py
```

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

---

## Recente PR historie

| PR | Inhoud | Status |
|---|---|---|
| #35 | fix: BirdNET tensor 545 + background-klasse structuur | ✅ main |
| #36 | feat: download NUC2 background-clips | ✅ main |
| #37 | fix: RMS stiltefilter in prepare_dataset.py | ✅ main |
| #38 | fix: YAMNet extractiepipeline (vervangt BirdNET) + `--prepared-dir` | ✅ main |
| #39 | feat: YAMNetMLPClassifier + config naar yamnet_mlp | ⏳ in review |

---

## Volgende stappen (na PR #39 merge)

1. Wacht tot YAMNet extractie klaar is: `tail -f /tmp/yamnet_extract.log`
2. Hertrainen (zie Training pipeline hierboven)
3. Container rebuilden
4. Valideren via mammalradar.net/review
5. Nicla Voice edge model (16kHz WAV klaarstaat in /mnt/usb/prepared/)
6. n8n Telegram alerts voor tier-1 soorten
