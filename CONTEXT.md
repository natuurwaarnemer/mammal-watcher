# 🦊 MammalRadar — Projectcontext

## Wat is dit project?
MammalRadar (repository: ) is een volledig **lokaal draaiend systeem** dat Nederlandse zoogdieren herkent op basis van geluid. Een goedkoop veldapparaat detecteert aan de rand, de server identificeert de soort, AI analyseert het gedrag, ecologische data valideert de detectie — en bij een bijzondere vondst gaat er een alert naar een bioloog.

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
2. **Soortherkenning** — BirdNET embeddings + PyTorch MLP op de T630 (18 soorten + background)
3. **Gedragsanalyse** — NatureLM: alarm / territorium / voortplanting / foerageren
4. **Plausibiliteit** — GBIF/NDFF: klopt deze soort op deze locatie en dit seizoen?
5. **Citizen science** — onverwachte detecties (hoge confidence, onbekende locatie) → Zoogdiervereniging/NDFF

---

## 🏗️ Architectuur — de omslag

### Oud (huidig, werkt niet goed)


### Nieuw (doel)


---

## 🔧 Huidige status (2026-06-12)

### ✅ Wat werkt
- Docker stack draait: mediamtx, mammal-watcher, review-api, mammalradar-web
- RTSP pipeline: ESP32 → MediaMTX → mammal-watcher
- Review workflow: web/review.html + review_api.py — bereikbaar op mammalradar.net/review
- Feedback loop: feedback_collector.py
- MQTT publisher (verbonden)
- mammalradar.net via Cloudflare tunnel
- BirdNET-Go op NUC2 (192.168.2.23) — 267k+ vogel-detecties, clips via HTTP API

### ✅ Gefixed in sessie 2026-06-12
- **Tensor bug (Copilot PR #32):** OUTPUT_TENSOR_INDEX 546→545, EMBEDDING_DIM 6522→1024
- **Background-klasse:** 18 niet-doelsoorten uit prepared/ mappen naar  label
- **Docker:** torch CPU-only wheel, tensorflow-cpu gepind op 2.15.1 (2.16+ blokkeert tensor 545)
- **BirdNET-Go clip download:** dataset/download_birdnet_clips.py — haalt vogel-WAVs op via HTTP API van NUC2
- **248 achtergrond-clips** gedownload van NUC2 (65 vogelsoorten, 16kHz WAV, zelfde microfoon)
- **min_confidence:** 0.95→0.70

### 🔄 Nu bezig (2026-06-12 avond)
**Embedding herextractie loopt** — ~13 uur, overnight:

23.856 clips × ~2s = klaar ~09:00 op 2026-06-13

### ⏳ Nog niet begonnen
- Nicla Voice integratie (nieuwe edge architectuur)
- NatureLM gedragsanalyse
- InfluxDB/Grafana
- n8n Telegram alerts
- Plausibiliteitslaag NDFF/GBIF

---

## 📋 Roadmap — in volgorde

| Stap | Wat | Status |
|------|-----|--------|
| **1** | Fix tensor + dim-mismatch | ✅ gedaan |
| **2** | Background-klasse toevoegen | ✅ gedaan (18 soorten + 248 vogelclips) |
| **3** | Embeddings herextraheren (23.856 clips) | 🔄 loopt overnight |
| **4** | Hertrainen MLP | ⏳ na extractie (~10 min) |
| **5** | Eerste detecties valideren via mammalradar.net/review | ⏳ na training |
| **6** | Nicla Voice: edge model trainen via Edge Impulse | ⏳ |
| **7** | n8n Telegram alerts voor tier-1 soorten | ⏳ |
| **8** | NatureLM gedragsanalyse | ⏳ |
| **9** | GBIF/NDFF plausibiliteitslaag | ⏳ |
| **10** | Citizen science koppeling NDFF | ⏳ |

---

## 🚀 Volgende sessie — begin hier

### Als extractie klaar is (check met )

**Stap 1: Hertrainen**


**Stap 2: Container herstarten**


**Stap 3: Detecties checken**

En via browser: mammalradar.net/review

### Als alles werkt: volgende fase
- Nicla Voice edge model — trainingsdata (16kHz WAV) staat al klaar in /mnt/usb/prepared/
- Edge Impulse account aanmaken, data uploaden, binair model trainen (zoogdier / niet)

---

## ⚙️ Werkregels (strikt — geleerd uit incident 15-05-2026)

| Type wijziging | Aanpak |
|---|---|
| Content (CONTEXT.md, README, docs, HTML tekst) | Direct commit op main ✅ |
| Code (, , , scripts) | **PR verplicht** 🔄 |
| Na  met codewijzigingen | Altijd  |
| Docker build/rebuild | **Altijd door gebruiker zelf uitvoeren** |
| Infrastructuur wijzigen | **Nooit zonder expliciete opdracht** |

### Regels voor Claude specifiek
- **Lees CONTEXT.md aan het begin van elke sessie** — dit is de enige bron van waarheid
- **Één stap tegelijk** — niet meerdere problemen tegelijk aanpakken
- **Zeg het als je iets niet weet** — niet speculeren en code schrijven
- **Geen autonome rebuilds** — code aanpassen ja, docker uitvoeren nee
- **PR beschrijving bevat altijd:** wat doet het, waarom, hoe te testen
- **Docker run voor training altijd met**  (training scripts zitten niet in image)

### Wat er mis ging op 15-05-2026
AI heeft zonder toezicht containers herbouwd en hergeïnstalleerd.
**Conclusie:** Claude schrijft code en PRs, gebruiker beslist over deployment.

---

## 🗂️ Bestandsstructuur (relevant)



---

## 🔑 Technische details (geleerd op 2026-06-12)

### BirdNET FP32 model tensor indices
- **Tensor 545:**  — shape (1, 1024) — **dit is de embedding**
- **Tensor 546:**  — shape (1, 6522) — soortklassificatie output
- TF >= 2.16 maakt tensor 545 ontoegankbaar met XNNPACK → gepind op 2.15.1
-  +  vereist vóór gebruik

### BirdNET-Go NUC2 API
- Clips ophalen:  → audio/mp4
- Detecties: 
- Max 1000 per request, geen authenticatie op lokaal netwerk
- Clips opgeslagen als AAC in  op NUC2 (SSH niet ingesteld vanaf T630)
- 267k+ detecties beschikbaar, 10k+ met confidence >= 0.85

### Background-klasse samenstelling
Automatisch bij training via BACKGROUND_SPECIES in train_mlp.py:
- 18 niet-doelsoorten uit prepared/ (hond, koe, paard, kip, enz.)
- 248 WAV-clips van NUC2 vogels (rechtstreeks gelabeld als  in index.csv)
- Cap: 500 clips via --max-per-species

### Sample rates
- Trainingsdata WAV: 16kHz (GBIF/iNaturalist) of 22kHz (Tierstimmen)
- BirdNET verwerking: librosa resampelt naar 48kHz intern
- Nicla Voice / Edge Impulse: 16kHz — zelfde WAV-bestanden direct bruikbaar

---

## 📊 Trainingsdata (USB /mnt/usb op T630)

| Map | Inhoud | Gebruik |
|-----|--------|---------|
|  | 35 soorten WAV clips + index.csv (23.856 entries) | Bron voor embeddings |
|  | 248 vogel-WAVs van NUC2 | Background-klasse |
|  | BirdNET .npy + embeddings_index.csv | MLP training input |
|  | 1024-dim sklearn .npy | **Niet gebruiken** |

---

## 🔗 Integraties

| Service | Adres | Status |
|---------|-------|--------|
| MQTT broker (HA) | homeassistant:1883 | ✅ verbonden |
| n8n | localhost:5678 | ✅ draait, workflow nog niet actief |
| BirdNET-Go (NUC2) | 192.168.2.23:8080 | ✅ HTTP API bereikbaar, SSH niet ingesteld |
| mammalradar.net | Cloudflare tunnel | ✅ live |
| Tierstimmen Archiv | — | Credits toegevoegd in CREDITS.md |

---

## 🛒 Hardware — wat bestellen

### Fase 1 — Prototype (WiFi, thuis/tuin testen)

| Component | Doel | Prijs |
|---|---|---|
| Arduino Nicla Voice | NDP120 + mic, edge model | ~€86 (NL prijs incl. BTW) |
| ESP32-WROOM-32 devboard | WiFi bridge + audio buffer | ~€8 |
| Micro SD module + 32GB kaart | Audio snippets opslaan | ~€10 |
| LiPo 3.7V 3000mAh | Stroom | ~€10 |
| TP4056 charger module | Laden via USB | ~€2 |
| Dupont/JST kabels | Verbindingen | ~€5 |
| **Totaal fase 1** | | **~€82** |

### Fase 2 — Veld (GSM, waterproof, autonoom)

| Component | Doel | Prijs |
|---|---|---|
| LilyGo T-SIM7600E | ESP32 + 4G in één board | ~€35 |
| Simbase SIM | Pay-per-MB, NL/EU dekking | ~€5 + gebruik |
| Zonnepaneel 5W + laadregelaar | Onbeperkt autonoom | ~€15 |
| IP67 behuizing 150x100x75mm | Weatherproof | ~€12 |
| Kabelwartels M12 | Waterdicht kabelinvoer | ~€5 |
| Windkap voor microfoon | Windruis onderdrukken | ~€4 |
| **Extra fase 2** | | **~€76** |

**Let op:** LilyGo T-SIM7600E vervangt de losse ESP32 uit fase 1.

### Verbinding Nicla Voice ↔ ESP32
- Protocol: **UART** (TX/RX)
- Nicla Voice stuurt: 
- ESP32 slaat audioclip op SD + stuurt door via WiFi/4G

### Edge Impulse
Gratis tier voldoende. Trainingsdata (16kHz WAV) staat klaar in /mnt/usb/prepared/.
