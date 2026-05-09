# 🦊 Animal Radar — Project Roadmap

## Visie
Een volledig lokaal draaiend systeem dat zoogdieren en vogels herkent op basis van geluid, met automatische alerts, dashboards en social media posts.

## Architectuur (7 stappen)

### ✅ Stap 1 — Audio verzamelen
- ESP32 microfoon → RTSP stream → MediaMTX
- ffmpeg bridge herlevert naar `rtsp://localhost:8554/mic`
- Status: **WERKEND**

### ✅ Stap 2 — YAMNet categorisatie (filter)
- YAMNet classificeert: bird / mammal / human / vehicle / weather
- Alleen 'mammal'-achtige scores gaan door
- Clips worden opgeslagen in `clips/confirmed` en `clips/uncertain`
- Status: **WERKEND**

### ⏳ Stap 3 — EcoSound analyse (optioneel / later)
- Subcategorisatie: carnivoor / knaagdier / groot zoogdier / rustling / squeak
- Kan als tussenlaag toegevoegd worden na stap 4
- Status: **UITGESTELD** — geen harde vereiste voor stap 4

### 🔄 Stap 4 — Eigen soortherkenningsmodel (NL zoogdieren)
- Doelsoorten: vos, das, otter, wezel, wolf, ree, edelhert, wild zwijn, bever, marter
- Trainingsdata: Xeno-Canto (zoogdiergeluiden), Macaulay Library, iNaturalist, Berlin Sound Archive
- Pipeline: download → normalize → YAMNet feature extraction → classifier (SVM of MLP)
- Model draait lokaal op HP630 / T630
- Output: species + confidence → MQTT → n8n
- Status: **IN ONTWIKKELING**

### ⏳ Stap 5 — NatureLM zero-shot (geavanceerde AI laag)
- Zero-shot soortherkenning
- Gedragsanalyse: alarm call / juveniel vs adult / song vs call
- Multi-species detectie
- Audio captioning: "Een vos die een territoriumroep produceert"
- Status: **TOEKOMST**

### ⏳ Stap 6 — InfluxDB + Grafana dashboards
- Alle detecties opslaan in InfluxDB op HP640
- Grafana: heatmaps, tijdreeksen, activiteitspatronen per soort
- Vogels (BirdNET) + zoogdieren (eigen model) gecombineerd
- Status: **TOEKOMST**

### ⏳ Stap 7 — n8n workflows
- Telegram alerts bij tier-1 detecties (wolf, otter, das)
- Mastodon / Instagram posts
- Dagelijkse samenvattingsrapporten
- Status: **TOEKOMST**

---

## Hardware
| Apparaat | Rol |
|---|---|
| ESP32 | Microfoon + RTSP stream |
| n8nserver (HP630/T630) | Mammal-watcher stack (YAMNet, eigen model) |
| NUC2 | BirdNET-Pi (vogels) |
| HP640 | InfluxDB + Grafana |

## Doelsoorten zoogdieren
| Soort | Wetenschappelijk | Prioriteit | Alert |
|---|---|---|---|
| Vos | Vulpes vulpes | Hoog | ✅ |
| Das | Meles meles | Hoog | ✅ |
| Otter | Lutra lutra | Zeer hoog | ✅ |
| Wolf | Canis lupus | Zeer hoog | ✅ |
| Wezel | Mustela nivalis | Hoog | ✅ |
| Hermelijn | Mustela erminea | Hoog | ✅ |
| Steenmarter | Martes foina | Middel | ✅ |
| Boommarter | Martes martes | Hoog | ✅ |
| Ree | Capreolus capreolus | Middel | ❌ |
| Edelhert | Cervus elaphus | Middel | ❌ |
| Wild zwijn | Sus scrofa | Laag | ❌ |
| Bever | Castor fiber | Hoog | ✅ |

## Trainingsdata bronnen
- [Xeno-Canto](https://xeno-canto.org) — vogels + enkele zoogdieren
- [Macaulay Library](https://www.macaulaylibrary.org) — Cornell Lab, zoogdiergeluiden
- [iNaturalist](https://www.inaturalist.org) — observaties met geluid
- [Berlin Sound Archive](https://tierstimmenarchiv.de) — Europese zoogdiergeluiden
- Eigen veldopnames via `clips/` map (active learning)
