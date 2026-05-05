# mammal-watcher 🦡

**mammal-watcher** is een Python-service die naast je bestaande
[BirdNET-Pi](https://github.com/mcguirepr89/BirdNET-Pi) installatie draait.
Hij luistert naar dezelfde audio-snippets die BirdNET-Pi aanmaakt en analyseert
die op **zoogdiergeluiden** in plaats van vogels. De resultaten worden via een
webhook naar n8n gestuurd, die ze doorstuurt naar InfluxDB (Grafana-dashboards)
en Telegram (alerts voor bijzondere soorten).

---

## Architectuur

```
 BirdNET-Pi snippetmap
 /home/pi/BirdNET-Pi/BirdSongs/Extracted/
         │
         │  nieuwe .wav of .flac bestanden
         ▼
 ┌────────────────────┐
 │  mammal-watcher    │  ← draait als Docker-container
 │  (deze service)    │
 └────────┬───────────┘
          │  JSON payload via HTTP POST
          ▼
 ┌────────────────────┐
 │  n8n webhook       │  ← op hp630.local
 │  /mammal-detection │
 └────────┬───────────┘
          │
    ┌─────┴──────┐
    ▼            ▼
 InfluxDB     Telegram
 (Grafana)    Tier1/Tier2
```

---

## Snelle start

### 1. Clone de repo

```bash
git clone https://github.com/natuurwaarnemer/mammal-watcher.git
cd mammal-watcher
```

### 2. Pas de instellingen aan

Open `config.yaml` in een tekstverwerker en stel in:

- `watch.snippet_dir` — pad naar de BirdNET-Pi snippetmap
- `n8n.webhook_url` — adres van je n8n-instantie
- `classifier.min_confidence` — minimale betrouwbaarheid voor een melding

```yaml
watch:
  snippet_dir: /home/pi/BirdNET-Pi/BirdSongs/Extracted
n8n:
  webhook_url: "http://hp630.local:5678/webhook/mammal-detection"
```

### 3. Start de service

```bash
docker compose up -d
```

De container start automatisch opnieuw op als je de computer herstart
(`restart: unless-stopped`).

### 4. Controleer of het werkt

```bash
docker compose logs -f
```

Je ziet regels als:
```
2026-05-05 18:00:00 INFO  Watching /home/pi/BirdNET-Pi/BirdSongs/Extracted
2026-05-05 18:02:13 INFO  Detected: Vulpes vulpes (vos) conf=0.82 tier=2
```

### Testen zonder echt te posten (dry-run)

```bash
python mammal_watcher.py --dry-run
```

Hiermee wordt er **niks naar n8n gestuurd** — payloads worden alleen op het
scherm geprint. Handig om te controleren of alles werkt.

---

## Roadmap — toekomstige PRs

| PR | Wat |
|----|-----|
| **#1** (dit) | Projectgeraamte met stub-classifier — bewijst de plumbing |
| **#2** | Echt ML-model (YAMNet + fine-tuning op NL-zoogdieren) |
| **#3** | Trainings-pipeline: data verzamelen, labels, model bouwen |
| **#4** | NatureLM-integratie voor gedragsanalyse (lokroep, alarm, juveniel) |
| **#5** | Veldkastje: Raspberry Pi + 4G + zonnepaneel |

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

Dit project is gebaseerd op de geweldige infrastructuur van
[BirdNET-Pi](https://github.com/mcguirepr89/BirdNET-Pi) — een open-source
project van Patrick McGuire dat real-time vogelherkenning mogelijk maakt op
een Raspberry Pi. Zonder die basis was mammal-watcher er niet geweest.

---

## Licentie

MIT — zie `LICENSE` (volgt in een latere PR).
