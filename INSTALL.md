# 🦡 Installatie-handleiding — mammal-watcher op de HP T630

Welkom! Deze gids leidt je stap voor stap door de installatie van
**mammal-watcher** op jouw HP T630 (`n8nserver`, 192.168.2.35).
Je hoeft geen programmeerervaring te hebben — als je de commando's
precies overneemt, gaat het goed.

---

## Voorvereisten

### Docker

Op jouw T630 is Docker **al geïnstalleerd** (versie 29.4.2). Controleer:

```bash
docker --version
docker compose version
```

Je zou iets als `Docker version 29.4.2` en `Docker Compose version v2.x.x`
moeten zien. ✅

> **Geen Docker?** Op een verse Ubuntu 24.04 installeer je Docker zo:
> ```bash
> curl -fsSL https://get.docker.com | sudo sh
> sudo usermod -aG docker $USER
> newgrp docker
> ```
> Controleer daarna opnieuw met `docker run --rm hello-world`.

---

## Stap 1 — Repo clonen

SSH naar de T630 en voer uit:

```bash
cd ~
git clone https://github.com/natuurwaarnemer/mammal-watcher.git
cd mammal-watcher
ls -la
```

Je zou bestanden moeten zien als `docker-compose.yml`, `mediamtx.yml`,
`config.yaml`, `Dockerfile` en `species_mammals_nl.csv`.

---

## Stap 2 — Config aanpassen

Open `config.yaml` in een tekstverwerker:

```bash
nano config.yaml
```

Controleer de MQTT-inloggegevens:

```yaml
mqtt:
  broker: homeassistant
  port: 1883
  username: birdnet
  password: secret   # ← pas aan als jouw wachtwoord anders is
```

> **Hoe check je de MQTT-credentials?**
> In Home Assistant: *Settings → Add-ons → Mosquitto broker → Configuration*.
> Kijk bij de gebruikersnamen en wachtwoorden.

Sla op met `Ctrl+O`, bevestig met `Enter`, sluit af met `Ctrl+X`.

---

## Stap 3 — Eerst alleen MediaMTX testen

MediaMTX is de relay die de ESP32-microfoon doorgeeft aan mammal-watcher
(de ESP32 accepteert maar één verbinding tegelijk).

```bash
docker compose up -d mediamtx
```

Wacht 10 seconden, controleer dan de logs:

```bash
docker logs mammal-mediamtx | head -20
```

Je zou iets als `INF MediaMTX v1.x.x starting` moeten zien.

Test of de stream bereikbaar is:

```bash
ffprobe -rtsp_transport tcp rtsp://localhost:8554/mic
```

> **Verwacht resultaat:** stream-info met `Audio: pcm_s16be` of vergelijkbaar.
> Als je `Connection refused` ziet, geef MediaMTX nog 30 seconden.

---

## Stap 4 — mammal-watcher starten

```bash
docker compose up -d mammal-watcher
```

Volg de logs live:

```bash
docker logs -f mammal-watcher
```

Je ziet regels als:
```
2026-05-05 18:00:00 INFO     Verbinding maken met RTSP-stroom: rtsp://localhost:8554/mic
2026-05-05 18:00:01 INFO     Verbonden met RTSP-stroom
2026-05-05 18:00:06 INFO     Gedetecteerd: Cervus elaphus (edelhert) conf=0.82 tier=1
```

Stop de logs met `Ctrl+C` (de container blijft draaien).

---

## Stap 5 — BirdNET-Go ompluggen naar de MediaMTX relay

> **Doe dit pas als stap 3 succesvol was.** We verleggen de RTSP-bron van de
> ESP32 naar de relay op de T630 zodat zowel BirdNET-Go als mammal-watcher
> de audio kunnen ontvangen.

SSH naar de NUC (BirdNET-Go host, 192.168.2.23) en bewerk de config:

```bash
nano ~/birdnet-go-app/config/config.yaml
```

Verander:
```yaml
rtsp:
  url: rtsp://192.168.2.20:8554/audio   # ← was: ESP32 direct
```
naar:
```yaml
rtsp:
  url: rtsp://n8nserver:8554/mic        # ← via MediaMTX relay op T630
```

Herstart BirdNET-Go:

```bash
systemctl --user restart birdnet-go
# of als service:
sudo systemctl restart birdnet-go
```

Controleer in de BirdNET-Go web UI (http://192.168.2.23:8080) dat detecties
nog steeds binnenkomen.

---

## Stap 6 — HPF-cutoff op de ESP32 verlagen (optioneel)

De meeste zoogdiergeluiden zitten lager dan vogels. De ESP32-microfoon heeft
standaard een High Pass Filter (HPF) van 700 Hz — dat filtert lagefrequente
zoogdiergeluiden weg.

1. Open in je browser: `http://192.168.2.20`
2. Zoek **HPF Cutoff** en verander `700` naar `150`
3. Klik **Set**

> **Reversibel:** als de vogel-detectiekwaliteit terugloopt, zet de HPF
> terug naar 700 Hz via dezelfde pagina.

---

## Stap 7 — n8n workflow importeren

1. Open n8n in je browser: `http://n8nserver:5678`
2. Ga naar *Workflows → Import from file*
3. Selecteer `n8n/mammal_workflow.json` uit de gekloonde repo
4. Maak een nieuwe **MQTT credential** aan:
   - Broker: `homeassistant`
   - Port: `1883`
   - Gebruikersnaam: `birdnet`
   - Wachtwoord: jouw MQTT-wachtwoord
5. Vul je **Telegram Bot Token** en **Chat ID** in bij de Telegram-nodes
6. Vul de **InfluxDB**-verbinding in (URL, bucket, org, token)
7. Zet de workflow op **Active**

---

## Stap 8 — Verificeer in Home Assistant

Na een paar minuten zou de entiteit
`sensor.mammal_watcher_last_detection` automatisch moeten verschijnen via
MQTT discovery.

Ga in Home Assistant naar *Settings → Devices & Services → MQTT* en kijk
of het apparaat **Mammal Watcher** er staat.

---

## Help, het werkt niet meer! 🚨

### BirdNET-Go stopt met detecteren

Zet BirdNET-Go terug naar de ESP32 direct:

```bash
# op de NUC:
nano ~/birdnet-go-app/config/config.yaml
# verander rtsp.url terug naar rtsp://192.168.2.20:8554/audio
systemctl --user restart birdnet-go
```

### HPF terugzetten

Open `http://192.168.2.20`, HPF Cutoff terug naar `700`, klik Set.

### mammal-watcher uitzetten

```bash
# op de T630:
docker compose down
```

Alle Docker-containers stoppen, n8n en BirdNET-Go worden **niet** geraakt.

---

## Handige commando's

| Actie | Commando |
|---|---|
| Status bekijken | `docker compose ps` |
| Logs live volgen | `docker logs -f mammal-watcher` |
| Opnieuw opstarten | `docker compose restart mammal-watcher` |
| Alles stoppen | `docker compose down` |
| Alles starten | `docker compose up -d` |
| MediaMTX stream testen | `ffprobe -rtsp_transport tcp rtsp://localhost:8554/mic` |

---

Vragen? Open een **Issue** op https://github.com/natuurwaarnemer/mammal-watcher.
