# Cloudflare Tunnel Setup voor mammalradar.net

## Stap 1 — Voeg toe aan bestaande tunnel

1. Ga naar dash.cloudflare.com → Zero Trust → Networks → Tunnels
2. Klik op je bestaande tunnel (of maak een nieuwe aan)
3. Ga naar "Public Hostname" → "Add a public hostname"
4. Vul in:
   - Subdomain: (leeg, of www)
   - Domain: mammalradar.net
   - Service Type: HTTP
   - URL: localhost:8080
5. Sla op

## Stap 2 — DNS in Cloudflare

mammalradar.net moet als domain zijn toegevoegd aan je Cloudflare account.
Voeg toe als CNAME:
  - Name: @ (of www)
  - Target: <tunnel-id>.cfargotunnel.com
  - Proxied: ✅

## Stap 3 — Start de web container

```bash
cd ~/mammal-watcher
docker compose up -d mammalradar-web
```
