# Review-auth instellen

Maak eenmalig een `.htpasswd` bestand aan in de root van de repo zodat `/review` en `/api/` achter HTTP Basic Auth staan.

```bash
# Eenmalig aanmaken (apache2-utils of httpd-tools vereist):
htpasswd -c ~/mammal-watcher/.htpasswd ruud

# Of zonder pakketinstallatie via Docker:
docker run --rm httpd:alpine htpasswd -nb ruud kiesEenWachtwoord > ~/mammal-watcher/.htpasswd
```

Herstart daarna de web-container:

```bash
docker compose up -d mammalradar-web
```
