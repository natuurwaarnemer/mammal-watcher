"""
Download zoogdiergeluiden van Freesound.org als aanvulling op de NatureLM dataset.
Vereist een gratis Freesound API key: https://freesound.org/apiv2/apply/

Gebruik: python dataset/download_freesound.py --output dataset/raw --species-file dataset/species_targets.yaml
Stel de API key in via: export FREESOUND_API_KEY=<jouw_key>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
import yaml

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

FREESOUND_API_BASE = "https://freesound.org/apiv2"
DOWNLOAD_DELAY = 1.0  # seconden tussen downloads (respecteer de server)
DEFAULT_MAX_PER_SPECIES = 50


def _slug(scientific: str) -> str:
    """Zet wetenschappelijke naam om naar bestandsvriendelijke slug."""
    return re.sub(r"\s+", "_", scientific.strip().lower())


def _load_species(yaml_path: Path) -> list[dict]:
    with yaml_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data["species"]


def _search_freesound(query: str, api_key: str, page_size: int = 15) -> list[dict]:
    """Zoek Freesound op en geef een lijst van geluidsobjecten terug."""
    results = []
    url = f"{FREESOUND_API_BASE}/search/text/"
    params = {
        "query": query,
        "fields": "id,name,previews,license,username,duration,description,tags",
        "filter": "duration:[1 TO 60]",  # 1–60 seconden
        "page_size": page_size,
        "token": api_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
    except requests.RequestException as exc:
        print(f"  ⚠ Freesound zoekopdracht mislukt: {exc}", file=sys.stderr)
    return results


def _download_preview(sound: dict, dest: Path, api_key: str) -> bool:
    """
    Download het HQ preview WAV/MP3 van een Freesound geluid.
    Geeft True terug als het nieuw gedownload is.
    """
    if dest.exists():
        return False

    previews = sound.get("previews", {})
    # Kies HQ wav preview, anders mp3
    url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3", "")
    if not url:
        return False

    try:
        resp = requests.get(url, params={"token": api_key}, timeout=60, stream=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
    return True


def _save_metadata(records: list[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def download_species(
    species: dict,
    output_dir: Path,
    max_per_species: int,
    api_key: str,
) -> None:
    """Download Freesound-opnames voor één soort."""
    slug = _slug(species["scientific"])
    species_dir = output_dir / slug

    # Zoekterm: wetenschappelijke naam + Engelse naam voor betere resultaten
    query = f"{species['scientific']} {species['en']}"
    print(f"\n→ {species['nl']} ({species['scientific']}) — query: '{query}'")

    sounds = _search_freesound(query, api_key, page_size=max_per_species)
    if not sounds:
        print("  Geen opnames gevonden.")
        return

    sounds = sounds[:max_per_species]
    print(f"  {len(sounds)} opname(s) gevonden, downloaden...")

    iterator = tqdm(sounds, desc=slug) if HAS_TQDM else sounds
    downloaded = 0
    skipped = 0
    metadata_records: list[dict] = []

    for sound in iterator:
        sound_id = sound.get("id", "unknown")
        # Detecteer extensie op basis van beschikbare preview
        previews = sound.get("previews", {})
        ext = "mp3" if previews.get("preview-hq-mp3") else "mp3"
        filename = f"fs{sound_id}.{ext}"
        dest = species_dir / filename

        try:
            is_new = _download_preview(sound, dest, api_key)
        except requests.RequestException as exc:
            print(f"  ⚠ Download mislukt ({filename}): {exc}", file=sys.stderr)
            continue

        if is_new:
            downloaded += 1
            meta = {
                "source": "freesound",
                "id": sound_id,
                "name": sound.get("name"),
                "license": sound.get("license"),
                "username": sound.get("username"),
                "duration": sound.get("duration"),
                "description": sound.get("description"),
                "tags": sound.get("tags"),
                "local_file": str(dest),
                "species_scientific": species["scientific"],
                "species_nl": species["nl"],
            }
            metadata_records.append(meta)
            time.sleep(DOWNLOAD_DELAY)
        else:
            skipped += 1

    if metadata_records:
        _save_metadata(metadata_records, species_dir / "metadata.jsonl")

    print(f"  ✓ {downloaded} nieuw gedownload, {skipped} overgeslagen.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="dataset/raw",
        help="Uitvoermap voor ruwe downloads (standaard: dataset/raw)",
    )
    parser.add_argument(
        "--max-per-species",
        type=int,
        default=DEFAULT_MAX_PER_SPECIES,
        help=f"Maximum aantal opnames per soort (standaard: {DEFAULT_MAX_PER_SPECIES})",
    )
    parser.add_argument(
        "--species-file",
        default="dataset/species_targets.yaml",
        help="Pad naar species_targets.yaml",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("FREESOUND_API_KEY", ""),
        help="Freesound API key (of stel FREESOUND_API_KEY in als omgevingsvariabele)",
    )
    args = parser.parse_args()

    if not args.api_key:
        print(
            "⚠ Geen Freesound API key opgegeven.\n"
            "  Maak een gratis account aan op https://freesound.org/apiv2/apply/\n"
            "  en stel de key in met: export FREESOUND_API_KEY=<jouw_key>",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output)
    species_file = Path(args.species_file)

    if not species_file.exists():
        print(f"Bestand niet gevonden: {species_file}", file=sys.stderr)
        sys.exit(1)

    species_list = _load_species(species_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Freesound downloader — {len(species_list)} soort(en), max {args.max_per_species} per soort")
    print(f"Uitvoer: {output_dir.resolve()}\n")

    for species in species_list:
        download_species(species, output_dir, args.max_per_species, args.api_key)

    print("\n✅ Klaar!")


if __name__ == "__main__":
    main()
