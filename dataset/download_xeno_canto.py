"""
Download zoogdiergeluiden van Xeno-Canto API v2.
Gebruik: python dataset/download_xeno_canto.py --output dataset/raw --max-per-species 20
"""

from __future__ import annotations

import argparse
import json
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

XENO_CANTO_API = "https://xeno-canto.org/api/2/recordings"
DOWNLOAD_DELAY = 1.0  # seconden tussen downloads (respecteer de server)


def _slug(scientific: str) -> str:
    """Zet wetenschappelijke naam om naar bestandsvriendelijke slug."""
    return re.sub(r"\s+", "_", scientific.strip().lower())


def _load_species(yaml_path: Path) -> list[dict]:
    with yaml_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data["species"]


def _search_xeno_canto(query: str) -> list[dict]:
    """Vraag Xeno-Canto API op en geef alle opnames terug (over alle pagina's)."""
    recordings: list[dict] = []
    page = 1
    while True:
        resp = requests.get(
            XENO_CANTO_API,
            params={"query": query, "page": page},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        recordings.extend(data.get("recordings", []))
        num_pages = int(data.get("numPages", 1))
        if page >= num_pages:
            break
        page += 1
        time.sleep(0.5)
    return recordings


def _download_recording(url: str, dest: Path) -> bool:
    """Download een enkel MP3-bestand. Geeft True terug als het nieuw is."""
    if dest.exists():
        return False
    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        for chunk in resp.iter_content(chunk_size=8192):
            fh.write(chunk)
    return True


def _save_metadata(records: list[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def download_species(
    species: dict,
    output_dir: Path,
    max_per_species: int,
) -> None:
    query = species["xeno_canto_query"]
    slug = _slug(species["scientific"])
    species_dir = output_dir / slug

    print(f"\n→ {species['nl']} ({species['scientific']}) — query: '{query}'")

    try:
        recordings = _search_xeno_canto(query)
    except requests.RequestException as exc:
        print(f"  ⚠ API-fout: {exc}", file=sys.stderr)
        return

    recordings = recordings[:max_per_species]
    if not recordings:
        print("  Geen opnames gevonden.")
        return

    print(f"  {len(recordings)} opname(s) gevonden, downloaden...")
    _save_metadata(recordings, species_dir / "metadata.jsonl")

    iterator = tqdm(recordings, desc=slug) if HAS_TQDM else recordings
    downloaded = 0
    skipped = 0
    for rec in iterator:
        # Xeno-Canto geeft een relatieve of absolute URL terug
        file_url = rec.get("file", "")
        if not file_url:
            continue
        if not file_url.startswith("http"):
            file_url = "https:" + file_url
        filename = f"xc{rec.get('id', 'unknown')}.mp3"
        dest = species_dir / filename
        try:
            is_new = _download_recording(file_url, dest)
        except requests.RequestException as exc:
            print(f"  ⚠ Download mislukt ({filename}): {exc}", file=sys.stderr)
            continue
        if is_new:
            downloaded += 1
            time.sleep(DOWNLOAD_DELAY)
        else:
            skipped += 1

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
        default=20,
        help="Maximum aantal opnames per soort (standaard: 20)",
    )
    parser.add_argument(
        "--species-file",
        default="dataset/species_targets.yaml",
        help="Pad naar species_targets.yaml",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    species_file = Path(args.species_file)

    if not species_file.exists():
        print(f"Bestand niet gevonden: {species_file}", file=sys.stderr)
        sys.exit(1)

    species_list = _load_species(species_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Xeno-Canto downloader — {len(species_list)} soort(en), max {args.max_per_species} per soort")
    print(f"Uitvoer: {output_dir.resolve()}\n")

    for species in species_list:
        download_species(species, output_dir, args.max_per_species)

    print("\n✅ Klaar!")


if __name__ == "__main__":
    main()
