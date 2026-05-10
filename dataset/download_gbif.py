"""
Download zoogdiergeluiden van GBIF als tweede bron naast iNaturalist.
Geen API key vereist — GBIF openbare data.

Gebruik:
    python dataset/download_gbif.py --output dataset/raw
    python dataset/download_gbif.py --output /mnt/usb/audio --max-per-species 100
    python dataset/download_gbif.py --species-file species_config.json

Laadt soorten standaard uit species_config.json (root van de repo).
Kan ook species_targets.yaml lezen als alternatief via --species-file.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

GBIF_API_BASE = "https://api.gbif.org/v1"
INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
DOWNLOAD_DELAY = 0.5  # seconden tussen downloads (respecteer de server)
PAGE_REQUEST_DELAY = 0.2  # seconden tussen pagina-verzoeken
PAGE_SIZE = 300  # maximaal toegestaan door GBIF API

GBIF_TAXON_KEYS = {
    "Vulpes vulpes": 5219243,
    "Canis lupus": 5219206,
    "Canis aureus": 5219218,
    "Martes martes": 5219030,
    "Martes foina": 5219032,
    "Meles meles": 5219047,
    "Lutra lutra": 5219035,
    "Capreolus capreolus": 2440897,
    "Cervus elaphus": 2440905,
    "Sus scrofa": 5219416,
    "Castor fiber": 5820003,
    "Lynx lynx": 5219072,
}


def _slug(scientific: str) -> str:
    """Zet wetenschappelijke naam om naar bestandsvriendelijke slug."""
    return re.sub(r"\s+", "_", scientific.strip().lower())


def _safe_occurrence_id(value: str) -> str:
    """Normaliseer occurrenceID voor veilig gebruik in bestandsnamen."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "unknown"


def _load_species(species_file: Path) -> list[dict]:
    """Laad soorten uit species_config.json of species_targets.yaml."""
    suffix = species_file.suffix.lower()
    with species_file.open(encoding="utf-8") as fh:
        if suffix == ".json":
            data = json.load(fh)
            species_list = data["species"]
        else:
            try:
                import yaml
            except ImportError:
                print("⚠ pip install pyyaml voor YAML ondersteuning", file=sys.stderr)
                sys.exit(1)
            data = yaml.safe_load(fh)
            species_list = data["species"]

    selected: list[dict] = []
    for species in species_list:
        scientific = species.get("scientific")
        taxon_key = species.get("gbif_taxon_key") or GBIF_TAXON_KEYS.get(scientific)
        if taxon_key:
            enriched = dict(species)
            enriched["gbif_taxon_key"] = int(taxon_key)
            selected.append(enriched)
    return selected


def _fetch_occurrences_page(taxon_key: int, offset: int) -> dict:
    """Haal één pagina occurrences op van de GBIF API."""
    url = f"{GBIF_API_BASE}/occurrence/search"
    params = {
        "mediaType": "Sound",
        "taxonKey": taxon_key,
        "excludeDatasetKey": INAT_DATASET_KEY,
        "limit": PAGE_SIZE,
        "offset": offset,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _collect_sound_media(occurrences: list[dict]) -> list[dict]:
    """Extraheer geluidsmedia uit GBIF occurrences."""
    sounds: list[dict] = []
    for occurrence in occurrences:
        occurrence_id = occurrence.get("occurrenceID") or str(occurrence.get("key") or "")
        if not occurrence_id:
            continue

        media_items = occurrence.get("media") or []
        for media in media_items:
            if media.get("type", "").lower() != "sound":
                continue
            url = media.get("identifier")
            if not url:
                continue
            sounds.append({
                "occurrence_id": occurrence_id,
                "taxon_key": occurrence.get("taxonKey"),
                "url": url,
                "license": media.get("license") or occurrence.get("license"),
                "format": media.get("format"),
            })
    return sounds


def _guess_extension(url: str, media_format: str | None) -> str:
    """Bepaal bestandsextensie op basis van URL en format."""
    path = urlparse(url).path.lower()
    if "." in path:
        suffix = path.rsplit(".", 1)[-1]
        if suffix in {"mp3", "wav", "ogg", "flac", "m4a", "aac", "opus"}:
            return suffix

    format_l = (media_format or "").lower()
    if "mpeg" in format_l or "mp3" in format_l:
        return "mp3"
    if "wav" in format_l:
        return "wav"
    if "ogg" in format_l or "opus" in format_l:
        return "ogg"
    if "flac" in format_l:
        return "flac"
    if "aac" in format_l:
        return "aac"
    if "m4a" in format_l or "mp4" in format_l:
        return "m4a"
    return "bin"


def _download_sound(url: str, dest: Path) -> bool:
    """
    Download een geluidsbestand naar dest.
    Geeft True terug als het nieuw gedownload is, False als het al bestond.
    """
    if dest.exists():
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, timeout=60, stream=True)
    resp.raise_for_status()
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
    max_per_species: int | None,
) -> int:
    """
    Download GBIF geluidsopnames voor één soort.
    Geeft het aantal nieuw gedownloade bestanden terug.
    """
    scientific = species["scientific"]
    taxon_key = species["gbif_taxon_key"]
    slug = _slug(scientific)
    species_dir = output_dir / slug
    nl_name = species.get("nl", scientific)

    print(f"\n→ {nl_name} ({scientific}) [taxonKey={taxon_key}]")

    all_sounds: list[dict] = []
    offset = 0
    total_results = None

    while True:
        try:
            data = _fetch_occurrences_page(taxon_key, offset)
        except requests.RequestException as exc:
            print(f"  ⚠ API verzoek mislukt (offset {offset}): {exc}", file=sys.stderr)
            break

        if total_results is None:
            total_results = data.get("count", 0)
            print(f"  {total_results} occurrence(s) met geluid gevonden")

        occurrences = data.get("results", [])
        if not occurrences:
            break

        sounds = _collect_sound_media(occurrences)
        all_sounds.extend(sounds)

        if max_per_species is not None and len(all_sounds) >= max_per_species:
            all_sounds = all_sounds[:max_per_species]
            break

        offset += len(occurrences)
        if offset >= (total_results or 0):
            break

        time.sleep(PAGE_REQUEST_DELAY)

    if not all_sounds:
        print("  Geen geluidsbestanden gevonden.")
        return 0

    print(f"  {len(all_sounds)} geluidsbestand(en) beschikbaar, downloaden...")

    iterator = tqdm(all_sounds, desc=slug) if HAS_TQDM else all_sounds
    downloaded = 0
    skipped = 0
    metadata_records: list[dict] = []

    total_sounds = len(all_sounds)
    for idx, sound in enumerate(iterator, start=1):
        occurrence_id = str(sound["occurrence_id"])
        url = sound["url"]
        ext = _guess_extension(url, sound.get("format"))
        safe_occurrence_id = _safe_occurrence_id(occurrence_id)

        filename = f"gbif_{safe_occurrence_id}.{ext}"
        dest = species_dir / filename

        try:
            is_new = _download_sound(url, dest)
        except requests.RequestException as exc:
            print(f"  ⚠ Download mislukt ({filename}): {exc}", file=sys.stderr)
            continue

        if is_new:
            downloaded += 1
            record_taxon_key = sound["taxon_key"] if sound["taxon_key"] is not None else taxon_key
            metadata_records.append({
                "source": "gbif",
                "occurrenceID": occurrence_id,
                "taxonKey": record_taxon_key,
                "license": sound.get("license"),
                "url": url,
                "local_file": str(dest),
                "species_scientific": scientific,
                "species_nl": nl_name,
            })
            time.sleep(DOWNLOAD_DELAY)
        else:
            skipped += 1

        if not HAS_TQDM and (idx % 25 == 0 or idx == total_sounds):
            print(f"  Voortgang: {idx}/{total_sounds}")

    if metadata_records:
        _save_metadata(metadata_records, species_dir / "metadata.jsonl")

    print(f"  ✓ {downloaded} nieuw gedownload, {skipped} overgeslagen.")
    return downloaded


def main() -> None:
    _script_dir = Path(__file__).resolve().parent
    _default_species_file = _script_dir.parent / "species_config.json"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="dataset/raw",
        help="Uitvoermap voor ruwe downloads (standaard: dataset/raw)",
    )
    parser.add_argument(
        "--max-per-species",
        type=int,
        default=None,
        help="Maximum aantal opnames per soort (standaard: alles)",
    )
    parser.add_argument(
        "--species-file",
        default=str(_default_species_file),
        help="Pad naar species_config.json of species_targets.yaml",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    species_file = Path(args.species_file)

    if not species_file.exists():
        print(f"Bestand niet gevonden: {species_file}", file=sys.stderr)
        sys.exit(1)

    species_list = _load_species(species_file)
    if not species_list:
        print("Geen soorten gevonden met een gbif_taxon_key.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    max_label = str(args.max_per_species) if args.max_per_species else "alles"
    print(f"GBIF downloader — {len(species_list)} soort(en), max {max_label} per soort")
    print(f"Uitvoer: {output_dir.resolve()}\n")

    counts: list[tuple[str, str, int]] = []

    for species in species_list:
        n = download_species(species, output_dir, args.max_per_species)
        counts.append((species.get("nl", "?"), species["scientific"], n))

    total = sum(c[2] for c in counts)
    print("\n" + "─" * 55)
    print(f"{'Soort':<20} {'Wetenschappelijk':<25} {'Nieuw':>6}")
    print("─" * 55)
    for nl, sci, n in counts:
        print(f"{nl:<20} {sci:<25} {n:>6}")
    print("─" * 55)
    print(f"{'TOTAAL':<46} {total:>6}")
    print("\n✅ Klaar!")


if __name__ == "__main__":
    main()
