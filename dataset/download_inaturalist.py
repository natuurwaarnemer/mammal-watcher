"""
Download zoogdiergeluiden van iNaturalist als aanvulling op de dataset.
Geen API key vereist — iNaturalist is open access.

Gebruik:
    python dataset/download_inaturalist.py --output dataset/raw
    python dataset/download_inaturalist.py --output /mnt/usb/audio --max-per-species 100
    python dataset/download_inaturalist.py --species-file species_config.json

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

import requests

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

INAT_API_BASE = "https://api.inaturalist.org/v1"
DOWNLOAD_DELAY = 0.5  # seconden tussen downloads (respecteer de server)
PAGE_SIZE = 200  # maximaal toegestaan door iNaturalist API


def _slug(scientific: str) -> str:
    """Zet wetenschappelijke naam om naar bestandsvriendelijke slug."""
    return re.sub(r"\s+", "_", scientific.strip().lower())


def _load_species(species_file: Path) -> list[dict]:
    """Laad soorten uit species_config.json of species_targets.yaml."""
    suffix = species_file.suffix.lower()
    with species_file.open(encoding="utf-8") as fh:
        if suffix == ".json":
            data = json.load(fh)
            species_list = data["species"]
            # Zorg dat alleen soorten met een taxon_id worden meegenomen
            return [s for s in species_list if s.get("inaturalist_taxon_id")]
        else:
            # YAML (species_targets.yaml of vergelijkbaar)
            try:
                import yaml
            except ImportError:
                print("⚠ pip install pyyaml voor YAML ondersteuning", file=sys.stderr)
                sys.exit(1)
            data = yaml.safe_load(fh)
            species_list = data["species"]
            return [s for s in species_list if s.get("inaturalist_taxon_id")]


def _fetch_observations_page(taxon_id: int, page: int) -> dict:
    """Haal één pagina observaties op van de iNaturalist API."""
    url = f"{INAT_API_BASE}/observations"
    params = {
        "taxon_id": taxon_id,
        "sounds": "true",
        "per_page": PAGE_SIZE,
        "page": page,
        "order": "desc",
        "order_by": "created_at",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _collect_sound_urls(observations: list[dict]) -> list[dict]:
    """Extraheer geluidsbestanden uit observaties."""
    sounds = []
    for obs in observations:
        obs_id = obs.get("id")
        license_code = obs.get("license_code", "unknown")
        taxon = obs.get("taxon", {})
        for sound in obs.get("sounds", []):
            url = sound.get("file_url") or sound.get("file")
            if not url:
                continue
            sounds.append({
                "observation_id": obs_id,
                "sound_id": sound.get("id"),
                "url": url,
                "license": license_code,
                "taxon_id": taxon.get("id"),
            })
    return sounds


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
    Download iNaturalist geluidsopnames voor één soort.
    Geeft het aantal nieuw gedownloade bestanden terug.
    """
    scientific = species["scientific"]
    taxon_id = species["inaturalist_taxon_id"]
    slug = _slug(scientific)
    species_dir = output_dir / slug
    nl_name = species.get("nl", scientific)

    print(f"\n→ {nl_name} ({scientific}) [taxon_id={taxon_id}]")

    # Verzamel alle beschikbare geluids-URLs via paginering
    all_sounds: list[dict] = []
    page = 1
    total_results = None

    while True:
        try:
            data = _fetch_observations_page(taxon_id, page)
        except requests.RequestException as exc:
            print(f"  ⚠ API verzoek mislukt (pagina {page}): {exc}", file=sys.stderr)
            break

        if total_results is None:
            total_results = data.get("total_results", 0)
            print(f"  {total_results} observatie(s) met geluid gevonden")

        observations = data.get("results", [])
        if not observations:
            break

        sounds = _collect_sound_urls(observations)
        all_sounds.extend(sounds)

        if max_per_species is not None and len(all_sounds) >= max_per_species:
            all_sounds = all_sounds[:max_per_species]
            break

        # Controleer of er meer pagina's zijn
        fetched_so_far = (page - 1) * PAGE_SIZE + len(observations)
        if fetched_so_far >= (total_results or 0):
            break

        page += 1
        time.sleep(0.2)  # korte pauze tussen pagina-verzoeken

    if not all_sounds:
        print("  Geen geluidsbestanden gevonden.")
        return 0

    print(f"  {len(all_sounds)} geluidsbestand(en) beschikbaar, downloaden...")

    iterator = tqdm(all_sounds, desc=slug) if HAS_TQDM else all_sounds
    downloaded = 0
    skipped = 0
    metadata_records: list[dict] = []

    for sound in iterator:
        obs_id = sound["observation_id"]
        url = sound["url"]

        # Bepaal bestandsextensie op basis van URL
        ext = "mp3"
        url_lower = url.lower().split("?")[0]
        if url_lower.endswith(".ogg"):
            ext = "ogg"
        elif url_lower.endswith(".wav"):
            ext = "wav"
        elif url_lower.endswith(".flac"):
            ext = "flac"

        filename = f"inat_{obs_id}.{ext}"
        dest = species_dir / filename

        try:
            is_new = _download_sound(url, dest)
        except requests.RequestException as exc:
            print(f"  ⚠ Download mislukt ({filename}): {exc}", file=sys.stderr)
            continue

        if is_new:
            downloaded += 1
            metadata_records.append({
                "source": "inaturalist",
                "observation_id": obs_id,
                "sound_id": sound.get("sound_id"),
                "taxon_id": sound.get("taxon_id") or taxon_id,
                "license": sound.get("license"),
                "url": url,
                "local_file": str(dest),
                "species_scientific": scientific,
                "species_nl": nl_name,
            })
            time.sleep(DOWNLOAD_DELAY)
        else:
            skipped += 1

    if metadata_records:
        _save_metadata(metadata_records, species_dir / "metadata.jsonl")

    print(f"  ✓ {downloaded} nieuw gedownload, {skipped} overgeslagen.")
    return downloaded


def main() -> None:
    # Bepaal standaard pad naar species_config.json relatief aan dit script
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
        print("Geen soorten gevonden met een inaturalist_taxon_id.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    max_label = str(args.max_per_species) if args.max_per_species else "alles"
    print(f"iNaturalist downloader — {len(species_list)} soort(en), max {max_label} per soort")
    print(f"Uitvoer: {output_dir.resolve()}\n")

    counts: list[tuple[str, str, int]] = []

    for species in species_list:
        n = download_species(species, output_dir, args.max_per_species)
        counts.append((species.get("nl", "?"), species["scientific"], n))

    # Overzichtstabel
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
