"""
Download achtergrond-clips van een lokale BirdNET-Go instantie voor de background-klasse.

Het script haalt vogel-detecties op via de BirdNET-Go API, downloadt de AAC-clips
en converteert ze naar WAV (16 kHz mono) — hetzelfde formaat als de rest van prepared/.

Gebruik (alle soorten, max 50 per soort):
    python dataset/download_birdnet_clips.py \
        --api http://192.168.2.23:8080 \
        --output /mnt/usb/prepared/background \
        --index /mnt/usb/prepared/index.csv \
        --clips 500 \
        --min-confidence 0.85

Gebruik (gericht op corviden — kraai/roek/kauw als background):
    python dataset/download_birdnet_clips.py \
        --api http://192.168.2.23:8080 \
        --output /mnt/usb/prepared/background \
        --index /mnt/usb/prepared/index.csv \
        --clips 900 \
        --min-confidence 0.80 \
        --species "Corvus frugilegus,Corvus corone,Corvus monedula"

Na afloop staan de WAV-bestanden in --output en zijn de paden toegevoegd aan index.csv.
Daarna: extract_embeddings.py opnieuw draaien met --force.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError
from urllib.parse import urlencode
import json


API_MAX_LIMIT = 1000
TARGET_SAMPLE_RATE = 16000
SPECIES_LABEL = "background"


def _fetch_detections(
    api_base: str,
    min_confidence: float,
    species_filter: list[str] | None = None,
    max_pages: int = 30,
) -> list[dict]:
    """Haal detecties op van BirdNET-Go API, optioneel gefilterd per soort."""
    if species_filter:
        detections: list[dict] = []
        for species in species_filter:
            detections.extend(
                _fetch_detections_for_species(api_base, min_confidence, species, max_pages)
            )
        return detections

    return _fetch_detections_for_species(api_base, min_confidence, None, max_pages)


def _fetch_detections_for_species(
    api_base: str,
    min_confidence: float,
    species: str | None,
    max_pages: int,
) -> list[dict]:
    detections: list[dict] = []
    offset = 0
    for _ in range(max_pages):
        params: dict = {"limit": API_MAX_LIMIT, "offset": offset}
        if species:
            params["species"] = species
        url = f"{api_base}/api/v2/detections?{urlencode(params)}"
        try:
            with urlopen(Request(url), timeout=10) as resp:
                data = json.loads(resp.read())
        except (URLError, json.JSONDecodeError) as exc:
            print(f"Fout bij ophalen detecties (offset={offset}): {exc}", file=sys.stderr)
            break
        batch = data.get("data", [])
        if not batch:
            break
        for det in batch:
            if det.get("confidence", 0) >= min_confidence:
                detections.append(det)
        offset += API_MAX_LIMIT
        if offset >= data.get("total", 0):
            break
    return detections


def _select_ids(
    detections: list[dict],
    target_total: int,
) -> list[tuple[int, str]]:
    """Kies clips zo dat soorten evenredig vertegenwoordigd zijn."""
    by_species: dict[str, list[int]] = defaultdict(list)
    for det in detections:
        by_species[det["scientificName"]].append(det["id"])

    max_per_species = max(1, target_total // max(len(by_species), 1))

    selected: list[tuple[int, str]] = []
    for species, ids in sorted(by_species.items(), key=lambda x: -len(x[1])):
        for clip_id in ids[:max_per_species]:
            selected.append((clip_id, species))
        if len(selected) >= target_total:
            break
    return selected[:target_total]


def _download_and_convert(
    api_base: str,
    clip_id: int,
    out_path: Path,
) -> bool:
    """Download AAC clip en converteer naar WAV 16kHz mono via ffmpeg."""
    url = f"{api_base}/api/v2/audio/{clip_id}"
    tmp_aac = out_path.with_suffix(".aac.tmp")
    try:
        with urlopen(Request(url), timeout=20) as resp:
            tmp_aac.write_bytes(resp.read())
    except URLError as exc:
        print(f"  Download mislukt {clip_id}: {exc}", file=sys.stderr)
        return False

    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(tmp_aac),
            "-ar", str(TARGET_SAMPLE_RATE),
            "-ac", "1",
            "-sample_fmt", "s16",
            str(out_path),
        ],
        capture_output=True,
    )
    tmp_aac.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  ffmpeg fout voor {clip_id}: {result.stderr[-200:]}", file=sys.stderr)
        return False
    return True


def _load_existing_index(index_path: Path) -> set[str]:
    """Geef alle bekende bestandspaden terug om duplicaten te voorkomen."""
    if not index_path.exists():
        return set()
    with index_path.open(newline="", encoding="utf-8") as fh:
        return {row["file"] for row in csv.DictReader(fh)}


def _append_to_index(index_path: Path, rows: list[dict]) -> None:
    write_header = not index_path.exists() or index_path.stat().st_size == 0
    with index_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "species_scientific"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://192.168.2.23:8080",
                        help="BirdNET-Go API base URL")
    parser.add_argument("--output", required=True,
                        help="Uitvoermap voor WAV-bestanden (bijv. /mnt/usb/prepared/background)")
    parser.add_argument("--index", required=True,
                        help="Pad naar prepared/index.csv")
    parser.add_argument("--clips", type=int, default=500,
                        help="Aantal te downloaden clips")
    parser.add_argument("--min-confidence", type=float, default=0.85,
                        help="Minimale BirdNET-Go confidence")
    parser.add_argument("--species", default=None,
                        help="Kommagescheiden lijst van wetenschappelijke namen om te filteren "
                             "(bijv. 'Corvus frugilegus,Corvus corone'). "
                             "Standaard: alle soorten, max 50 per soort.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(args.index)

    species_filter = [s.strip() for s in args.species.split(",")] if args.species else None

    print(f"BirdNET-Go API: {args.api}")
    print(f"Uitvoer: {out_dir}")
    print(f"Doel: {args.clips} clips, min confidence {args.min_confidence}")
    if species_filter:
        print(f"Soortfilter: {', '.join(species_filter)}")

    existing = _load_existing_index(index_path)
    print(f"Al in index: {len(existing)} bestanden")

    print("Detecties ophalen...")
    detections = _fetch_detections(args.api, args.min_confidence, species_filter)
    print(f"  {len(detections)} detecties met confidence >= {args.min_confidence}")

    if not detections:
        print("Geen detecties gevonden. Controleer de API-URL of soortnaam.", file=sys.stderr)
        sys.exit(1)

    selected = _select_ids(detections, args.clips)
    print(f"  {len(selected)} clips geselecteerd uit {len(set(s for _, s in selected))} soorten")

    new_rows: list[dict] = []
    ok = skip = fail = 0

    for i, (clip_id, bird_species) in enumerate(selected, 1):
        wav_path = out_dir / f"birdnet_{clip_id}.wav"
        if str(wav_path) in existing or wav_path.exists():
            skip += 1
            continue

        print(f"[{i}/{len(selected)}] {clip_id} ({bird_species})", end=" ")
        if _download_and_convert(args.api, clip_id, wav_path):
            print("✓")
            new_rows.append({"file": str(wav_path), "species_scientific": SPECIES_LABEL})
            ok += 1
        else:
            fail += 1

        if new_rows and len(new_rows) % 50 == 0:
            _append_to_index(index_path, new_rows)
            existing.update(r["file"] for r in new_rows)
            new_rows = []
            time.sleep(0.2)

    if new_rows:
        _append_to_index(index_path, new_rows)

    print(f"\nKlaar: {ok} nieuw, {skip} overgeslagen, {fail} mislukt")
    print(f"Index bijgewerkt: {index_path}")


if __name__ == "__main__":
    main()
