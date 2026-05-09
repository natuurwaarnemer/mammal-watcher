"""
Download zoogdiergeluiden via NatureLM dataset (Earth Species Project) van Hugging Face.
Gebruik: python dataset/download_naturelm.py --output dataset/raw --species-file dataset/species_targets.yaml
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from pathlib import Path

import yaml

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Primaire dataset: NatureLM-audio van Earth Species Project
NATURELM_DATASET = "EarthSpeciesProject/NatureLM-audio-training"
# Bekende kolomnamen voor soortnaam (directe kolommen)
SPECIES_COLUMNS = ["scientific_name", "species", "label", "common_name"]
# Sleutels binnen een 'metadata' dict kolom
METADATA_SPECIES_KEYS = ["scientific_name", "species", "Scientific Name", "Species"]
SAMPLE_RATE = 16000  # Hz — standaard voor YAMNet


def _slug(scientific: str) -> str:
    """Zet wetenschappelijke naam om naar bestandsvriendelijke slug."""
    return re.sub(r"\s+", "_", scientific.strip().lower())


def _load_species(yaml_path: Path) -> list[dict]:
    with yaml_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data["species"]


def _scientific_names(species_list: list[dict]) -> set[str]:
    """Geef set van wetenschappelijke namen terug (lowercase voor vergelijking)."""
    return {s["scientific"].lower() for s in species_list}


def _find_species_column(features: dict) -> str | None:
    """Zoek de kolom met de wetenschappelijke naam in het dataset-schema."""
    for col in SPECIES_COLUMNS:
        if col in features:
            return col
    return None


def _extract_species_from_sample(sample: dict) -> str:
    """
    Haal soortnaam op uit een sample.
    Probeert directe kolommen, daarna metadata dict, daarna instruction_text.
    """
    # 1. Directe kolommen
    for col in SPECIES_COLUMNS:
        val = sample.get(col)
        if val and isinstance(val, str):
            return val.strip()

    # 2. metadata kolom (kan dict of JSON-string zijn)
    meta = sample.get("metadata")
    if meta:
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        if isinstance(meta, dict):
            for key in METADATA_SPECIES_KEYS:
                val = meta.get(key)
                if val and isinstance(val, str):
                    return val.strip()

    # 3. output kolom (NatureLM gebruikt dit als beschrijving/label)
    output = sample.get("output", "")
    if output and isinstance(output, str):
        return output.strip()

    return ""


def _save_audio_as_wav(audio_data: dict | list, dest: Path, sample_rate: int = SAMPLE_RATE) -> None:
    """Sla audio op als WAV-bestand (16kHz mono)."""
    import struct

    if isinstance(audio_data, dict):
        samples = audio_data.get("array", [])
        sr = audio_data.get("sampling_rate", sample_rate)
    else:
        samples = audio_data
        sr = sample_rate

    # Converteer naar 16-bit PCM
    pcm = [max(-32768, min(32767, int(s * 32767))) for s in samples]
    raw = struct.pack(f"<{len(pcm)}h", *pcm)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(raw)


def _save_metadata(records: list[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _print_sample_debug(sample: dict) -> None:
    """Print eerste sample voor debugging van dataset-schema."""
    print("\n🔍 Eerste sample (schema debug):")
    for k, v in sample.items():
        if k == "audio":
            print(f"  {k}: <audio data>")
        elif isinstance(v, str) and len(v) > 100:
            print(f"  {k}: {v[:100]}...")
        else:
            print(f"  {k}: {v!r}")
    print()


def download_from_naturelm(
    species_list: list[dict],
    output_dir: Path,
    max_per_species: int,
    debug: bool = False,
) -> dict[str, int]:
    """
    Stream de NatureLM dataset en sla audio op per doelsoort.
    Geeft dict terug met {slug: aantal_downloads}.
    """
    if not HAS_DATASETS:
        print(
            "⚠ 'datasets' library niet gevonden. Installeer met: pip install datasets>=2.14",
            file=sys.stderr,
        )
        sys.exit(1)

    target_names = _scientific_names(species_list)
    species_by_name = {s["scientific"].lower(): s for s in species_list}

    counters: dict[str, int] = {_slug(s["scientific"]): 0 for s in species_list}
    metadata_buffers: dict[str, list[dict]] = {_slug(s["scientific"]): [] for s in species_list}

    print(f"📡 Verbinden met dataset: {NATURELM_DATASET} ...")
    try:
        ds = load_dataset(NATURELM_DATASET, split="train", streaming=True)
    except Exception as exc:
        print(f"⚠ Dataset kon niet geladen worden: {exc}", file=sys.stderr)
        print("  Controleer of je internettoegang hebt en de dataset beschikbaar is.", file=sys.stderr)
        sys.exit(1)

    print(f"  Beschikbare kolommen: {list(ds.features.keys())}")
    print(f"  Zoek naar {len(species_list)} soort(en), max {max_per_species} per soort\n")

    all_done = False
    processed = 0
    first_sample_shown = False

    iterator = tqdm(ds, desc="Streamen", unit=" samples") if HAS_TQDM else ds

    for sample in iterator:
        if all_done:
            break

        # Toon eerste sample voor debug
        if not first_sample_shown and debug:
            _print_sample_debug(sample)
            first_sample_shown = True

        name_raw = _extract_species_from_sample(sample)
        name_lower = name_raw.lower().strip()

        if not name_lower or name_lower not in target_names:
            continue

        species_info = species_by_name[name_lower]
        slug = _slug(species_info["scientific"])

        if counters[slug] >= max_per_species:
            if all(counters[s] >= max_per_species for s in counters):
                all_done = True
            continue

        sample_id = sample.get("id", sample.get("audio_id", f"nlm_{processed}_{slug}"))
        dest = output_dir / slug / f"{sample_id}.wav"

        if dest.exists():
            counters[slug] += 1
            processed += 1
        else:
            audio = sample.get("audio", sample.get("audio_array", []))
            if not audio:
                continue
            try:
                sr = SAMPLE_RATE
                if isinstance(audio, dict):
                    sr = audio.get("sampling_rate", SAMPLE_RATE)
                _save_audio_as_wav(audio, dest, sr)
                counters[slug] += 1
                processed += 1

                meta = {k: v for k, v in sample.items() if k != "audio" and k != "audio_array"}
                meta["source"] = NATURELM_DATASET
                meta["local_file"] = str(dest)
                metadata_buffers[slug].append(meta)
            except Exception as exc:
                print(f"\n  ⚠ Opslaan mislukt ({dest.name}): {exc}", file=sys.stderr)

    for species_info in species_list:
        slug = _slug(species_info["scientific"])
        meta_list = metadata_buffers[slug]
        if meta_list:
            meta_dest = output_dir / slug / "metadata.jsonl"
            _save_metadata(meta_list, meta_dest)

    return counters


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
        default=50,
        help="Maximum aantal opnames per soort (standaard: 50)",
    )
    parser.add_argument(
        "--species-file",
        default="dataset/species_targets.yaml",
        help="Pad naar species_targets.yaml",
    )
    parser.add_argument(
        "--dataset",
        default=NATURELM_DATASET,
        help=f"HuggingFace dataset naam (standaard: {NATURELM_DATASET})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Toon eerste sample voor schema-debugging",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    species_file = Path(args.species_file)

    if not species_file.exists():
        print(f"Bestand niet gevonden: {species_file}", file=sys.stderr)
        sys.exit(1)

    species_list = _load_species(species_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"NatureLM downloader — {len(species_list)} soort(en), max {args.max_per_species} per soort")
    print(f"Dataset: {args.dataset}")
    print(f"Uitvoer: {output_dir.resolve()}\n")

    counters = download_from_naturelm(species_list, output_dir, args.max_per_species, debug=args.debug)

    print("\n📊 Resultaat:")
    total = 0
    for species_info in species_list:
        slug = _slug(species_info["scientific"])
        count = counters[slug]
        total += count
        status = "✓" if count > 0 else "✗"
        print(f"  {status} {species_info['nl']:20s} ({species_info['scientific']}): {count}")

    print(f"\n✅ Klaar! Totaal: {total} opname(s) in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
