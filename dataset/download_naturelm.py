"""
Download zoogdiergeluiden via NatureLM dataset (Earth Species Project) van Hugging Face.

Strategie:
  1. Stream dataset ZONDER audio (cast_column Audio decode=False) → snel scannen
  2. Download alleen audio voor matching IDs via tweede stream pass
  3. Sla op als WAV via soundfile

Gebruik: python dataset/download_naturelm.py --output dataset/raw --species-file dataset/species_targets.yaml
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import wave
from pathlib import Path

import yaml

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    from datasets import load_dataset, Audio
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

NATURELM_DATASET = "EarthSpeciesProject/NatureLM-audio-training"
SAMPLE_RATE = 16000


def _slug(scientific: str) -> str:
    return re.sub(r"\s+", "_", scientific.strip().lower())


def _load_species(yaml_path: Path) -> list[dict]:
    with yaml_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data["species"]


def _scientific_names(species_list: list[dict]) -> set[str]:
    return {s["scientific"].lower() for s in species_list}


def _extract_species_from_output(output: str) -> str:
    """Laatste 2 woorden van taxonomie = genus + soort."""
    if not output or not isinstance(output, str):
        return ""
    words = output.strip().split()
    if len(words) >= 2:
        return " ".join(words[-2:])
    return output.strip()


def _build_index(species_list: list[dict], max_per_species: int, debug: bool) -> dict[str, list[str]]:
    """
    Stap 1: Stream dataset ZONDER audio te decoderen.
    Bouwt index {slug: [id, ...]} van alle matching samples.
    """
    print("📋 Stap 1: Index bouwen (geen audio, alleen tekst)...")
    target_names = _scientific_names(species_list)

    try:
        ds_meta = load_dataset(NATURELM_DATASET, split="train", streaming=True)
        # Geen audio decoding — alleen tekst kolommen lezen
        ds_meta = ds_meta.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        print(f"⚠ Dataset laden mislukt: {exc}", file=sys.stderr)
        sys.exit(1)

    index: dict[str, list[str]] = {_slug(s["scientific"]): [] for s in species_list}

    # Check of alle soorten al vol zijn
    def _all_done() -> bool:
        return all(len(index[_slug(s["scientific"])]) >= max_per_species for s in species_list)

    iterator = tqdm(ds_meta, desc="Index scannen", unit=" samples") if HAS_TQDM else ds_meta

    for sample in iterator:
        if _all_done():
            break

        # Veilige task-check: task kan None zijn
        task = sample.get("task") or ""
        if "taxonomic" not in task:
            continue

        output = sample.get("output") or ""
        species_name = _extract_species_from_output(output).lower()

        if species_name not in target_names:
            continue

        slug = _slug(species_name)
        sample_id = sample.get("id") or ""
        if sample_id and len(index[slug]) < max_per_species:
            index[slug].append(sample_id)
            if debug:
                print(f"  ✓ {species_name} → {sample_id}")

    return index


def _save_audio_as_wav(audio_data, dest: Path, sample_rate: int = SAMPLE_RATE) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if isinstance(audio_data, dict):
            array = audio_data.get("array")
            sr = audio_data.get("sampling_rate", sample_rate)
            if array is not None and HAS_SOUNDFILE:
                if HAS_NUMPY:
                    array = np.array(array, dtype=np.float32)
                sf.write(str(dest), array, sr, subtype="PCM_16")
                return True

        if isinstance(audio_data, (bytes, bytearray)) and HAS_SOUNDFILE:
            buf = io.BytesIO(audio_data)
            array, sr = sf.read(buf, dtype="float32", always_2d=False)
            sf.write(str(dest), array, sr, subtype="PCM_16")
            return True

        import struct
        if isinstance(audio_data, dict):
            samples = audio_data.get("array", [])
            sr = audio_data.get("sampling_rate", sample_rate)
        else:
            samples = list(audio_data) if audio_data else []
            sr = sample_rate
        if not samples:
            return False
        pcm = [max(-32768, min(32767, int(float(s) * 32767))) for s in samples]
        raw = struct.pack(f"<{len(pcm)}h", *pcm)
        with wave.open(str(dest), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(raw)
        return True
    except Exception as exc:
        print(f"\n  ⚠ Audio opslaan mislukt: {exc}", file=sys.stderr)
        return False


def _save_metadata(records: list[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def download_from_naturelm(
    species_list: list[dict],
    output_dir: Path,
    max_per_species: int,
    debug: bool = False,
) -> dict[str, int]:
    if not HAS_DATASETS:
        print("⚠ pip install datasets", file=sys.stderr)
        sys.exit(1)

    species_by_name = {s["scientific"].lower(): s for s in species_list}
    counters: dict[str, int] = {_slug(s["scientific"]): 0 for s in species_list}
    metadata_buffers: dict[str, list[dict]] = {_slug(s["scientific"]): [] for s in species_list}

    # === STAP 1: Index bouwen (geen audio) ===
    index = _build_index(species_list, max_per_species, debug)

    total_matches = sum(len(v) for v in index.values())
    print(f"\n  Totaal: {total_matches} opnames gevonden")
    for species_info in species_list:
        slug = _slug(species_info["scientific"])
        n = len(index.get(slug, []))
        print(f"  {'✓' if n > 0 else '✗'} {species_info['nl']:20s}: {n} opnames")

    if total_matches == 0:
        print("\n⚠ Geen matches gevonden.", file=sys.stderr)
        return counters

    # Maak set van gewenste IDs
    wanted_ids: dict[str, str] = {}
    for slug, ids in index.items():
        for sample_id in ids[:max_per_species]:
            wanted_ids[sample_id] = slug

    print(f"\n📡 Stap 2: Audio downloaden voor {len(wanted_ids)} opnames...")

    # === STAP 2: Audio ophalen (met decoding) ===
    try:
        ds_audio = load_dataset(NATURELM_DATASET, split="train", streaming=True)
    except Exception as exc:
        print(f"⚠ Dataset laden mislukt: {exc}", file=sys.stderr)
        sys.exit(1)

    remaining = set(wanted_ids.keys())
    iterator = tqdm(ds_audio, desc="Audio ophalen", unit=" samples") if HAS_TQDM else ds_audio

    for sample in iterator:
        if not remaining:
            break
        sample_id = sample.get("id") or ""
        if sample_id not in remaining:
            continue
        remaining.discard(sample_id)
        slug = wanted_ids[sample_id]
        if counters[slug] >= max_per_species:
            continue

        output = sample.get("output") or ""
        species_name = _extract_species_from_output(output).lower()
        species_info = species_by_name.get(species_name, {})

        dest = output_dir / slug / f"{sample_id}.wav"
        if dest.exists():
            counters[slug] += 1
            continue

        audio = sample.get("audio")
        if not audio:
            continue

        if _save_audio_as_wav(audio, dest):
            counters[slug] += 1
            meta = {k: v for k, v in sample.items() if k not in ("audio", "audio_array")}
            meta["source"] = NATURELM_DATASET
            meta["local_file"] = str(dest)
            metadata_buffers[slug].append(meta)
            if debug and species_info:
                print(f"  ✓ {species_info.get('nl', slug)}: {dest.name}")

    for species_info in species_list:
        slug = _slug(species_info["scientific"])
        meta_list = metadata_buffers[slug]
        if meta_list:
            _save_metadata(meta_list, output_dir / slug / "metadata.jsonl")

    return counters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dataset/raw")
    parser.add_argument("--max-per-species", type=int, default=50)
    parser.add_argument("--species-file", default="dataset/species_targets.yaml")
    parser.add_argument("--dataset", default=NATURELM_DATASET)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output)
    species_file = Path(args.species_file)

    if not species_file.exists():
        print(f"Bestand niet gevonden: {species_file}", file=sys.stderr)
        sys.exit(1)
    if not HAS_SOUNDFILE:
        print("⚠ pip install soundfile", file=sys.stderr)
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
