"""
Download zoogdiergeluiden via NatureLM dataset (Earth Species Project) van Hugging Face.

Strategie:
  1. Download alleen de metadata-kolommen via Parquet (geen audio) → razendsnel filteren
  2. Haal alleen audio op voor de rijen die matchen met doelsoorten
  3. Sla op als WAV via soundfile (geen torchcodec nodig)

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
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Primaire dataset
NATURELM_DATASET = "EarthSpeciesProject/NatureLM-audio-training"
SAMPLE_RATE = 16000  # Hz — standaard voor YAMNet


def _slug(scientific: str) -> str:
    return re.sub(r"\s+", "_", scientific.strip().lower())


def _load_species(yaml_path: Path) -> list[dict]:
    with yaml_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data["species"]


def _scientific_names(species_list: list[dict]) -> set[str]:
    return {s["scientific"].lower() for s in species_list}


def _extract_species_from_output(output: str) -> str:
    """
    NatureLM output kolom bevat volledige taxonomie, bijv.:
    'Chordata Mammalia Carnivora Canidae Vulpes vulpes'
    De soortnaam zijn altijd de laatste 2 woorden (genus + soort).
    """
    if not output:
        return ""
    words = output.strip().split()
    if len(words) >= 2:
        return " ".join(words[-2:])
    return output.strip()


def _save_audio_as_wav(audio_data, dest: Path, sample_rate: int = SAMPLE_RATE) -> bool:
    """
    Sla audio op als WAV. Ondersteunt:
    - dict met 'array' + 'sampling_rate' (HuggingFace Audio object)
    - bytes (ruwe audio bytes → soundfile decode)
    - numpy array
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        # HuggingFace Audio object: dict met array
        if isinstance(audio_data, dict):
            array = audio_data.get("array")
            sr = audio_data.get("sampling_rate", sample_rate)
            if array is not None and HAS_SOUNDFILE:
                if not isinstance(array, (list,)) and HAS_NUMPY:
                    array = np.array(array, dtype=np.float32)
                sf.write(str(dest), array, sr, subtype="PCM_16")
                return True

        # Ruwe bytes → soundfile decode
        if isinstance(audio_data, (bytes, bytearray)) and HAS_SOUNDFILE:
            buf = io.BytesIO(audio_data)
            array, sr = sf.read(buf, dtype="float32", always_2d=False)
            sf.write(str(dest), array, sr, subtype="PCM_16")
            return True

        # Fallback: lijst/array direct als PCM
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


def _build_index(target_names: set[str], debug: bool) -> dict[str, list[str]]:
    """
    Stap 1: Download alleen tekst-kolommen via Parquet (geen audio).
    Bouw een index: {slug: [id1, id2, ...]} van matching rijen.
    Veel sneller dan audio streamen!
    """
    print("📋 Stap 1: Index bouwen via Parquet (geen audio download)...")

    try:
        # Laad dataset zonder audio te decoderen
        ds_meta = load_dataset(
            NATURELM_DATASET,
            split="train",
            streaming=True,
        )
        # Verwijder audio kolom uit decoding om alleen metadata te lezen
        if "audio" in ds_meta.features:
            from datasets import Audio
            ds_meta = ds_meta.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        print(f"⚠ Index bouwen mislukt: {exc}", file=sys.stderr)
        return {}

    index: dict[str, list[str]] = {slug: [] for slug in [_slug(n) for n in target_names]}
    # slug → scientific mapping
    slug_to_scientific: dict[str, str] = {}

    total_scanned = 0
    total_found = 0

    iterator = tqdm(ds_meta, desc="Index scannen", unit=" samples") if HAS_TQDM else ds_meta

    for sample in iterator:
        total_scanned += 1
        output = sample.get("output", "")
        task = sample.get("task", "")

        # Snel pre-filter: alleen taxonomic-classification samples
        if task and "taxonomic" not in task:
            continue

        species_name = _extract_species_from_output(output).lower()
        if not species_name:
            continue

        if species_name in target_names:
            slug = _slug(species_name)
            sample_id = sample.get("id", "")
            if sample_id and sample_id not in index.get(slug, []):
                if slug not in index:
                    index[slug] = []
                index[slug].append(sample_id)
                slug_to_scientific[slug] = species_name
                total_found += 1

                if debug:
                    print(f"  ✓ Match: {species_name} → {sample_id}")

    print(f"\n  Index klaar: {total_scanned} gescand, {total_found} matches gevonden")
    for slug, ids in index.items():
        if ids:
            print(f"  {slug}: {len(ids)} opnames")

    return index


def download_from_naturelm(
    species_list: list[dict],
    output_dir: Path,
    max_per_species: int,
    debug: bool = False,
) -> dict[str, int]:
    """
    Stap 1: Bouw index via Parquet (alleen tekst, geen audio).
    Stap 2: Download audio alleen voor matching IDs.
    """
    if not HAS_DATASETS:
        print("⚠ 'datasets' niet gevonden. pip install datasets", file=sys.stderr)
        sys.exit(1)

    target_names = _scientific_names(species_list)
    species_by_name = {s["scientific"].lower(): s for s in species_list}
    counters: dict[str, int] = {_slug(s["scientific"]): 0 for s in species_list}
    metadata_buffers: dict[str, list[dict]] = {_slug(s["scientific"]): [] for s in species_list}

    # === STAP 1: Index bouwen ===
    index = _build_index(target_names, debug)

    total_matches = sum(len(v) for v in index.values())
    if total_matches == 0:
        print("\n⚠ Geen matches gevonden in index. Controleer species_targets.yaml.", file=sys.stderr)
        return counters

    # Maak set van gewenste IDs (beperkt tot max_per_species)
    wanted_ids: dict[str, str] = {}  # id → slug
    for slug, ids in index.items():
        for sample_id in ids[:max_per_species]:
            wanted_ids[sample_id] = slug

    print(f"\n📡 Stap 2: Audio downloaden voor {len(wanted_ids)} opnames...")

    # === STAP 2: Audio ophalen voor matching IDs ===
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

        sample_id = sample.get("id", "")
        if sample_id not in remaining:
            continue

        remaining.discard(sample_id)
        slug = wanted_ids[sample_id]

        if counters[slug] >= max_per_species:
            continue

        # Zoek species info
        output = sample.get("output", "")
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
                nl = species_info.get("nl", slug)
                print(f"  ✓ {nl}: {dest.name}")

    # Sla metadata op
    for species_info in species_list:
        slug = _slug(species_info["scientific"])
        meta_list = metadata_buffers[slug]
        if meta_list:
            meta_dest = output_dir / slug / "metadata.jsonl"
            _save_metadata(meta_list, meta_dest)

    return counters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dataset/raw")
    parser.add_argument("--max-per-species", type=int, default=50)
    parser.add_argument("--species-file", default="dataset/species_targets.yaml")
    parser.add_argument("--dataset", default=NATURELM_DATASET)
    parser.add_argument("--debug", action="store_true", help="Toon matches tijdens scannen")
    args = parser.parse_args()

    output_dir = Path(args.output)
    species_file = Path(args.species_file)

    if not species_file.exists():
        print(f"Bestand niet gevonden: {species_file}", file=sys.stderr)
        sys.exit(1)

    if not HAS_SOUNDFILE:
        print("⚠ soundfile niet gevonden. pip install soundfile", file=sys.stderr)
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
