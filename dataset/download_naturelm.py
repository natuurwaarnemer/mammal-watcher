"""
Download zoogdiergeluiden via NatureLM dataset (Earth Species Project) van Hugging Face.

Strategie:
  1. Laad alleen Parquet metadata kolommen (id, task, output) — geen audio bytes → minuten i.p.v. uren
  2. Filter in-memory op gewenste soorten en sla index op als checkpoint JSON
  3. Download alleen audio voor matching IDs via streaming pass
  4. Sla op als WAV via soundfile

Gebruik (soorten downloaden naar prepared dir):
    python dataset/download_naturelm.py \
        --output /mnt/usb/prepared \
        --species-file dataset/species_targets.yaml \
        --max-per-species 500

Gebruik (background downloaden via NatureLM WavCaps/UrbanSound):
    python dataset/download_naturelm.py \
        --output /mnt/usb/prepared \
        --skip-species \
        --background-clips 2000

Gebruik (alles tegelijk — soorten + background):
    python dataset/download_naturelm.py \
        --output /mnt/usb/prepared \
        --max-per-species 500 \
        --background-clips 2000
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

NATURELM_DATASET = "EarthSpeciesProject/NatureLM-audio-training"
SAMPLE_RATE = 16000
BACKGROUND_SOURCES = {"WavCaps", "UrbanSound", "UrbanSound8K", "AudioCaps"}


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


def _build_index(
    species_list: list[dict],
    max_per_species: int,
    debug: bool,
    checkpoint_path: Path,
    force_reindex: bool = False,
) -> dict[str, list[str]]:
    """
    Stap 1: Laad metadata via Parquet (geen audio), filter op soorten.
    Bouwt index {slug: [id, ...]} van alle matching samples.
    Slaat de index op als checkpoint JSON voor herstart.
    """
    # Laad checkpoint als het bestaat en --force-reindex niet gezet is
    if not force_reindex and checkpoint_path.exists():
        print(f"📂 Checkpoint gevonden: {checkpoint_path}")
        try:
            with checkpoint_path.open(encoding="utf-8") as fh:
                data: dict[str, list[str]] = json.load(fh)
            total = sum(len(v) for v in data.values())
            print(f"  ✓ Index geladen uit checkpoint: {total} opnames")
            return data
        except Exception as exc:
            print(f"  ⚠ Checkpoint laden mislukt ({exc}), herbouw index...", file=sys.stderr)

    print("📋 Stap 1: Metadata streamen (vroege exit zodra alle soorten vol zijn)...")
    target_names = _scientific_names(species_list)

    try:
        ds_meta = load_dataset(
            NATURELM_DATASET,
            split="train",
            streaming=True,
        )
    except Exception as exc:
        print(f"⚠ Dataset laden mislukt: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  Filter op {len(species_list)} soorten (max {max_per_species} per soort)...")
    index: dict[str, list[str]] = {_slug(s["scientific"]): [] for s in species_list}
    found_total = 0
    scanned = 0

    def _all_done() -> bool:
        return all(len(index[_slug(s["scientific"])]) >= max_per_species for s in species_list)

    iterator = tqdm(ds_meta, desc="Streaming", unit=" samples") if HAS_TQDM else ds_meta

    for sample in iterator:
        scanned += 1
        if _all_done():
            break

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
            found_total += 1
            if debug:
                print(f"  ✓ {species_name} → {sample_id}")

            # Tussentijds checkpoint elke 50 matches
            if found_total % 50 == 0:
                try:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    with checkpoint_path.open("w", encoding="utf-8") as fh:
                        json.dump(index, fh, ensure_ascii=False)
                except Exception:
                    pass

    print(f"  Gescand: {scanned:,} samples, gevonden: {found_total} matches")

    for species_info in species_list:
        slug = _slug(species_info["scientific"])
        n = len(index.get(slug, []))
        print(f"  {'✓' if n > 0 else '✗'} {species_info['scientific']}: {n} matches")

    # Sla index op als checkpoint
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_path.open("w", encoding="utf-8") as fh:
            json.dump(index, fh, ensure_ascii=False, indent=2)
        print(f"  Index opgeslagen: {checkpoint_path}")
    except Exception as exc:
        print(f"  ⚠ Checkpoint opslaan mislukt: {exc}", file=sys.stderr)

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
    checkpoint_path: Path | None = None,
    force_reindex: bool = False,
) -> dict[str, int]:
    if not HAS_DATASETS:
        print("⚠ pip install datasets", file=sys.stderr)
        sys.exit(1)

    if checkpoint_path is None:
        checkpoint_path = output_dir.parent / "index_checkpoint.json"

    species_by_name = {s["scientific"].lower(): s for s in species_list}
    counters: dict[str, int] = {_slug(s["scientific"]): 0 for s in species_list}
    metadata_buffers: dict[str, list[dict]] = {_slug(s["scientific"]): [] for s in species_list}

    # === STAP 1: Index bouwen via Parquet metadata (geen audio) ===
    index = _build_index(species_list, max_per_species, debug, checkpoint_path, force_reindex)

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



def _download_background(
    output_dir: Path,
    sources: set[str],
    max_clips: int,
    debug: bool = False,
) -> int:
    """Stream WavCaps/UrbanSound/AudioCaps samples als background-klasse."""
    bg_dir = output_dir / "background"
    bg_dir.mkdir(parents=True, exist_ok=True)

    existing = {f.stem for f in bg_dir.glob("*.wav")}
    remaining = max_clips - len(existing)
    if remaining <= 0:
        print(f"  Background al volledig ({len(existing)} clips aanwezig)")
        return len(existing)

    print(f"\n Achtergrond downloaden ({', '.join(sorted(sources))})...")
    print(f"  Doel: {max_clips} clips, al aanwezig: {len(existing)}")

    try:
        ds = load_dataset(NATURELM_DATASET, split="train", streaming=True)
    except Exception as exc:
        print(f"Dataset laden mislukt: {exc}", file=sys.stderr)
        return len(existing)

    ok = 0
    iterator = tqdm(ds, desc="Background", unit=" samples") if HAS_TQDM else ds

    for sample in iterator:
        if ok >= remaining:
            break

        source = sample.get("source_dataset", "")
        if not any(s.lower() in source.lower() for s in sources):
            continue

        sample_id = (sample.get("id") or "").replace("/", "_")
        if sample_id in existing:
            continue

        dest = bg_dir / f"bg_{sample_id}.wav"
        if dest.exists():
            existing.add(sample_id)
            continue

        audio = sample.get("audio")
        if not audio:
            continue

        if _save_audio_as_wav(audio, dest):
            existing.add(sample_id)
            ok += 1
            if debug:
                print(f"  [background/{source}] {dest.name}")

    total = len(existing)
    print(f"  Klaar: {ok} nieuw, {total} totaal background clips")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/mnt/usb/prepared")
    parser.add_argument("--max-per-species", type=int, default=50)
    parser.add_argument("--species-file", default="dataset/species_targets.yaml")
    parser.add_argument("--dataset", default=NATURELM_DATASET)
    parser.add_argument("--background-clips", type=int, default=0,
                        help="Aantal background clips via WavCaps/UrbanSound/AudioCaps (0 = uit)")
    parser.add_argument("--background-sources", default="WavCaps,UrbanSound,AudioCaps",
                        help="Kommagescheiden bronnen voor background (standaard: WavCaps,UrbanSound,AudioCaps)")
    parser.add_argument("--skip-species", action="store_true",
                        help="Sla soort-download over, doe alleen background")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Negeer bestaand checkpoint en herbouw de index",
    )
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

    counters = download_from_naturelm(
        species_list,
        output_dir,
        args.max_per_species,
        debug=args.debug,
        force_reindex=args.force_reindex,
    )

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
