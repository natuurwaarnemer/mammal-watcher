"""
Download zoogdiergeluiden via NatureLM dataset (Earth Species Project) van Hugging Face.

Strategie:
  1. DuckDB query op HuggingFace Parquet via datasets-server API → index in seconden
  2. Download alleen audio voor matching IDs via HuggingFace datasets streaming
  3. Sla op als WAV via soundfile

Gebruik: python dataset/download_naturelm.py --output dataset/raw --species-file dataset/species_targets.yaml
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import urllib.request
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
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

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

# HuggingFace Datasets Server API — geeft echte Parquet CDN URLs terug
HF_DATASETS_SERVER = (
    "https://datasets-server.huggingface.co/parquet"
    "?dataset=EarthSpeciesProject%2FNatureLM-audio-training"
)

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
    if not output:
        return ""
    words = output.strip().split()
    if len(words) >= 2:
        return " ".join(words[-2:])
    return output.strip()


def _get_parquet_urls() -> list[str]:
    """
    Haal Parquet CDN URLs op via HuggingFace Datasets Server API.
    Retourneert lijst van HTTPS URLs die DuckDB direct kan lezen.
    """
    try:
        req = urllib.request.Request(
            HF_DATASETS_SERVER,
            headers={"User-Agent": "mammal-watcher/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())

        # Datasets Server geeft {"parquet_files": [{url, filename, split, ...}, ...]}
        urls = []
        parquet_files = data.get("parquet_files", [])
        for f in parquet_files:
            url = f.get("url")
            if url:
                urls.append(url)

        return urls

    except Exception as exc:
        print(f"  ⚠ Parquet URLs ophalen mislukt: {exc}", file=sys.stderr)
        return []


def _build_index_duckdb(species_list: list[dict], max_per_species: int, debug: bool) -> dict[str, list[str]]:
    """
    Stap 1: DuckDB query op HuggingFace Parquet CDN URLs.
    Razendsnel — geen audio, alleen tekst kolommen.
    """
    print("🦆 Stap 1: DuckDB index bouwen via HuggingFace Parquet...")

    print("  Parquet URLs ophalen via datasets-server...", end=" ", flush=True)
    urls = _get_parquet_urls()
    if not urls:
        print("mislukt!")
        return {}
    print(f"{len(urls)} bestanden gevonden ✓")
    if debug:
        for u in urls[:2]:
            print(f"    {u}")

    url_list = ", ".join(f"'{u}'" for u in urls)

    con = duckdb.connect()
    try:
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("SET enable_progress_bar = false;")
    except Exception:
        pass

    index: dict[str, list[str]] = {}

    for species_info in species_list:
        scientific = species_info["scientific"]
        slug = _slug(scientific)
        nl = species_info["nl"]
        parts = scientific.split(" ", 1)
        genus = parts[0]
        soort = parts[1] if len(parts) > 1 else ""
        like_pattern = f"%{genus} {soort}%"

        print(f"  Zoeken: {nl} ({scientific})...", end=" ", flush=True)
        try:
            result = con.execute(f"""
                SELECT id
                FROM read_parquet([{url_list}])
                WHERE task = 'taxonomic-classification'
                  AND output LIKE '{like_pattern}'
                LIMIT {max_per_species}
            """).fetchall()

            ids = [row[0] for row in result if row[0]]
            index[slug] = ids
            print(f"{len(ids)} gevonden ✓")

        except Exception as exc:
            print(f"mislukt ({exc})")
            index[slug] = []

    con.close()
    return index


def _build_index_stream(species_list: list[dict], max_per_species: int, debug: bool) -> dict[str, list[str]]:
    """Fallback: stream dataset zonder audio decoding."""
    print("📋 Stap 1: Index bouwen via streaming (fallback)...")
    target_names = _scientific_names(species_list)

    try:
        from datasets import Audio
        ds_meta = load_dataset(NATURELM_DATASET, split="train", streaming=True)
        ds_meta = ds_meta.cast_column("audio", Audio(decode=False))
    except Exception as exc:
        print(f"⚠ Dataset laden mislukt: {exc}", file=sys.stderr)
        return {}

    index: dict[str, list[str]] = {_slug(s["scientific"]): [] for s in species_list}
    iterator = tqdm(ds_meta, desc="Index scannen", unit=" samples") if HAS_TQDM else ds_meta

    for sample in iterator:
        if "taxonomic" not in sample.get("task", ""):
            continue
        species_name = _extract_species_from_output(sample.get("output", "")).lower()
        if species_name in target_names:
            slug = _slug(species_name)
            sample_id = sample.get("id", "")
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

    # === STAP 1: Index ===
    if HAS_DUCKDB:
        index = _build_index_duckdb(species_list, max_per_species, debug)
        if not index or all(len(v) == 0 for v in index.values()):
            print("  DuckDB leverde geen resultaten → streaming fallback...")
            index = _build_index_stream(species_list, max_per_species, debug)
    else:
        print("⚠ pip install duckdb voor snellere index.")
        index = _build_index_stream(species_list, max_per_species, debug)

    total_matches = sum(len(v) for v in index.values())
    print(f"\n  Totaal: {total_matches} opnames gevonden")
    for species_info in species_list:
        slug = _slug(species_info["scientific"])
        n = len(index.get(slug, []))
        print(f"  {'✓' if n > 0 else '✗'} {species_info['nl']:20s}: {n} opnames")

    if total_matches == 0:
        print("\n⚠ Geen matches. Controleer species_targets.yaml.", file=sys.stderr)
        return counters

    wanted_ids: dict[str, str] = {}
    for slug, ids in index.items():
        for sample_id in ids[:max_per_species]:
            wanted_ids[sample_id] = slug

    print(f"\n📡 Stap 2: Audio downloaden voor {len(wanted_ids)} opnames...")

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
    print(f"Uitvoer: {output_dir.resolve()}")
    print(f"DuckDB: {'✓' if HAS_DUCKDB else '✗ pip install duckdb'}\n")

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
