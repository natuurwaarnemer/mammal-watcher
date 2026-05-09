"""
Converteer ruwe downloads naar genormaliseerde 16kHz mono WAV.
Gebruik: python dataset/prepare_dataset.py --input dataset/raw --output dataset/prepared
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import av

    HAS_AV = True
except ImportError:
    HAS_AV = False

try:
    from scipy.io import wavfile as scipy_wavfile
    from scipy.signal import resample_poly

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

TARGET_SR = 16000
CHUNK_SECONDS = 5
INDEX_FILENAME = "index.csv"


def _decode_audio_av(path: Path) -> tuple[np.ndarray, int]:
    """Decodeer een audiobestand via PyAV naar float32 numpy-array."""
    container = av.open(str(path))
    audio_stream = next(s for s in container.streams if s.type == "audio")
    sr = audio_stream.sample_rate
    frames: list[np.ndarray] = []
    for packet in container.demux(audio_stream):
        for frame in packet.decode():
            arr = frame.to_ndarray()  # shape: (channels, samples)
            frames.append(arr)
    container.close()
    if not frames:
        raise ValueError(f"Geen audio frames in {path}")
    audio = np.concatenate(frames, axis=-1)  # (channels, total_samples)
    # Naar mono
    if audio.ndim > 1:
        audio = audio.mean(axis=0)
    # Naar float32 in [-1, 1]
    audio = audio.astype(np.float32)
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        audio /= max_val
    return audio, sr


def _resample(audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio van src_sr naar target_sr."""
    if src_sr == target_sr:
        return audio
    if HAS_SCIPY:
        from math import gcd

        g = gcd(src_sr, target_sr)
        audio = resample_poly(audio, target_sr // g, src_sr // g).astype(np.float32)
    else:
        # Eenvoudige lineaire interpolatie als fallback
        duration = len(audio) / src_sr
        new_length = int(duration * target_sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, new_length),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return audio


def _chunk_audio(audio: np.ndarray, sr: int, chunk_s: int) -> list[np.ndarray]:
    """Splits audio in gelijke chunks van chunk_s seconden."""
    chunk_len = sr * chunk_s
    chunks = []
    for start in range(0, len(audio) - chunk_len + 1, chunk_len):
        chunks.append(audio[start : start + chunk_len])
    return chunks


def _species_meta_from_dir(species_dir: Path) -> dict:
    """Haal soortmeta op uit de mapnaam (slug = scientific_name.lower().replace(' ', '_'))."""
    return {"slug": species_dir.name}


def process_file(
    mp3_path: Path,
    output_dir: Path,
    species_slug: str,
    species_scientific: str,
    species_nl: str,
    source: str,
) -> list[dict]:
    """Verwerk één MP3 en sla WAV-chunks op. Geeft indexrijen terug."""
    if not HAS_AV:
        print(f"  ⚠ PyAV niet beschikbaar, sla over: {mp3_path.name}", file=sys.stderr)
        return []

    try:
        audio, sr = _decode_audio_av(mp3_path)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ Decodering mislukt ({mp3_path.name}): {exc}", file=sys.stderr)
        return []

    audio_16k = _resample(audio, sr, TARGET_SR)
    chunks = _chunk_audio(audio_16k, TARGET_SR, CHUNK_SECONDS)
    if not chunks:
        return []

    out_species_dir = output_dir / species_slug
    out_species_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    stem = mp3_path.stem
    for i, chunk in enumerate(chunks):
        chunk_name = f"{stem}_chunk{i:04d}.wav"
        chunk_path = out_species_dir / chunk_name
        if not chunk_path.exists():
            sf.write(str(chunk_path), chunk, TARGET_SR, subtype="FLOAT")
        rows.append(
            {
                "file": str(chunk_path),
                "species_scientific": species_scientific,
                "species_nl": species_nl,
                "duration_s": CHUNK_SECONDS,
                "source": source,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="dataset/raw",
        help="Map met ruwe MP3-downloads (standaard: dataset/raw)",
    )
    parser.add_argument(
        "--output",
        default="dataset/prepared",
        help="Uitvoermap voor genormaliseerde WAV-chunks (standaard: dataset/prepared)",
    )
    parser.add_argument(
        "--species-file",
        default="dataset/species_targets.yaml",
        help="Pad naar species_targets.yaml",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    species_file = Path(args.species_file)

    if not input_dir.exists():
        print(f"Invoermap niet gevonden: {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Laad soortmetadata als lookup op slug
    import yaml  # noqa: PLC0415

    species_map: dict[str, dict] = {}
    if species_file.exists():
        import re  # noqa: PLC0415

        with species_file.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        for sp in data.get("species", []):
            slug = re.sub(r"\s+", "_", sp["scientific"].strip().lower())
            species_map[slug] = sp

    output_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict] = []

    species_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    iterator = tqdm(species_dirs, desc="Soorten") if HAS_TQDM else species_dirs

    for species_dir in iterator:
        slug = species_dir.name
        meta = species_map.get(slug, {})
        scientific = meta.get("scientific", slug.replace("_", " ").title())
        nl_name = meta.get("nl", slug)

        mp3_files = sorted(species_dir.glob("*.mp3"))
        if not mp3_files:
            continue

        file_iter = tqdm(mp3_files, desc=slug, leave=False) if HAS_TQDM else mp3_files
        for mp3_path in file_iter:
            rows = process_file(
                mp3_path,
                output_dir,
                slug,
                scientific,
                nl_name,
                source="xeno-canto",
            )
            index_rows.extend(rows)

    # Schrijf index.csv
    index_path = output_dir / INDEX_FILENAME
    fieldnames = ["file", "species_scientific", "species_nl", "duration_s", "source"]
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"\n✅ {len(index_rows)} chunks verwerkt → {index_path}")


if __name__ == "__main__":
    main()
