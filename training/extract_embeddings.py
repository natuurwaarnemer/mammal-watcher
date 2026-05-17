"""
Bereken BirdNET embeddings voor alle WAV clips in prepared/index.csv.

Gebruik:
    python training/extract_embeddings.py \
        --data /mnt/usb/prepared/index.csv \
        --embeddings-dir /mnt/usb/embeddings

Output: één .npy bestand per clip (1024-dim float32 vector) en embeddings_index.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


TARGET_SR = 16000
EMBEDDING_DIM = 1024


# ---------------------------------------------------------------------------
# BirdNET feature extractor
# ---------------------------------------------------------------------------

def _load_birdnet_extractor():
    """Laad de BirdNET feature extractor via birdnetlib."""
    try:
        from birdnetlib import Recording
        from birdnetlib.analyzer import Analyzer

        analyzer = Analyzer()

        def extract_fn(wav_path: str) -> np.ndarray:
            recording = Recording(
                analyzer,
                wav_path,
                lat=52.0,
                lon=5.0,
                min_conf=0.0,
            )
            recording.analyze()
            if recording.embeddings is not None and len(recording.embeddings) > 0:
                emb = np.mean(np.array(recording.embeddings, dtype=np.float32), axis=0)
            else:
                emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
            return emb.reshape(EMBEDDING_DIM)

        return extract_fn, "birdnetlib"

    except ImportError as exc:
        raise RuntimeError(
            "BirdNET extractor niet beschikbaar. Installeer birdnetlib met TensorFlow Lite "
            "ondersteuning (tensorflow-cpu)."
        ) from exc


# ---------------------------------------------------------------------------
# Hoofdfunctie
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Pad naar prepared/index.csv")
    parser.add_argument(
        "--embeddings-dir",
        required=True,
        help="Uitvoermap voor .npy embedding bestanden",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    embeddings_dir = Path(args.embeddings_dir)

    if not data_path.exists():
        print(f"Indexbestand niet gevonden: {data_path}", file=sys.stderr)
        sys.exit(1)

    embeddings_dir.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kwargs):  # type: ignore[misc]
            return it

    extract_fn, backend = _load_birdnet_extractor()
    print(f"Feature extractor: {backend}")

    rows: list[dict[str, str]] = []
    with data_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    index_rows: list[dict[str, str]] = []
    skipped = 0

    for row in tqdm(rows, desc="Embeddings"):
        wav_path = Path(row["file"])
        species = row.get("species_scientific", "unknown")

        # Sla bestandsnaam op als hash van het pad
        import hashlib
        path_hash = hashlib.md5(str(wav_path).encode()).hexdigest()[:16]  # noqa: S324
        embedding_file = embeddings_dir / f"{path_hash}.npy"

        if embedding_file.exists():
            skipped += 1
            index_rows.append(
                {
                    "file": str(wav_path),
                    "species_scientific": species,
                    "embedding_file": str(embedding_file),
                }
            )
            continue

        if not wav_path.exists():
            print(f"Bestand niet gevonden, overgeslagen: {wav_path}", file=sys.stderr)
            continue

        try:
            embedding = extract_fn(str(wav_path))
        except Exception as exc:  # noqa: BLE001
            print(f"Fout bij {wav_path}: {exc}", file=sys.stderr)
            continue

        np.save(str(embedding_file), embedding.astype(np.float32))
        index_rows.append(
            {
                "file": str(wav_path),
                "species_scientific": species,
                "embedding_file": str(embedding_file),
            }
        )

    index_path = embeddings_dir / "embeddings_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "species_scientific", "embedding_file"])
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"\nKlaar! {len(index_rows)} embeddings opgeslagen ({skipped} overgeslagen).")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
