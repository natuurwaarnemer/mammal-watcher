"""
Bereken BirdNET embeddings voor alle WAV clips in prepared/index.csv.

Gebruik:
    python training/extract_embeddings.py \
        --data /mnt/usb/prepared/index.csv \
        --embeddings-dir /mnt/usb/embeddings

Output: één .npy bestand per clip (6522-dim float32 vector) en embeddings_index.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import librosa
import numpy as np
import tensorflow as tf


SAMPLE_RATE = 48000
CHUNK_SAMPLES = 144000
EMBEDDING_DIM = 1024
OUTPUT_TENSOR_INDEX = 545
DEFAULT_MODEL_PATH = (
    "/usr/local/lib/python3.11/site-packages/birdnetlib/models/analyzer/"
    "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
)


def _load_interpreter(model_path: str) -> tuple[tf.lite.Interpreter, int]:
    """Laad één TFLite interpreter voor alle bestanden."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.resize_tensor_input(0, [1, 144000])
    interpreter.allocate_tensors()
    input_index = int(interpreter.get_input_details()[0]["index"])
    return interpreter, input_index


def _extract_embedding_from_audio(
    audio: np.ndarray,
    interpreter: tf.lite.Interpreter,
    input_index: int,
) -> np.ndarray:
    """Bereken gemiddelde embedding over 3s chunks."""
    if len(audio) == 0:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    embeddings: list[np.ndarray] = []
    for start in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[start : start + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        chunk = chunk.astype(np.float32, copy=False)

        interpreter.set_tensor(input_index, np.array([chunk], dtype=np.float32))
        interpreter.invoke()
        tensor = interpreter.get_tensor(OUTPUT_TENSOR_INDEX).copy()
        embeddings.append(tensor.reshape(EMBEDDING_DIM))

    if not embeddings:
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    return np.mean(np.vstack(embeddings), axis=0).astype(np.float32)


def _extract_embedding_for_file(
    wav_path: Path,
    interpreter: tf.lite.Interpreter,
    input_index: int,
) -> np.ndarray:
    audio, _ = librosa.load(
        str(wav_path),
        sr=SAMPLE_RATE,
        mono=True,
        res_type="kaiser_fast",
    )
    return _extract_embedding_from_audio(audio, interpreter, input_index)


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
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help="Pad naar BirdNET TFLite model",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Herbereken embeddings ook als .npy al bestaat (nodig na model-update)",
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

    try:
        interpreter, input_index = _load_interpreter(args.model)
    except Exception as exc:  # noqa: BLE001
        print(f"Kon TFLite model niet laden ({args.model}): {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Feature extractor: tflite ({args.model})")

    rows: list[dict[str, str]] = []
    with data_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    index_rows: list[dict[str, str]] = []
    skipped = 0

    for row in tqdm(rows, desc="Embeddings"):
        wav_path = Path(row["file"])
        species = row.get("species_scientific", "unknown")

        # Sla bestandsnaam op als hash van het pad
        path_hash = hashlib.md5(str(wav_path).encode()).hexdigest()[:16]  # noqa: S324
        embedding_file = embeddings_dir / f"{path_hash}.npy"

        if embedding_file.exists() and not args.force:
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
            embedding = _extract_embedding_for_file(wav_path, interpreter, input_index)
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
