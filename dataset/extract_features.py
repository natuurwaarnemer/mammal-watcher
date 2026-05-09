"""
Extraheer YAMNet embeddings voor training van eigen classifier.
Gebruik: python dataset/extract_features.py --input dataset/prepared --output dataset/features
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

TARGET_SR = 16000
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"


def _load_yamnet():
    """Laad het YAMNet model via TensorFlow Hub."""
    try:
        import tensorflow_hub as hub
    except ImportError as exc:
        raise RuntimeError(
            "YAMNet vereist tensorflow-cpu en tensorflow-hub. "
            "Installeer via: pip install tensorflow-cpu tensorflow-hub"
        ) from exc
    return hub.load(YAMNET_MODEL_URL)


def _extract_embedding(model, audio: np.ndarray) -> np.ndarray | None:
    """Extraheer de YAMNet 512-dim embedding (gemiddeld over tijd) voor één chunk."""
    import tensorflow as tf  # noqa: PLC0415

    waveform = tf.constant(audio, dtype=tf.float32)
    _, embeddings, _ = model(waveform)
    # embeddings shape: (num_frames, 512); gemiddel over frames
    embedding = np.asarray(embeddings).mean(axis=0)  # (512,)
    return embedding.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="dataset/prepared",
        help="Map met genormaliseerde WAV-chunks (standaard: dataset/prepared)",
    )
    parser.add_argument(
        "--output",
        default="dataset/features",
        help="Uitvoermap voor feature-arrays (standaard: dataset/features)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Invoermap niet gevonden: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print("YAMNet model laden...")
    model = _load_yamnet()
    print("Model geladen ✓\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    species_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    species_map: dict[int, str] = {}
    all_labels: list[int] = []
    all_embeddings: list[np.ndarray] = []

    for label_idx, species_dir in enumerate(species_dirs):
        slug = species_dir.name
        species_map[label_idx] = slug
        wav_files = sorted(species_dir.glob("*.wav"))
        if not wav_files:
            print(f"  ⚠ Geen WAV-bestanden in {slug}, overgeslagen.")
            continue

        print(f"[{label_idx}] {slug}: {len(wav_files)} bestanden")
        embeddings: list[np.ndarray] = []
        iterator = tqdm(wav_files, desc=slug) if HAS_TQDM else wav_files

        for wav_path in iterator:
            try:
                audio, sr = sf.read(str(wav_path), dtype="float32")
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠ Leesfout ({wav_path.name}): {exc}", file=sys.stderr)
                continue
            if audio.ndim > 1:
                audio = audio.mean(axis=-1)
            if sr != TARGET_SR:
                # Bestanden zijn al 16kHz vanuit prepare_dataset.py
                print(f"  ⚠ Onverwachte sample rate {sr} in {wav_path.name}", file=sys.stderr)
                continue
            emb = _extract_embedding(model, audio)
            if emb is not None:
                embeddings.append(emb)
                all_labels.append(label_idx)

        if embeddings:
            species_features = np.stack(embeddings, axis=0)
            out_path = output_dir / f"{slug}.npy"
            np.save(str(out_path), species_features)
            all_embeddings.extend(embeddings)
            print(f"  → {out_path} ({species_features.shape})")

    # Sla gecombineerde labels + species_map op
    labels_path = output_dir / "labels.npy"
    np.save(str(labels_path), np.array(all_labels, dtype=np.int32))

    species_map_path = output_dir / "species_map.json"
    with species_map_path.open("w", encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in species_map.items()}, fh, indent=2, ensure_ascii=False)

    print(f"\n✅ {len(all_labels)} embeddings opgeslagen in {output_dir}")
    print(f"   labels.npy: {labels_path}")
    print(f"   species_map.json: {species_map_path}")


if __name__ == "__main__":
    main()
