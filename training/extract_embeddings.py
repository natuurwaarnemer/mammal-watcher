"""
Bereken YAMNet embeddings voor alle WAV clips in de prepared directory.

Gebruik (aanbevolen — verwerkt ALLE soorten inclusief background):
    python training/extract_embeddings.py \
        --prepared-dir /mnt/usb/prepared \
        --embeddings-dir /mnt/usb/embeddings_yamnet

Gebruik (alleen clips uit een bestaande index CSV):
    python training/extract_embeddings.py \
        --data /mnt/usb/prepared/index.csv \
        --embeddings-dir /mnt/usb/embeddings_yamnet

Output: één .npy bestand per clip (1024-dim float32 YAMNet embedding) + embeddings_index.csv.

Waarom YAMNet?
  BirdNET embeddings zijn vogelspecifiek (getraind op AudioSet vogeldetectie).
  YAMNet is general audio (521 AudioSet klassen) en maakt onderscheid tussen
  zoogdiergeluiden, vogels, mens, en omgevingsgeluid — cruciaal voor een
  betrouwbare background-klasse.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SR = 16000
EMBEDDING_DIM = 1024
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"


def _load_yamnet():
    try:
        import tensorflow_hub as hub
    except ImportError as exc:
        raise RuntimeError(
            "YAMNet vereist tensorflow-hub. "
            "Installeer via: pip install tensorflow-hub"
        ) from exc
    print("YAMNet model laden (eerste keer ~30s download)...")
    model = hub.load(YAMNET_MODEL_URL)
    print("Model geladen ✓")
    return model


def _extract_embedding(model, audio: np.ndarray) -> np.ndarray:
    """Extraheer YAMNet 1024-dim embedding, gemiddeld over tijdframes."""
    import tensorflow as tf
    waveform = tf.constant(audio, dtype=tf.float32)
    _, embeddings, _ = model(waveform)
    # embeddings shape: (num_frames, 1024) — gemiddeld over frames
    return np.asarray(embeddings).mean(axis=0).astype(np.float32)


def _load_wav_16k(wav_path: Path) -> np.ndarray | None:
    """Laad WAV als mono float32 op 16kHz; resampelt indien nodig."""
    try:
        audio, sr = sf.read(str(wav_path), dtype="float32")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ Leesfout ({wav_path.name}): {exc}", file=sys.stderr)
        return None
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != TARGET_SR:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ Resample fout ({wav_path.name}, {sr}Hz): {exc}", file=sys.stderr)
            return None
    return audio


def _slug(name: str) -> str:
    return re.sub(r"_+", "_", name.strip().lower().replace(" ", "_"))


def _collect_from_prepared_dir(prepared_dir: Path) -> list[tuple[Path, str]]:
    """Scan alle soort-submappen; directory naam = species slug."""
    entries: list[tuple[Path, str]] = []
    for species_dir in sorted(prepared_dir.iterdir()):
        if not species_dir.is_dir():
            continue
        slug = species_dir.name
        wav_files = sorted(species_dir.glob("*.wav"))
        if not wav_files:
            print(f"  ⚠ Geen WAV-bestanden in {slug}, overgeslagen.")
            continue
        print(f"  {slug}: {len(wav_files)} clips")
        for wav_file in wav_files:
            entries.append((wav_file, slug))
    return entries


def _collect_from_csv(csv_path: Path) -> list[tuple[Path, str]]:
    """Lees index.csv; geef (wav_path, species_slug) tuples."""
    entries: list[tuple[Path, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            wav_path = Path(row["file"])
            species_slug = _slug(row.get("species_scientific", "unknown"))
            entries.append((wav_path, species_slug))
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--prepared-dir",
        help="Map met soort-submappen (bijv. /mnt/usb/prepared). "
             "Verwerkt ALLE submappen inclusief background.",
    )
    src.add_argument(
        "--data",
        help="Pad naar index.csv (verwerkt alleen vermelde clips).",
    )
    parser.add_argument(
        "--embeddings-dir",
        required=True,
        help="Uitvoermap voor .npy bestanden en embeddings_index.csv",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Herbereken embeddings ook als .npy al bestaat",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings_dir = Path(args.embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    if args.prepared_dir:
        prepared_dir = Path(args.prepared_dir)
        if not prepared_dir.exists():
            print(f"Map niet gevonden: {prepared_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Soorten in {prepared_dir}:")
        entries = _collect_from_prepared_dir(prepared_dir)
    else:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"Indexbestand niet gevonden: {data_path}", file=sys.stderr)
            sys.exit(1)
        entries = _collect_from_csv(data_path)

    if not entries:
        print("Geen WAV-bestanden gevonden.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{len(entries)} clips gevonden. YAMNet laden...")
    model = _load_yamnet()

    try:
        from tqdm import tqdm
        iter_entries = tqdm(entries, desc="Embeddings")
    except ImportError:
        iter_entries = entries  # type: ignore[assignment]

    index_rows: list[dict[str, str]] = []
    skipped = 0
    errors = 0

    for wav_path, species_slug in iter_entries:
        path_hash = hashlib.md5(str(wav_path).encode()).hexdigest()[:16]  # noqa: S324
        embedding_file = embeddings_dir / f"{path_hash}.npy"

        if embedding_file.exists() and not args.force:
            skipped += 1
            index_rows.append({
                "file": str(wav_path),
                "species_scientific": species_slug,
                "embedding_file": str(embedding_file),
            })
            continue

        if not wav_path.exists():
            print(f"Niet gevonden, overgeslagen: {wav_path}", file=sys.stderr)
            errors += 1
            continue

        audio = _load_wav_16k(wav_path)
        if audio is None:
            errors += 1
            continue

        try:
            embedding = _extract_embedding(model, audio)
        except Exception as exc:  # noqa: BLE001
            print(f"Extractiefout ({wav_path.name}): {exc}", file=sys.stderr)
            errors += 1
            continue

        np.save(str(embedding_file), embedding)
        index_rows.append({
            "file": str(wav_path),
            "species_scientific": species_slug,
            "embedding_file": str(embedding_file),
        })

    index_path = embeddings_dir / "embeddings_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "species_scientific", "embedding_file"])
        writer.writeheader()
        writer.writerows(index_rows)

    processed = len(index_rows) - skipped
    print(f"\n✅ Klaar: {len(index_rows)} entries → {index_path}")
    print(f"   Nieuw: {processed} | Overgeslagen: {skipped} | Fouten: {errors}")


if __name__ == "__main__":
    main()
