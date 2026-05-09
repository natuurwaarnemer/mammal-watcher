"""
Train SVM/MLP classifier op YAMNet embeddings.
Gebruik: python dataset/train_classifier.py --features dataset/features --output models/
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

MIN_SAMPLES_PER_CLASS = 5


def _load_features(features_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """Laad alle per-soort .npy bestanden en bouw X, y matrices."""
    species_map_path = features_dir / "species_map.json"
    labels_path = features_dir / "labels.npy"

    if not species_map_path.exists():
        raise FileNotFoundError(f"species_map.json niet gevonden in {features_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.npy niet gevonden in {features_dir}")

    with species_map_path.open(encoding="utf-8") as fh:
        raw_map = json.load(fh)
    species_map = {int(k): v for k, v in raw_map.items()}

    labels = np.load(str(labels_path))

    # Bouw X uit per-soort .npy bestanden
    all_X: list[np.ndarray] = []
    all_y: list[int] = []
    for label_idx, slug in sorted(species_map.items()):
        npy_path = features_dir / f"{slug}.npy"
        if not npy_path.exists():
            print(f"  ⚠ {npy_path.name} niet gevonden, overgeslagen.")
            continue
        features = np.load(str(npy_path))
        n = features.shape[0]
        if n < MIN_SAMPLES_PER_CLASS:
            print(
                f"  ⚠ {slug}: slechts {n} samples (minimum {MIN_SAMPLES_PER_CLASS}), "
                "model kan minder goed presteren."
            )
        all_X.append(features)
        all_y.extend([label_idx] * n)

    if not all_X:
        raise ValueError("Geen feature-bestanden gevonden. Voer eerst extract_features.py uit.")

    X = np.concatenate(all_X, axis=0)
    y = np.array(all_y, dtype=np.int32)
    return X, y, species_map


def _train_svm(X_train: np.ndarray, y_train: np.ndarray):
    """Train een SVM met RBF kernel en gestandaardiseerde features."""
    from sklearn.pipeline import Pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
    from sklearn.svm import SVC  # noqa: PLC0415

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, C=10.0, gamma="scale")),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def _train_mlp(X_train: np.ndarray, y_train: np.ndarray):
    """Train een MLP classifier als alternatief voor SVM."""
    from sklearn.neural_network import MLPClassifier  # noqa: PLC0415
    from sklearn.pipeline import Pipeline  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        default="dataset/features",
        help="Map met YAMNet feature-arrays (standaard: dataset/features)",
    )
    parser.add_argument(
        "--output",
        default="models/",
        help="Uitvoermap voor getraind model (standaard: models/)",
    )
    parser.add_argument(
        "--model",
        choices=["svm", "mlp"],
        default="svm",
        help="Type classifier: svm (standaard) of mlp",
    )
    args = parser.parse_args()

    features_dir = Path(args.features)
    output_dir = Path(args.output)

    if not features_dir.exists():
        print(f"Features-map niet gevonden: {features_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        from sklearn.metrics import classification_report  # noqa: PLC0415
        from sklearn.model_selection import train_test_split  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn is vereist. Installeer via: pip install scikit-learn"
        ) from exc

    print("Features laden...")
    X, y, species_map = _load_features(features_dir)
    print(f"  X: {X.shape}, y: {y.shape}, {len(species_map)} soort(en)\n")

    # Stratified train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Training: {len(X_train)} samples, test: {len(X_test)} samples\n")

    print(f"Classifier trainen ({args.model.upper()})...")
    if args.model == "svm":
        clf = _train_svm(X_train, y_train)
    else:
        clf = _train_mlp(X_train, y_train)
    print("Training klaar ✓\n")

    # Evaluatie
    y_pred = clf.predict(X_test)
    target_names = [species_map[i] for i in sorted(species_map.keys())]
    print("=== Classification Report ===")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            zero_division=0,
        )
    )

    # Opslaan
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "mammal_classifier_v1.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(clf, fh)

    species_map_out = output_dir / "species_map.json"
    with species_map_out.open("w", encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in species_map.items()}, fh, indent=2, ensure_ascii=False)

    print(f"\n✅ Model opgeslagen: {model_path}")
    print(f"   Species map: {species_map_out}")


if __name__ == "__main__":
    main()
