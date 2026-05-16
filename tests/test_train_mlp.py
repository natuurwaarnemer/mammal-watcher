"""Tests voor training/train_mlp.py — MLP training op BirdNET embeddings."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "training" / "train_mlp.py"
    spec = importlib.util.spec_from_file_location("train_mlp", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mlp_forward_pass_works() -> None:
    torch = pytest.importorskip("torch")
    module = _load_module()

    model = module.MammalMLP(input_dim=1024, num_classes=15)
    dummy = torch.randn(4, 1024)
    output = model(dummy)

    assert output.shape == (4, 15)


def test_mlp_compute_class_weights_shape() -> None:
    torch = pytest.importorskip("torch")
    module = _load_module()

    labels = [0, 0, 0, 1, 1, 2]
    weights = module._compute_class_weights(labels, num_classes=3, device=torch.device("cpu"))

    assert weights.shape == (3,)
    assert (weights > 0).all()


def test_mlp_compute_class_weights_missing_class_gets_one() -> None:
    torch = pytest.importorskip("torch")
    module = _load_module()

    labels = [0, 0, 1, 1]
    weights = module._compute_class_weights(labels, num_classes=3, device=torch.device("cpu"))

    assert weights.shape == (3,)
    assert float(weights[2]) == pytest.approx(1.0)


def test_mlp_compute_sample_weights() -> None:
    _ = pytest.importorskip("torch")
    module = _load_module()

    labels = [0, 0, 1, 2, 2, 2]
    weights = module._compute_sample_weights(labels)

    assert len(weights) == 6
    assert float(weights[0]) == pytest.approx(0.5)
    assert float(weights[2]) == pytest.approx(1.0)


def test_mlp_numpy_confusion_matrix() -> None:
    _ = pytest.importorskip("torch")
    module = _load_module()

    y_true = [0, 0, 1, 1, 2]
    y_pred = [0, 1, 1, 1, 0]
    cm = module._numpy_confusion_matrix(y_true, y_pred, num_classes=3)

    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1  # class 0 correctly predicted
    assert cm[0, 1] == 1  # class 0 predicted as 1
    assert cm[1, 1] == 2  # class 1 correctly predicted
    assert cm[2, 0] == 1  # class 2 predicted as 0


def test_mlp_split_indices_proportions() -> None:
    _ = pytest.importorskip("torch")
    module = _load_module()

    train_idx, val_idx, test_idx = module._split_indices(100, seed=42)

    assert len(train_idx) == 70
    assert len(val_idx) == 15
    assert len(test_idx) == 15
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0


def test_mlp_embedding_dataset(tmp_path: Path) -> None:
    _ = pytest.importorskip("torch")
    module = _load_module()

    # Maak dummy embedding bestanden aan
    for name in ["wolf.npy", "vos.npy"]:
        emb = np.zeros(1024, dtype=np.float32)
        np.save(str(tmp_path / name), emb)

    index_path = tmp_path / "embeddings_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "species_scientific", "embedding_file"])
        writer.writeheader()
        writer.writerow(
            {
                "file": "/some/wolf.wav",
                "species_scientific": "Canis lupus",
                "embedding_file": str(tmp_path / "wolf.npy"),
            }
        )
        writer.writerow(
            {
                "file": "/some/vos.wav",
                "species_scientific": "Vulpes vulpes",
                "embedding_file": str(tmp_path / "vos.npy"),
            }
        )

    dataset = module.EmbeddingDataset(
        index_csv=index_path,
        class_to_idx={"canis_lupus": 0, "vulpes_vulpes": 1},
    )

    assert len(dataset) == 2
    emb, label = dataset[0]
    assert emb.shape == (1024,)
    assert label in (0, 1)


def test_mlp_save_model(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    module = _load_module()

    model = module.MammalMLP(input_dim=1024, num_classes=15)
    training_info = {
        "samples_per_species": {"canis_lupus": 5},
        "total_samples": 5,
        "trained_at": "2026-05-16T00:00:00+00:00",
    }
    model_path = module.save_model(
        model=model,
        output_dir=tmp_path,
        class_mapping={0: "canis_lupus"},
        val_accuracy=0.85,
        training_info=training_info,
    )

    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    assert checkpoint["model_type"] == "mlp"
    assert checkpoint["input_dim"] == 1024
    assert checkpoint["val_accuracy"] == pytest.approx(0.85)
    assert checkpoint["training_info"] == training_info
