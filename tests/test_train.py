from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "training" / "train.py"
    spec = importlib.util.spec_from_file_location("train", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cnn_forward_pass_works() -> None:
    torch = pytest.importorskip("torch")
    _ = pytest.importorskip("torchaudio")
    module = _load_module()

    model = module.MammalCNN(num_classes=12)
    dummy_input = torch.randn(4, 1, 64, 157)
    output = model(dummy_input)

    assert output.shape == (4, 12)


def test_dataset_loading_works(tmp_path: Path) -> None:
    _ = pytest.importorskip("torch")
    _ = pytest.importorskip("torchaudio")
    module = _load_module()

    audio_path = tmp_path / "chunk.wav"
    waveform = np.random.default_rng(1).normal(0, 0.1, 16000 * 5).astype(np.float32)
    sf.write(str(audio_path), waveform, 16000)

    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["file", "species_scientific", "species_nl", "duration_s", "source"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "file": str(audio_path),
                "species_scientific": "Canis lupus",
                "species_nl": "wolf",
                "duration_s": 5,
                "source": "gbif",
            }
        )

    dataset = module.AudioChunkDataset(
        index_csv=index_path,
        class_to_idx={"canis_lupus": 0},
        mel_params=module.MEL_PARAMS,
    )

    assert len(dataset) == 1
    mel, label = dataset[0]
    assert label == 0
    assert mel.ndim == 3
    assert mel.shape[0] == 1
    assert mel.shape[1] == 64


def test_mel_spectrogram_has_expected_shape() -> None:
    torch = pytest.importorskip("torch")
    _ = pytest.importorskip("torchaudio")
    module = _load_module()

    transform = module.create_mel_transform(module.MEL_PARAMS)
    waveform = torch.randn(1, 16000 * 5)
    mel = transform(waveform)

    assert mel.ndim == 3
    assert mel.shape[0] == 1
    assert mel.shape[1] == 64
    assert mel.shape[2] > 0


def test_augment_waveform_preserves_shape() -> None:
    torch = pytest.importorskip("torch")
    _ = pytest.importorskip("torchaudio")
    module = _load_module()

    waveform = torch.randn(1, 16000 * 5)
    sample_rate = module.MEL_PARAMS["sample_rate"]
    augmented = module._augment_waveform(waveform, sample_rate)

    assert augmented.shape == waveform.shape


def test_augment_spectrogram_preserves_shape() -> None:
    torch = pytest.importorskip("torch")
    _ = pytest.importorskip("torchaudio")
    module = _load_module()

    mel_db = torch.randn(1, 64, 157)
    augmented = module._augment_spectrogram(mel_db)

    assert augmented.shape == mel_db.shape


def test_augmented_train_dataset_wraps_subset(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    _ = pytest.importorskip("torchaudio")
    module = _load_module()

    # Maak twee audio-bestanden aan voor twee klassen
    for name, species in [("wolf.wav", "canis_lupus"), ("vos.wav", "vulpes_vulpes")]:
        audio_path = tmp_path / name
        waveform = np.random.default_rng(0).normal(0, 0.1, 16000 * 5).astype(np.float32)
        sf.write(str(audio_path), waveform, 16000)

    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["file", "species_scientific", "species_nl", "duration_s", "source"],
        )
        writer.writeheader()
        for name, species, nl in [
            ("wolf.wav", "Canis lupus", "wolf"),
            ("vos.wav", "Vulpes vulpes", "vos"),
        ]:
            writer.writerow(
                {
                    "file": str(tmp_path / name),
                    "species_scientific": species,
                    "species_nl": nl,
                    "duration_s": 5,
                    "source": "gbif",
                }
            )

    class_to_idx = {"canis_lupus": 0, "vulpes_vulpes": 1}
    dataset = module.AudioChunkDataset(
        index_csv=index_path,
        class_to_idx=class_to_idx,
        mel_params=module.MEL_PARAMS,
    )

    from torch.utils.data import Subset

    subset = Subset(dataset, list(range(len(dataset))))
    aug_dataset = module.AugmentedTrainDataset(subset, augment=True)

    assert len(aug_dataset) == len(dataset)
    labels = aug_dataset.get_labels()
    assert len(labels) == len(dataset)
    mel, label = aug_dataset[0]
    assert mel.ndim == 3
    assert mel.shape[0] == 1
    assert mel.shape[1] == 64
    assert label in (0, 1)


def test_compute_class_weights_returns_correct_shape() -> None:
    torch = pytest.importorskip("torch")
    _ = pytest.importorskip("torchaudio")
    module = _load_module()

    labels = [0, 0, 0, 1, 1, 2]
    weights = module._compute_class_weights(labels, num_classes=3, device=torch.device("cpu"))

    assert weights.shape == (3,)
    assert (weights > 0).all()


def test_compute_class_weights_missing_class_gets_one() -> None:
    torch = pytest.importorskip("torch")
    _ = pytest.importorskip("torchaudio")
    module = _load_module()

    # Klasse 2 ontbreekt in de labels
    labels = [0, 0, 1, 1]
    weights = module._compute_class_weights(labels, num_classes=3, device=torch.device("cpu"))

    assert weights.shape == (3,)
    assert float(weights[2]) == pytest.approx(1.0)


def test_parse_args_augment_defaults_true() -> None:
    _ = pytest.importorskip("torch")
    _ = pytest.importorskip("torchaudio")
    module = _load_module()

    args = module.parse_args.__wrapped__() if hasattr(module.parse_args, "__wrapped__") else None
    # Controleer via directe aanroep met minimale argumenten
    import sys as _sys

    orig_argv = _sys.argv
    try:
        _sys.argv = ["train.py", "--data", "dummy.csv"]
        args = module.parse_args()
    finally:
        _sys.argv = orig_argv

    assert args.augment is True


def test_parse_args_no_augment_flag() -> None:
    _ = pytest.importorskip("torch")
    _ = pytest.importorskip("torchaudio")
    module = _load_module()

    import sys as _sys

    orig_argv = _sys.argv
    try:
        _sys.argv = ["train.py", "--data", "dummy.csv", "--no-augment"]
        args = module.parse_args()
    finally:
        _sys.argv = orig_argv

    assert args.augment is False


def test_dataset_respects_max_per_species(tmp_path: Path) -> None:
    _ = pytest.importorskip("torch")
    _ = pytest.importorskip("torchaudio")
    module = _load_module()

    rows = [
        ("wolf1.wav", "Canis lupus", "wolf"),
        ("wolf2.wav", "Canis lupus", "wolf"),
        ("wolf3.wav", "Canis lupus", "wolf"),
        ("vos1.wav", "Vulpes vulpes", "vos"),
    ]
    for index, (name, _, _) in enumerate(rows):
        audio_path = tmp_path / name
        waveform = np.random.default_rng(index).normal(0, 0.1, 16000).astype(np.float32)
        sf.write(str(audio_path), waveform, 16000)

    index_path = tmp_path / "index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["file", "species_scientific", "species_nl", "duration_s", "source"],
        )
        writer.writeheader()
        for name, scientific, nl_name in rows:
            writer.writerow(
                {
                    "file": str(tmp_path / name),
                    "species_scientific": scientific,
                    "species_nl": nl_name,
                    "duration_s": 1,
                    "source": "gbif",
                }
            )

    dataset = module.AudioChunkDataset(
        index_csv=index_path,
        class_to_idx={"canis_lupus": 0, "vulpes_vulpes": 1},
        mel_params=module.MEL_PARAMS,
        max_per_species=2,
        seed=123,
    )

    assert len(dataset) == 3
    assert dataset.samples_per_species == {"canis_lupus": 2, "vulpes_vulpes": 1}


def test_save_model_persists_training_info(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    _ = pytest.importorskip("torchaudio")
    module = _load_module()

    model = module.MammalCNN(num_classes=2)
    training_info = {
        "samples_per_species": {"canis_lupus": 2},
        "total_samples": 2,
        "max_per_species": 1000,
        "clip_duration_s": 10,
        "trained_at": "2026-05-14T12:00:00+00:00",
    }

    model_path = module.save_model(
        model=model,
        output_dir=tmp_path,
        class_mapping={0: "canis_lupus", 1: "vulpes_vulpes"},
        mel_params=module.MEL_PARAMS,
        val_accuracy=0.75,
        training_info=training_info,
    )

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    assert checkpoint["training_info"] == training_info
    assert checkpoint["val_accuracy"] == pytest.approx(0.75)
