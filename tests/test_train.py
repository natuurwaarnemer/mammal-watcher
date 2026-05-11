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
