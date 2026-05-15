from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_yamnet_classifier_returns_valid_dict() -> None:
    pytest.importorskip("tensorflow")
    from classifier import YAMNetClassifier

    class FakeYamnet:
        def __call__(self, audio: np.ndarray) -> tuple[np.ndarray, None, None]:
            scores = np.zeros((3, 521), dtype=np.float32)
            scores[:, 69] = 0.91
            return scores, None, None

    clf = YAMNetClassifier(model=FakeYamnet())
    audio = np.random.default_rng(1).normal(0, 0.02, 48000).astype(np.float32)
    result = clf.classify(audio, sr=48000)

    assert result is not None
    required = {
        "species_scientific",
        "species_nl",
        "species_en",
        "confidence",
        "tier",
        "model_version",
    }
    assert required.issubset(result.keys())
    assert result["model_version"] == "yamnet-1.0"


def test_yamnet_classifier_interface() -> None:
    pytest.importorskip("tensorflow")
    from classifier import BaseClassifier, YAMNetClassifier

    assert issubclass(YAMNetClassifier, BaseClassifier)


def test_yamnet_preprocess_normalizes_low_amplitude_audio() -> None:
    from classifier import YAMNetClassifier

    class FakeYamnet:
        def __call__(self, audio: np.ndarray) -> tuple[np.ndarray, None, None]:
            return np.zeros((1, 521), dtype=np.float32), None, None

    clf = YAMNetClassifier(model=FakeYamnet())
    audio = np.array([0.05, -0.025], dtype=np.float32)

    processed = clf._preprocess(audio, sr=16000)

    assert np.max(np.abs(processed)) == pytest.approx(1.0)
    assert processed[0] == pytest.approx(1.0)


def test_clip_saver_creates_files(tmp_path: Path) -> None:
    from mammal_watcher import ClipSaver

    clip_saver = ClipSaver(
        clips_dir=str(tmp_path),
        enabled=True,
        save_uncertain=True,
        uncertain_threshold=0.5,
    )
    audio = np.random.default_rng(2).normal(0, 0.01, 16000).astype(np.float32)
    payload = {
        "timestamp": "2026-05-09T15:36:14Z",
        "species_scientific": "Vulpes vulpes",
        "species_nl": "vos",
        "confidence": 0.42,
        "tier": 2,
        "model_version": "yamnet-1.0",
        "duration_s": 1.0,
        "rms": 0.01,
    }

    saved = clip_saver.save(audio, 16000, payload)
    assert saved is not None
    assert (tmp_path / "uncertain").exists()
    assert (tmp_path / "index.jsonl").exists()

    rows = (tmp_path / "index.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    data = json.loads(rows[0])
    assert data["species_scientific"] == "Vulpes vulpes"
    assert data["filename"].startswith("uncertain/")
