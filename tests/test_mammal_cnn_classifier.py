from __future__ import annotations

import builtins
import numpy as np
import pytest

from classifier import BirdNetMLPClassifier, MammalCNNClassifier


def test_slug_to_scientific_conversion() -> None:
    assert MammalCNNClassifier._slug_to_scientific("vulpes_vulpes") == "Vulpes vulpes"


def test_species_meta_falls_back_when_csv_missing() -> None:
    clf = MammalCNNClassifier.__new__(MammalCNNClassifier)
    clf._species_lookup = {}
    scientific, nl_name, en_name, tier = clf._resolve_species_meta("meles_meles")
    assert scientific == "Meles meles"
    assert nl_name == "meles meles"
    assert en_name == "Meles meles"
    assert tier == 3


def test_preprocess_audio_normalizes_low_amplitude_audio() -> None:
    torch = pytest.importorskip("torch")

    clf = MammalCNNClassifier.__new__(MammalCNNClassifier)
    clf._torch = torch
    clf._torchaudio = None

    audio = np.array([0.05, -0.025], dtype=np.float32)
    waveform = clf._preprocess_audio(audio, sr=16000)

    assert waveform is not None
    assert float(torch.max(torch.abs(waveform)).item()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# BirdNetMLPClassifier tests (geen model nodig — test klasse-structuur)
# ---------------------------------------------------------------------------

def test_birdnet_mlp_build_model_returns_correct_output_shape() -> None:
    torch = pytest.importorskip("torch")

    model = BirdNetMLPClassifier._build_model(input_dim=1024, num_classes=15)
    dummy = torch.randn(4, 1024)
    output = model(dummy)
    assert output.shape == (4, 15)


def test_birdnet_mlp_build_model_small_input() -> None:
    torch = pytest.importorskip("torch")

    model = BirdNetMLPClassifier._build_model(input_dim=208, num_classes=5)
    dummy = torch.randn(2, 208)
    output = model(dummy)
    assert output.shape == (2, 5)


def test_birdnet_mlp_classify_returns_none_on_empty_audio() -> None:
    torch = pytest.importorskip("torch")

    clf = BirdNetMLPClassifier.__new__(BirdNetMLPClassifier)
    clf._torch = torch
    clf.min_confidence = 0.1
    clf.EMBEDDING_DIM = 1024
    clf.TARGET_SR = 16000

    result = clf.classify(np.array([], dtype=np.float32), sr=16000)
    assert result is None


def test_birdnet_mlp_species_meta_falls_back_when_csv_missing() -> None:
    clf = BirdNetMLPClassifier.__new__(BirdNetMLPClassifier)
    clf._species_lookup = {}
    scientific, nl_name, en_name, tier = clf._resolve_species_meta("lutra_lutra")
    assert scientific == "Lutra lutra"
    assert tier == 3


def test_birdnet_mlp_classify_with_mock_model() -> None:
    torch = pytest.importorskip("torch")

    class _FakeModel(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(x.shape[0], 3)
            out[:, 1] = 10.0  # klasse 1 wint altijd
            return out

    clf = BirdNetMLPClassifier.__new__(BirdNetMLPClassifier)
    clf._torch = torch
    clf.min_confidence = 0.1
    clf.EMBEDDING_DIM = 8
    clf.TARGET_SR = 16000
    clf._idx_to_class = {0: "vulpes_vulpes", 1: "canis_lupus", 2: "meles_meles"}
    clf._species_lookup = {}
    clf._model = _FakeModel()
    clf._model.eval()

    def _fake_extract(audio: np.ndarray, sr: int) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)

    clf._extract_fn = _fake_extract

    audio = np.random.default_rng(0).random(16000).astype(np.float32) * 0.5
    result = clf.classify(audio, sr=16000)

    assert result is not None
    assert result["species_scientific"] == "Canis lupus"
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["model_version"] == BirdNetMLPClassifier.MODEL_VERSION


def test_birdnet_mlp_classify_returns_none_for_background_prediction() -> None:
    torch = pytest.importorskip("torch")

    class _FakeModel(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(x.shape[0], 2)
            out[:, 1] = 10.0
            return out

    clf = BirdNetMLPClassifier.__new__(BirdNetMLPClassifier)
    clf._torch = torch
    clf.min_confidence = 0.1
    clf.EMBEDDING_DIM = 8
    clf.TARGET_SR = 16000
    clf._idx_to_class = {0: "canis_lupus", 1: "background"}
    clf._species_lookup = {}
    clf._model = _FakeModel()
    clf._model.eval()

    def _fake_extract(audio: np.ndarray, sr: int) -> np.ndarray:
        return np.zeros(8, dtype=np.float32)

    clf._extract_fn = _fake_extract

    audio = np.random.default_rng(1).random(16000).astype(np.float32) * 0.5
    result = clf.classify(audio, sr=16000)
    assert result is None


def test_birdnet_mlp_load_extractor_hard_fails_without_birdnetlib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "birdnetlib" or name.startswith("birdnetlib."):
            raise ImportError("birdnetlib ontbreekt")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    clf = BirdNetMLPClassifier.__new__(BirdNetMLPClassifier)
    with pytest.raises(RuntimeError, match="tensorflow-cpu"):
        clf._load_extractor()
