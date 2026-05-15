from __future__ import annotations

import numpy as np
import pytest

from classifier import MammalCNNClassifier


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
