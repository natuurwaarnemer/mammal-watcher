from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "training" / "extract_embeddings.py"
    spec = importlib.util.spec_from_file_location("extract_embeddings", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeYAMNetModel:
    """Minimale YAMNet-mock: geeft vaste 1024-dim embeddings terug."""

    def __init__(self, fill_value: float = 0.5, n_frames: int = 3) -> None:
        self.fill_value = fill_value
        self.n_frames = n_frames

    def __call__(self, waveform):
        scores = np.zeros((self.n_frames, 521), dtype=np.float32)
        embeddings = np.full((self.n_frames, 1024), self.fill_value, dtype=np.float32)
        spectrogram = np.zeros((96, 64), dtype=np.float32)
        return scores, embeddings, spectrogram


def test_embedding_dim_constant() -> None:
    module = _load_module()
    assert module.EMBEDDING_DIM == 1024


def test_extract_embedding_returns_correct_shape() -> None:
    module = _load_module()
    audio = np.zeros(16000, dtype=np.float32)

    embedding = module._extract_embedding(_FakeYAMNetModel(), audio)

    assert embedding.shape == (module.EMBEDDING_DIM,)
    assert embedding.dtype == np.float32


def test_extract_embedding_averages_over_frames() -> None:
    module = _load_module()
    audio = np.zeros(16000, dtype=np.float32)

    embedding = module._extract_embedding(_FakeYAMNetModel(fill_value=0.75), audio)

    # Mock geeft 0.75 voor alle frames; gemiddelde moet ook 0.75 zijn
    assert np.allclose(embedding, 0.75)


def test_collect_from_prepared_dir(tmp_path: Path) -> None:
    module = _load_module()

    (tmp_path / "vulpes_vulpes").mkdir()
    (tmp_path / "vulpes_vulpes" / "clip1.wav").touch()
    (tmp_path / "background").mkdir()
    (tmp_path / "background" / "bg1.wav").touch()
    (tmp_path / "background" / "bg2.wav").touch()
    # Submap zonder WAV-bestanden mag niet crashen
    (tmp_path / "leeg").mkdir()

    entries = module._collect_from_prepared_dir(tmp_path)

    species = [slug for _, slug in entries]
    assert species.count("vulpes_vulpes") == 1
    assert species.count("background") == 2
    assert "leeg" not in species


def test_collect_from_csv(tmp_path: Path) -> None:
    module = _load_module()

    csv_path = tmp_path / "index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "species_scientific"])
        writer.writeheader()
        writer.writerow({"file": "/data/vulpes_vulpes/clip.wav", "species_scientific": "Vulpes vulpes"})
        writer.writerow({"file": "/data/sus_scrofa/clip.wav", "species_scientific": "Sus scrofa"})

    entries = module._collect_from_csv(csv_path)

    assert len(entries) == 2
    assert entries[0][1] == "vulpes_vulpes"
    assert entries[1][1] == "sus_scrofa"
