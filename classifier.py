"""
Classifier module voor mammal-watcher.
"""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseClassifier(ABC):
    """Abstracte basisklasse voor alle zoogdier-classifiers.

    Elke concrete implementatie moet ``classify`` implementeren en een
    dict teruggeven dat voldoet aan het payload-schema van mammal-watcher.
    """

    @abstractmethod
    def classify(self, audio: np.ndarray, sr: int) -> dict:
        """Classificeer een audio-fragment.

        Parameters
        ----------
        audio:
            Mono audio-samples als numpy array (float32 of float64).
        sr:
            Sample-rate in Hz (bijv. 48000).

        Returns
        -------
        dict met de sleutels:
            species_scientific, species_nl, species_en, confidence, tier,
            model_version
        """


class MammalCNNClassifier(BaseClassifier):
    """Classifier op basis van het lokaal getrainde MammalCNN model (mammal_cnn.pt)."""

    MODEL_VERSION = "mammal-cnn-1.0"
    TARGET_SR = 16000
    CLIP_SECONDS = 10
    MEL_PARAMS = {"sample_rate": 16000, "n_mels": 64, "n_fft": 1024, "hop_length": 512}

    def __init__(
        self,
        model_path: str = "models/mammal_cnn.pt",
        min_confidence: float = 0.1,
        species_csv_path: str = "species_mammals_nl.csv",
    ) -> None:
        import torch
        import torchaudio

        self.min_confidence = float(min_confidence)
        self._torch = torch
        self._torchaudio = torchaudio
        self._species_lookup = self._load_species_lookup(species_csv_path)

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        class_mapping = checkpoint["class_mapping"]
        self._idx_to_class: dict[int, str] = {int(idx): str(slug) for idx, slug in class_mapping.items()}
        self._mel_params: dict[str, int] = {
            **self.MEL_PARAMS,
            **checkpoint.get("mel_params", self.MEL_PARAMS),
        }

        num_classes = len(self._idx_to_class)
        self._model = self._build_model(num_classes)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self._mel_params["sample_rate"],
            n_mels=self._mel_params["n_mels"],
            n_fft=self._mel_params["n_fft"],
            hop_length=self._mel_params["hop_length"],
        )
        self._to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    @staticmethod
    def _build_model(num_classes: int):
        import torch.nn as nn

        class MammalCNN(nn.Module):
            def __init__(self, n_classes: int) -> None:
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 96, kernel_size=3, padding=1),
                    nn.BatchNorm2d(96),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(p=0.4),
                    nn.Linear(96, n_classes),
                )

            def forward(self, x):
                x = self.features(x)
                return self.classifier(x)

        return MammalCNN(num_classes)

    @staticmethod
    def _slug_to_scientific(slug: str) -> str:
        scientific = slug.strip().replace("_", " ").lower()
        if not scientific:
            return "Unknown species"
        return scientific.capitalize()

    @staticmethod
    def _load_species_lookup(path: str) -> dict[str, dict[str, str]]:
        csv_path = Path(path)
        if not csv_path.exists():
            return {}

        lookup: dict[str, dict[str, str]] = {}
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = row.get("scientific_name", "").strip().lower()
                if key:
                    lookup[key] = row
        return lookup

    def _preprocess_audio(self, audio: np.ndarray, sr: int):
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0 or sr <= 0:
            return None

        peak = float(np.max(np.abs(samples)))
        if peak > 0.0:
            samples = samples / peak

        waveform = self._torch.from_numpy(samples)
        if sr != self.TARGET_SR:
            waveform = self._torchaudio.functional.resample(
                waveform,
                orig_freq=sr,
                new_freq=self.TARGET_SR,
            )

        expected_samples = self.TARGET_SR * self.CLIP_SECONDS
        if waveform.shape[0] > expected_samples:
            waveform = waveform[:expected_samples]
        elif waveform.shape[0] < expected_samples:
            waveform = self._torch.nn.functional.pad(waveform, (0, expected_samples - waveform.shape[0]))

        return waveform.unsqueeze(0)

    def _resolve_species_meta(self, slug: str) -> tuple[str, str, str, int]:
        scientific = self._slug_to_scientific(slug)
        row = self._species_lookup.get(scientific.lower(), {})
        nl_name = str(row.get("nl_name", "")).strip() or slug.replace("_", " ")
        en_name = str(row.get("en_name", "")).strip() or scientific
        try:
            tier = int(row.get("tier", 3))
        except (TypeError, ValueError):
            tier = 3
        return scientific, nl_name, en_name, tier

    def classify(self, audio: np.ndarray, sr: int) -> dict | None:
        waveform = self._preprocess_audio(audio, sr)
        if waveform is None:
            return None

        mel = self._mel_transform(waveform)
        mel_db = self._to_db(mel).to(dtype=self._torch.float32)
        model_input = mel_db.unsqueeze(0)

        with self._torch.no_grad():
            logits = self._model(model_input)
            probabilities = self._torch.softmax(logits, dim=1).squeeze(0)

        best_score, best_idx = self._torch.max(probabilities, dim=0)
        confidence = float(best_score.item())
        if confidence < self.min_confidence:
            return None

        class_idx = int(best_idx.item())
        slug = self._idx_to_class.get(class_idx, "unknown_species")
        scientific, nl_name, en_name, tier = self._resolve_species_meta(slug)
        return {
            "species_scientific": scientific,
            "species_nl": nl_name,
            "species_en": en_name,
            "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
            "tier": int(tier),
            "model_version": self.MODEL_VERSION,
        }


class BirdNetMLPClassifier(BaseClassifier):
    """Classifier op basis van BirdNET embeddings + kleine PyTorch MLP (mammal_mlp.pt)."""

    MODEL_VERSION = "birdnet-mlp-1.0"
    TARGET_SR = 16000
    EMBEDDING_DIM = 1024

    def __init__(
        self,
        model_path: str = "models/mammal_mlp.pt",
        min_confidence: float = 0.1,
        species_csv_path: str = "species_mammals_nl.csv",
    ) -> None:
        import torch

        self.min_confidence = float(min_confidence)
        self._torch = torch
        self._species_lookup = MammalCNNClassifier._load_species_lookup(species_csv_path)

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        class_mapping = checkpoint["class_mapping"]
        self._idx_to_class: dict[int, str] = {int(idx): str(slug) for idx, slug in class_mapping.items()}
        self._input_dim: int = int(checkpoint.get("input_dim", self.EMBEDDING_DIM))

        num_classes = len(self._idx_to_class)
        self._model = self._build_model(self._input_dim, num_classes)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

        self._extract_fn = self._load_extractor()

    @staticmethod
    def _build_model(input_dim: int, num_classes: int):
        import torch.nn as nn

        class MammalMLP(nn.Module):
            def __init__(self, in_dim: int, n_classes: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, n_classes),
                )

            def forward(self, x):
                return self.net(x)

        return MammalMLP(input_dim, num_classes)

    def _load_extractor(self):
        """Laad BirdNET feature extractor via birdnetlib."""
        try:
            from birdnetlib import Recording
            from birdnetlib.analyzer import Analyzer

            analyzer = Analyzer()

            def extract_birdnet(audio: np.ndarray, sr: int) -> np.ndarray:
                import os
                import tempfile

                import soundfile as sf

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                samples = audio.astype(np.float32)
                sf.write(tmp_path, samples, sr)
                try:
                    recording = Recording(analyzer, tmp_path, lat=52.0, lon=5.0, min_conf=0.0)
                    recording.analyze()
                    if recording.embeddings is not None and len(recording.embeddings) > 0:
                        emb = np.mean(np.array(recording.embeddings, dtype=np.float32), axis=0)
                    else:
                        emb = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                return emb.reshape(self.EMBEDDING_DIM).astype(np.float32)

            return extract_birdnet

        except ImportError as exc:
            raise RuntimeError(
                "BirdNetMLPClassifier vereist birdnetlib met TensorFlow Lite ondersteuning "
                "(installeer tensorflow-cpu)."
            ) from exc

    def _resolve_species_meta(self, slug: str) -> tuple[str, str, str, int]:
        scientific = MammalCNNClassifier._slug_to_scientific(slug)
        row = self._species_lookup.get(scientific.lower(), {})
        nl_name = str(row.get("nl_name", "")).strip() or slug.replace("_", " ")
        en_name = str(row.get("en_name", "")).strip() or scientific
        try:
            tier = int(row.get("tier", 3))
        except (TypeError, ValueError):
            tier = 3
        return scientific, nl_name, en_name, tier

    def classify(self, audio: np.ndarray, sr: int) -> dict | None:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0 or sr <= 0:
            return None

        peak = float(np.max(np.abs(samples)))
        if peak > 0.0:
            samples = samples / peak

        try:
            embedding = self._extract_fn(samples, sr)
        except Exception:  # noqa: BLE001
            return None

        tensor = self._torch.from_numpy(embedding).unsqueeze(0)

        with self._torch.no_grad():
            logits = self._model(tensor)
            probabilities = self._torch.softmax(logits, dim=1).squeeze(0)

        best_score, best_idx = self._torch.max(probabilities, dim=0)
        confidence = float(best_score.item())
        if confidence < self.min_confidence:
            return None

        class_idx = int(best_idx.item())
        slug = self._idx_to_class.get(class_idx, "unknown_species")
        # Background/ruis → geen detectie rapporteren
        if slug == "background":
            return None
        scientific, nl_name, en_name, tier = self._resolve_species_meta(slug)
        return {
            "species_scientific": scientific,
            "species_nl": nl_name,
            "species_en": en_name,
            "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 4),
            "tier": int(tier),
            "model_version": self.MODEL_VERSION,
        }
