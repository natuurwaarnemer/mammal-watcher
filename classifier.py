"""
Classifier module voor mammal-watcher.

Bevat de abstracte basisklasse en een stub-implementatie die wordt gebruikt
totdat het echte ML-model beschikbaar is (PR #4).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from abc import ABC, abstractmethod
from typing import Any

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


class StubClassifier(BaseClassifier):
    """STUB — vervangen in PR #4 door echt model (YAMNet of fine-tuned variant).

    Geeft een deterministische mock-voorspelling terug op basis van de hash
    van de audio-bytes. Zelfde input → zelfde output altijd.

    Circa 10% van de inputs (hash % 10 == 0) retourneert een tier-1
    rewilding-soort zodat er tijdens het testen Telegram-alerts komen.
    """

    MODEL_VERSION: str = "stub-0.2"

    # Tier-1 rewilding-soorten voor de ~10%-bias
    _TIER1_REWILDING: list[tuple[str, str, str]] = [
        ("Canis lupus",       "wolf",          "grey wolf"),
        ("Canis aureus",      "goudjakhals",   "golden jackal"),
        ("Lynx lynx",         "lynx",          "Eurasian lynx"),
        ("Felis silvestris",  "wilde kat",     "European wildcat"),
        ("Castor fiber",      "bever",         "Eurasian beaver"),
        ("Cervus elaphus",    "edelhert",      "red deer"),
        ("Martes martes",     "boommarter",    "pine marten"),
        ("Eliomys quercinus", "eikelmuis",     "garden dormouse"),
    ]

    # RMS-bins → (scientific, nl_name, en_name, tier)
    _BINS: list[tuple[float, str, str, str, int]] = [
        (0.01, "Vulpes vulpes",       "vos",           "red fox",           2),
        (0.02, "Capreolus capreolus", "ree",           "roe deer",          2),
        (0.04, "Meles meles",         "das",           "European badger",   1),
        (0.06, "Sciurus vulgaris",    "rode eekhoorn", "red squirrel",      2),
        (0.08, "Erinaceus europaeus", "egel",          "European hedgehog", 2),
        (0.10, "Martes martes",       "boommarter",    "pine marten",       1),
        (0.12, "Sus scrofa",          "wild zwijn",    "wild boar",         1),
        (0.15, "Lepus europaeus",     "haas",          "brown hare",        2),
    ]
    _DEFAULT: tuple[str, str, str, int] = (
        "Mustela nivalis", "wezel", "least weasel", 3
    )

    def classify(self, audio: np.ndarray, sr: int) -> dict:
        """Return a mock prediction. Deterministic: same audio → same output.

        ~10% of inputs return a tier-1 rewilding species (determined by hash
        of audio bytes) to generate Telegram alerts during testing.
        """
        audio_hash = int(
            # MD5 used for deterministic species selection only — not for security
            hashlib.md5(audio.tobytes()).hexdigest(),  # noqa: S324
            16,
        )
        rms = float(np.mean(np.abs(audio)))

        if audio_hash % 10 == 0:
            idx = audio_hash % len(self._TIER1_REWILDING)
            scientific, nl_name, en_name = self._TIER1_REWILDING[idx]
            tier = 1
            confidence = float(np.clip(rms * 8.0 + 0.3, 0.75, 0.95))
        else:
            scientific, nl_name, en_name, tier = self._DEFAULT
            for threshold, sci, nl, en, t in self._BINS:
                if rms < threshold:
                    scientific, nl_name, en_name, tier = sci, nl, en, t
                    break
            confidence = float(np.clip(rms * 8.0, 0.40, 0.95))

        return {
            "species_scientific": scientific,
            "species_nl": nl_name,
            "species_en": en_name,
            "confidence": round(confidence, 4),
            "tier": tier,
            "model_version": self.MODEL_VERSION,
        }


class YAMNetClassifier(BaseClassifier):
    """YAMNet-gebaseerde classifier met eenvoudige AudioSet→soort-mapping."""

    MODEL_VERSION = "yamnet-1.0"
    MODEL_URL = "https://tfhub.dev/google/yamnet/1"
    TARGET_SR = 16000

    # Niet-doelklassen uit AudioSet-ontology (https://research.google.com/audioset/)
    # die veel false positives geven in de buitenomgeving.
    _NON_MAMMAL_IGNORE = {0, 400, 494}  # Speech, Rustling leaves, Silence
    _GENERIC_CLASSES = {67, 68, 78}  # Animal, Domestic animals, Wild animals
    _MAPPING: dict[int, tuple[str, str, str, int]] = {
        67: ("Vulpes vulpes", "vos", "red fox", 3),
        68: ("Canis lupus familiaris", "hond (loslopend)", "domestic dog", 3),
        69: ("Canis lupus familiaris", "hond (loslopend)", "domestic dog", 3),
        74: ("Felis catus", "kat (verwilderd)", "feral cat", 3),
        78: ("Vulpes vulpes", "vos", "red fox", 3),
        79: ("Microtus arvalis", "veldmuis", "common vole", 3),
        80: ("Apodemus sylvaticus", "bosmuis", "wood mouse", 3),
        81: ("Rattus norvegicus", "bruine rat", "brown rat", 3),
        82: ("Sciurus vulgaris", "rode eekhoorn", "red squirrel", 2),
        84: ("Sus scrofa", "wild zwijn", "wild boar", 1),
        86: ("Capreolus capreolus", "ree", "roe deer", 2),
    }

    def __init__(self, min_score: float = 0.1, model: Any | None = None) -> None:
        self.min_score = float(min_score)
        if model is not None:
            self._model = model
            return

        try:
            import tensorflow_hub as hub
        except ImportError as exc:
            raise RuntimeError("YAMNet vereist tensorflow-cpu en tensorflow-hub.") from exc

        self._model = hub.load(self.MODEL_URL)

    def _preprocess(self, audio: np.ndarray, sr: int) -> np.ndarray:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return samples

        peak = float(np.max(np.abs(samples)))
        if peak > 1.0:
            samples = samples / peak

        if sr <= 0:
            return np.array([], dtype=np.float32)

        if sr != self.TARGET_SR:
            try:
                from scipy.signal import resample_poly
            except ImportError as exc:
                raise RuntimeError("scipy is nodig voor resampling naar 16kHz.") from exc
            samples = resample_poly(samples, self.TARGET_SR, sr).astype(np.float32)

        return np.clip(samples, -1.0, 1.0).astype(np.float32)

    def _rodent_fallback(self, audio_16k: np.ndarray) -> tuple[str, str, str, int]:
        if audio_16k.size == 0:
            return self._MAPPING[79]

        spectrum = np.fft.rfft(audio_16k)
        magnitudes = np.abs(spectrum)
        freqs = np.fft.rfftfreq(audio_16k.size, d=1.0 / self.TARGET_SR)
        denom = float(np.sum(magnitudes))
        if denom <= 0.0:
            return self._MAPPING[79]

        centroid_hz = float(np.sum(freqs * magnitudes) / denom)
        # Spectrale centroid benadert waar de energieband zit: hoger = kleinere piepende dieren.
        # >3kHz typisch piepband (muis), 1.8–3kHz vaker eekhoorn.
        if centroid_hz > 3000:
            return self._MAPPING[80]
        if centroid_hz > 1800:
            return self._MAPPING[82]
        return self._MAPPING[81]

    def _wild_animals_fallback(self, audio_16k: np.ndarray) -> tuple[str, str, str, int]:
        rms = float(np.sqrt(np.mean(np.square(audio_16k))))
        hour = datetime.now(tz=timezone.utc).hour
        is_night = hour < 6 or hour >= 20
        # Bij genormaliseerde audio (~[-1, 1]) is RMS > 0.08 vaak een luidere roep.
        # Overdag + luider venster mappen we conservatief op ree, anders vos.
        if rms > 0.08 and not is_night:
            return self._MAPPING[86]
        return self._MAPPING[67]

    def classify(self, audio: np.ndarray, sr: int) -> dict | None:
        audio_16k = self._preprocess(audio, sr)
        if audio_16k.size == 0:
            return None

        scores, _, _ = self._model(audio_16k)
        mean_scores = np.asarray(scores).mean(axis=0)

        best_idx = -1
        best_score = 0.0
        for idx, score in enumerate(mean_scores):
            if idx in self._NON_MAMMAL_IGNORE:
                continue
            if idx not in self._MAPPING:
                continue
            if float(score) > best_score:
                best_idx = idx
                best_score = float(score)

        if best_idx == -1 or best_score < self.min_score:
            return None

        if best_idx == 79:
            scientific, nl_name, en_name, tier = self._rodent_fallback(audio_16k)
        elif best_idx == 78:
            scientific, nl_name, en_name, tier = self._wild_animals_fallback(audio_16k)
        else:
            scientific, nl_name, en_name, tier = self._MAPPING[best_idx]

        if best_idx in self._GENERIC_CLASSES:
            tier = 3

        return {
            "species_scientific": scientific,
            "species_nl": nl_name,
            "species_en": en_name,
            "confidence": round(float(np.clip(best_score, 0.0, 1.0)), 4),
            "tier": int(tier),
            "model_version": self.MODEL_VERSION,
        }
