"""
Classifier module voor mammal-watcher.

Bevat de abstracte basisklasse en een stub-implementatie die wordt gebruikt
totdat het echte ML-model beschikbaar is (PR #4).
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

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

