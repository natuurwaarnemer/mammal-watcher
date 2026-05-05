"""
Classifier module voor mammal-watcher.

Bevat de abstracte basisklasse en een stub-implementatie die wordt gebruikt
totdat het echte ML-model beschikbaar is (PR #2).
"""

from __future__ import annotations

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
            species_scientific, species_nl, confidence, tier, model_version
        """


class StubClassifier(BaseClassifier):
    """STUB — wordt vervangen in PR #2 door echt model.

    Geeft een deterministische mock-voorspelling terug op basis van het
    gemiddelde absolute amplitude (RMS-benadering) van het audio-fragment.
    Verschillende RMS-bins mappen op verschillende soorten zodat de pipeline
    end-to-end getest kan worden zonder een echt ML-model.
    """

    MODEL_VERSION: str = "stub-0.1"

    # RMS-bins → (scientific, nl_name, tier)
    _BINS: list[tuple[float, str, str, int]] = [
        (0.01, "Vulpes vulpes",       "vos",             2),
        (0.02, "Capreolus capreolus", "ree",             2),
        (0.04, "Meles meles",         "das",             1),
        (0.06, "Sciurus vulgaris",    "rode eekhoorn",   2),
        (0.08, "Erinaceus europaeus", "egel",            2),
        (0.10, "Martes martes",       "boommarter",      1),
        (0.12, "Sus scrofa",          "wild zwijn",      1),
        (0.15, "Lepus europaeus",     "haas",            2),
    ]
    _DEFAULT: tuple[str, str, int] = ("Mustela nivalis", "wezel", 3)

    def classify(self, audio: np.ndarray, sr: int) -> dict:
        """Return a mock prediction based on the mean absolute amplitude.

        The result is fully deterministic: same audio → same output.
        """
        rms = float(np.mean(np.abs(audio)))

        scientific, nl_name, tier = self._DEFAULT
        for threshold, sci, nl, t in self._BINS:
            if rms < threshold:
                scientific, nl_name, tier = sci, nl, t
                break

        # Confidence is mapped linearly from rms, clamped to [0.40, 0.95].
        confidence = float(np.clip(rms * 8.0, 0.40, 0.95))

        return {
            "species_scientific": scientific,
            "species_nl": nl_name,
            "confidence": round(confidence, 4),
            "tier": tier,
            "model_version": self.MODEL_VERSION,
        }
