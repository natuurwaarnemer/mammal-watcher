from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Verzamel en beheer pending detecties voor handmatige review."""

    DEFAULT_PENDING_SPECIES = (
        "meles_meles",
        "martes_martes",
        "martes_foina",
        "lynx_lynx",
    )

    def __init__(
        self,
        enabled: bool = True,
        feedback_dir: str = "./feedback",
        pending_species: list[str] | None = None,
        min_pending_confidence: float = 0.40,
        active_min_confidence: float = 0.40,
    ) -> None:
        self.enabled = enabled
        self.feedback_dir = Path(feedback_dir)
        self.needs_review_dir = self.feedback_dir / "needs_review"
        self.confirmed_dir = self.feedback_dir / "confirmed"
        self.rejected_dir = self.feedback_dir / "rejected"
        self._pending_counts: Counter[str] = Counter()

        selected_pending = pending_species or list(self.DEFAULT_PENDING_SPECIES)
        self._pending_species = {
            self.normalize_species_name(species) for species in selected_pending
        }
        self.SPECIES_STATUS: dict[str, dict[str, Any]] = {
            species: {"status": "pending", "min_confidence": min_pending_confidence}
            for species in self._pending_species
        }
        self.SPECIES_STATUS["__default__"] = {
            "status": "active",
            "min_confidence": active_min_confidence,
        }

        if self.enabled:
            self.needs_review_dir.mkdir(parents=True, exist_ok=True)
            self.confirmed_dir.mkdir(parents=True, exist_ok=True)
            self.rejected_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_species_name(species_scientific: str) -> str:
        return species_scientific.strip().lower().replace(" ", "_")

    def is_pending(self, species_scientific: str) -> bool:
        species_key = self.normalize_species_name(species_scientific)
        status = self.SPECIES_STATUS.get(
            species_key, self.SPECIES_STATUS["__default__"]
        )
        return status.get("status") == "pending"

    def get_min_confidence(self, species_scientific: str) -> float:
        species_key = self.normalize_species_name(species_scientific)
        status = self.SPECIES_STATUS.get(
            species_key, self.SPECIES_STATUS["__default__"]
        )
        return float(status.get("min_confidence", 0.0))

    def save_pending(
        self, audio: np.ndarray, sr: int, payload: dict[str, Any]
    ) -> str | None:
        if not self.enabled:
            return None

        species_scientific = str(payload.get("species_scientific", "unknown"))
        species_key = self.normalize_species_name(species_scientific)
        species_dir = self.needs_review_dir / species_key
        species_dir.mkdir(parents=True, exist_ok=True)

        timestamp_raw = str(payload.get("timestamp", ""))
        try:
            ts_dt = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
        except ValueError:
            ts_dt = datetime.now(tz=timezone.utc)
        ts_safe = ts_dt.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")

        wav_path = species_dir / f"{ts_safe}.wav"
        sidecar_path = species_dir / f"{ts_safe}.json"

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        sf.write(str(wav_path), samples, sr, subtype="PCM_16")

        metadata = {
            "timestamp": ts_dt.astimezone(timezone.utc).isoformat(),
            "species_scientific": species_scientific,
            "species_nl": payload.get("species_nl", ""),
            "confidence": float(payload.get("confidence", 0.0)),
            "review_status": payload.get("review_status", "needs_review"),
            "audio_path": str(wav_path),
        }
        with open(sidecar_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, ensure_ascii=False, indent=2)

        self._pending_counts[species_key] += 1
        logger.info(
            "Pending statistiek %s: %d clips",
            species_key,
            self._pending_counts[species_key],
        )
        return str(wav_path)

    def confirm(self, clip_path: str) -> str:
        return self._move_clip(clip_path, self.confirmed_dir)

    def reject(self, clip_path: str) -> str:
        return self._move_clip(clip_path, self.rejected_dir)

    def _move_clip(self, clip_path: str, destination_root: Path) -> str:
        clip = Path(clip_path)
        if not clip.exists():
            raise FileNotFoundError(f"Clip niet gevonden: {clip_path}")

        species = clip.parent.name
        target_dir = destination_root / species
        target_dir.mkdir(parents=True, exist_ok=True)
        target_clip = target_dir / clip.name

        shutil.move(str(clip), str(target_clip))

        source_sidecar = clip.with_suffix(".json")
        if source_sidecar.exists():
            shutil.move(str(source_sidecar), str(target_clip.with_suffix(".json")))

        from_needs_review = False
        try:
            clip.relative_to(self.needs_review_dir)
            from_needs_review = True
        except ValueError:
            from_needs_review = False

        if from_needs_review and self._pending_counts[species] > 0:
            self._pending_counts[species] -= 1
        return str(target_clip)
