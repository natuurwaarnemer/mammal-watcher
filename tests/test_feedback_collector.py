from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from feedback_collector import FeedbackCollector


def test_is_pending_for_pending_and_active_species(tmp_path: Path) -> None:
    collector = FeedbackCollector(
        enabled=True,
        feedback_dir=str(tmp_path / "feedback"),
    )

    assert collector.is_pending("meles_meles")
    assert collector.is_pending("Meles meles")
    assert not collector.is_pending("vulpes_vulpes")


def test_save_pending_creates_structure_and_sidecar(tmp_path: Path) -> None:
    collector = FeedbackCollector(
        enabled=True,
        feedback_dir=str(tmp_path / "feedback"),
    )
    audio = np.zeros(48000, dtype=np.float32)
    payload = {
        "timestamp": datetime(2026, 5, 13, 21, 34, 0, tzinfo=timezone.utc).isoformat(),
        "species_scientific": "meles_meles",
        "species_nl": "das",
        "confidence": 0.67,
        "review_status": "needs_review",
    }

    clip_path = collector.save_pending(audio, 48000, payload)
    assert clip_path is not None
    clip = Path(clip_path)
    sidecar = clip.with_suffix(".json")
    assert clip.exists()
    assert sidecar.exists()

    with open(sidecar, encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert metadata["species_scientific"] == "meles_meles"
    assert metadata["species_nl"] == "das"
    assert metadata["confidence"] == 0.67
    assert metadata["review_status"] == "needs_review"
    assert metadata["audio_path"] == str(clip)


def test_confirm_moves_clip_to_confirmed(tmp_path: Path) -> None:
    collector = FeedbackCollector(
        enabled=True,
        feedback_dir=str(tmp_path / "feedback"),
    )
    audio = np.zeros(48000, dtype=np.float32)
    payload = {
        "timestamp": datetime(2026, 5, 13, 21, 34, 0, tzinfo=timezone.utc).isoformat(),
        "species_scientific": "meles_meles",
        "species_nl": "das",
        "confidence": 0.67,
    }

    clip_path = collector.save_pending(audio, 48000, payload)
    assert clip_path is not None

    confirmed_path = collector.confirm(clip_path)
    confirmed_clip = Path(confirmed_path)
    assert confirmed_clip.exists()
    assert "confirmed" in confirmed_path
    assert not Path(clip_path).exists()
    assert confirmed_clip.with_suffix(".json").exists()


def test_reject_moves_clip_to_rejected(tmp_path: Path) -> None:
    collector = FeedbackCollector(
        enabled=True,
        feedback_dir=str(tmp_path / "feedback"),
    )
    audio = np.zeros(48000, dtype=np.float32)
    payload = {
        "timestamp": datetime(2026, 5, 13, 21, 35, 0, tzinfo=timezone.utc).isoformat(),
        "species_scientific": "lynx_lynx",
        "species_nl": "lynx",
        "confidence": 0.51,
    }

    clip_path = collector.save_pending(audio, 48000, payload)
    assert clip_path is not None

    rejected_path = collector.reject(clip_path)
    rejected_clip = Path(rejected_path)
    assert rejected_clip.exists()
    assert "rejected" in rejected_path
    assert not Path(clip_path).exists()
    assert rejected_clip.with_suffix(".json").exists()
