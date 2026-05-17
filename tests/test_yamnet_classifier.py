from __future__ import annotations

import json
from pathlib import Path

import numpy as np


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
