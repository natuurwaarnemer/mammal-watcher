from __future__ import annotations

import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient


def _load_review_api_module(tmp_path: Path, monkeypatch):
    clips_dir = tmp_path / "clips"
    feedback_dir = tmp_path / "feedback"
    monkeypatch.setenv("CLIPS_DIR", str(clips_dir))
    monkeypatch.setenv("FEEDBACK_DIR", str(feedback_dir))
    import review_api

    return importlib.reload(review_api), clips_dir, feedback_dir


def test_stats_and_detections_include_needs_review(tmp_path: Path, monkeypatch) -> None:
    review_api, clips_dir, feedback_dir = _load_review_api_module(tmp_path, monkeypatch)
    species_dir = feedback_dir / "needs_review" / "lynx_lynx"
    species_dir.mkdir(parents=True, exist_ok=True)
    wav_path = species_dir / "20260514_101500.wav"
    wav_path.write_bytes(b"RIFF")
    with open(wav_path.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "timestamp": "2026-05-14T10:15:00+00:00",
                "species_scientific": "Lynx lynx",
                "species_nl": "lynx",
                "confidence": 0.81,
                "review_status": "needs_review",
            },
            fh,
        )

    (clips_dir / "confirmed").mkdir(parents=True, exist_ok=True)
    with open(clips_dir / "index.jsonl", "w", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "timestamp": "2026-05-14T09:00:00+00:00",
                    "filename": "confirmed/example.wav",
                    "species_scientific": "Vulpes vulpes",
                    "species_nl": "vos",
                    "confidence": 0.72,
                }
            )
            + "\n"
        )
    (clips_dir / "confirmed" / "example.wav").write_bytes(b"RIFF")

    client = TestClient(review_api.app)
    detections = client.get("/api/detections").json()
    assert any(row.get("source") == "needs_review" for row in detections)
    assert any(row.get("source") == "clips" for row in detections)

    stats = client.get("/api/stats").json()
    assert stats["needs_review"] == 1
    assert stats["clips_total"] == 1


def test_confirm_moves_clip_to_confirmed(tmp_path: Path, monkeypatch) -> None:
    review_api, _, feedback_dir = _load_review_api_module(tmp_path, monkeypatch)
    species_dir = feedback_dir / "needs_review" / "meles_meles"
    species_dir.mkdir(parents=True, exist_ok=True)
    wav_path = species_dir / "20260514_102000.wav"
    wav_path.write_bytes(b"RIFF")
    with open(wav_path.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump({"review_status": "needs_review"}, fh)

    client = TestClient(review_api.app)
    resp = client.post("/api/confirm", json={"clip_path": str(wav_path)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "confirmed"

    target = feedback_dir / "confirmed" / "meles_meles" / wav_path.name
    assert target.exists()
    with open(target.with_suffix(".json"), encoding="utf-8") as fh:
        assert json.load(fh)["review_status"] == "confirmed"
