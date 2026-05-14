from __future__ import annotations

import importlib
import json
from pathlib import Path

from fastapi.testclient import TestClient


def _load_review_api_module(tmp_path: Path, monkeypatch):
    clips_dir = tmp_path / "clips"
    feedback_dir = tmp_path / "feedback"
    species_file = tmp_path / "species_config.json"
    species_file.write_text(
        json.dumps(
            {
                "species": [
                    {"scientific": "Canis lupus", "nl": "wolf"},
                    {"scientific": "Lynx lynx", "nl": "lynx"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLIPS_DIR", str(clips_dir))
    monkeypatch.setenv("FEEDBACK_DIR", str(feedback_dir))
    monkeypatch.setenv("SPECIES_CONFIG_PATH", str(species_file))
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
    needs_review = next(row for row in detections if row.get("source") == "needs_review")
    confirmed = next(row for row in detections if row.get("source") == "clips")
    assert needs_review["clip_path"] == "lynx_lynx/20260514_101500.wav"
    assert needs_review["audio_url"] == "/api/audio/needs_review/lynx_lynx/20260514_101500.wav"
    assert confirmed["clip_path"] == "clips/confirmed/example.wav"
    assert confirmed["audio_url"] == "/api/audio/clips/confirmed/example.wav"

    stats = client.get("/api/stats").json()
    assert stats["needs_review"] == 1
    assert stats["clips_total"] == 1
    assert stats["corrected"] == 0


def test_confirm_moves_clip_to_confirmed(tmp_path: Path, monkeypatch) -> None:
    review_api, _, feedback_dir = _load_review_api_module(tmp_path, monkeypatch)
    species_dir = feedback_dir / "needs_review" / "meles_meles"
    species_dir.mkdir(parents=True, exist_ok=True)
    wav_path = species_dir / "20260514_102000.wav"
    wav_path.write_bytes(b"RIFF")
    with open(wav_path.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump({"review_status": "needs_review"}, fh)

    client = TestClient(review_api.app)
    resp = client.post("/api/confirm", json={"clip_path": f"meles_meles/{wav_path.name}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "confirmed"

    target = feedback_dir / "confirmed" / "meles_meles" / wav_path.name
    assert target.exists()
    with open(target.with_suffix(".json"), encoding="utf-8") as fh:
        assert json.load(fh)["review_status"] == "confirmed"


def test_confirm_rejects_path_traversal(tmp_path: Path, monkeypatch) -> None:
    review_api, _, _ = _load_review_api_module(tmp_path, monkeypatch)
    client = TestClient(review_api.app)
    resp = client.post("/api/confirm", json={"clip_path": "../outside.wav"})
    assert resp.status_code == 403


def test_get_species_returns_species_from_config(tmp_path: Path, monkeypatch) -> None:
    review_api, _, _ = _load_review_api_module(tmp_path, monkeypatch)
    client = TestClient(review_api.app)

    resp = client.get("/api/species")

    assert resp.status_code == 200
    assert resp.json() == [
        {"scientific": "Canis lupus", "nl": "wolf", "slug": "canis_lupus"},
        {"scientific": "Lynx lynx", "nl": "lynx", "slug": "lynx_lynx"},
    ]


def test_correct_moves_clip_to_corrected_species(tmp_path: Path, monkeypatch) -> None:
    review_api, _, feedback_dir = _load_review_api_module(tmp_path, monkeypatch)
    species_dir = feedback_dir / "needs_review" / "lynx_lynx"
    species_dir.mkdir(parents=True, exist_ok=True)
    wav_path = species_dir / "20260514_103000.wav"
    wav_path.write_bytes(b"RIFF")
    with open(wav_path.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump({"species_scientific": "Lynx lynx", "review_status": "needs_review"}, fh)

    client = TestClient(review_api.app)
    resp = client.post(
        "/api/correct",
        json={"clip_path": "lynx_lynx/20260514_103000.wav", "corrected_species": "canis_lupus"},
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "corrected"
    target = feedback_dir / "corrected" / "canis_lupus" / wav_path.name
    assert target.exists()
    with open(target.with_suffix(".json"), encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert metadata["review_status"] == "corrected"
    assert metadata["original_species"] == "lynx_lynx"
    assert metadata["corrected_species"] == "canis_lupus"
