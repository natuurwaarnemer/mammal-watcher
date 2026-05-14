"""
Review API voor MammalRadar.
Endpoints:
  GET  /api/detections         - lijst van recente detecties (clips/index.jsonl + needs_review)
  GET  /api/audio/{rel_path}   - serveer een WAV-bestand
  POST /api/confirm            - bevestig een detectie (verplaats naar confirmed)
  POST /api/reject             - wijs een detectie af (verplaats naar rejected)
  GET  /api/stats              - tellingen per soort
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

CLIPS_DIR = Path(os.environ.get("CLIPS_DIR", "/app/clips")).resolve()
FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", "/app/feedback")).resolve()
NEEDS_REVIEW_DIR = FEEDBACK_DIR / "needs_review"
CONFIRMED_DIR = FEEDBACK_DIR / "confirmed"
REJECTED_DIR = FEEDBACK_DIR / "rejected"

for directory in (CLIPS_DIR, FEEDBACK_DIR, NEEDS_REVIEW_DIR, CONFIRMED_DIR, REJECTED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MammalRadar Review API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    results: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def _list_needs_review() -> list[dict]:
    items: list[dict] = []
    if not NEEDS_REVIEW_DIR.exists():
        return items

    for sidecar in sorted(NEEDS_REVIEW_DIR.rglob("*.json"), reverse=True)[:200]:
        try:
            with open(sidecar, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue

        wav = sidecar.with_suffix(".wav")
        if wav.exists():
            data["audio_url"] = f"/api/audio/{wav.relative_to(FEEDBACK_DIR)}"
            data["source"] = "needs_review"
            data["clip_path"] = str(wav)
            items.append(data)
    return items


def _list_confirmed_clips(limit: int = 100) -> list[dict]:
    all_clips = _read_jsonl(CLIPS_DIR / "index.jsonl")
    clips = list(reversed(all_clips))[:limit]
    for clip in clips:
        filename = str(clip.get("filename", "")).strip()
        wav = CLIPS_DIR / filename
        if wav.exists():
            clip["audio_url"] = f"/api/audio/clips/{filename}"
            clip["source"] = "clips"
            clip["clip_path"] = str(wav)
    return clips


@app.get("/api/detections")
def get_detections(limit: int = 50) -> list[dict]:
    needs_review = _list_needs_review()
    clips = _list_confirmed_clips(limit=limit)
    combined = needs_review + clips
    combined.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return combined[:limit]


def _resolve_audio_path(rel_path: str) -> Path:
    if rel_path.startswith("clips/"):
        full_path = CLIPS_DIR / rel_path[len("clips/") :]
    else:
        full_path = FEEDBACK_DIR / rel_path
    return full_path.resolve()


@app.get("/api/audio/{rel_path:path}")
def get_audio(rel_path: str) -> FileResponse:
    full_path = _resolve_audio_path(rel_path)

    try:
        full_path.relative_to(CLIPS_DIR)
    except ValueError:
        try:
            full_path.relative_to(FEEDBACK_DIR)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Verboden pad") from exc

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Bestand niet gevonden")

    return FileResponse(str(full_path), media_type="audio/wav")


class ReviewRequest(BaseModel):
    clip_path: str


def _validate_path(path: Path) -> None:
    try:
        path.relative_to(FEEDBACK_DIR)
        return
    except ValueError:
        pass

    try:
        path.relative_to(CLIPS_DIR)
        return
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Verboden pad") from exc


def _move_clip_and_sidecar(clip: Path, destination_root: Path, status: str) -> Path:
    species = clip.parent.name
    target_dir = destination_root / species
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / clip.name
    shutil.move(str(clip), str(target))

    sidecar = clip.with_suffix(".json")
    if sidecar.exists():
        sidecar_target = target.with_suffix(".json")
        shutil.move(str(sidecar), str(sidecar_target))
        try:
            with open(sidecar_target, encoding="utf-8") as fh:
                data = json.load(fh)
            data["review_status"] = status
            with open(sidecar_target, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
        except (OSError, json.JSONDecodeError):
            pass
    return target


@app.post("/api/confirm")
def confirm_detection(req: ReviewRequest) -> dict[str, Any]:
    clip = Path(req.clip_path).resolve()
    _validate_path(clip)
    if not clip.exists():
        raise HTTPException(status_code=404, detail="Clip niet gevonden")
    target = _move_clip_and_sidecar(clip, CONFIRMED_DIR, "confirmed")
    return {"status": "confirmed", "path": str(target)}


@app.post("/api/reject")
def reject_detection(req: ReviewRequest) -> dict[str, Any]:
    clip = Path(req.clip_path).resolve()
    _validate_path(clip)
    if not clip.exists():
        raise HTTPException(status_code=404, detail="Clip niet gevonden")
    target = _move_clip_and_sidecar(clip, REJECTED_DIR, "rejected")
    return {"status": "rejected", "path": str(target)}


@app.get("/api/stats")
def get_stats() -> dict[str, Any]:
    confirmed = sum(1 for _ in CONFIRMED_DIR.rglob("*.wav")) if CONFIRMED_DIR.exists() else 0
    rejected = sum(1 for _ in REJECTED_DIR.rglob("*.wav")) if REJECTED_DIR.exists() else 0
    needs_review = sum(1 for _ in NEEDS_REVIEW_DIR.rglob("*.wav")) if NEEDS_REVIEW_DIR.exists() else 0
    clips_total = len(_read_jsonl(CLIPS_DIR / "index.jsonl"))
    return {
        "needs_review": needs_review,
        "confirmed": confirmed,
        "rejected": rejected,
        "clips_total": clips_total,
    }
