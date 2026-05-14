"""
Review API voor MammalRadar.
Endpoints:
  GET  /api/detections         - lijst van recente detecties (clips/index.jsonl + needs_review)
  GET  /api/audio/{rel_path}   - serveer een WAV-bestand
  GET  /api/species            - lijst met beschikbare soorten voor correcties
  POST /api/confirm            - bevestig een detectie (verplaats naar confirmed)
  POST /api/reject             - wijs een detectie af (verplaats naar rejected)
  POST /api/correct            - corrigeer soort (verplaats naar corrected/<species>)
  GET  /api/stats              - tellingen per soort
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

CLIPS_DIR = Path(os.environ.get("CLIPS_DIR", "/app/clips")).resolve()
FEEDBACK_DIR = Path(os.environ.get("FEEDBACK_DIR", "/app/feedback")).resolve()
SPECIES_CONFIG_PATH = Path(os.environ.get("SPECIES_CONFIG_PATH", "/app/species_config.json")).resolve()
NEEDS_REVIEW_DIR = FEEDBACK_DIR / "needs_review"
CONFIRMED_DIR = FEEDBACK_DIR / "confirmed"
REJECTED_DIR = FEEDBACK_DIR / "rejected"
CORRECTED_DIR = FEEDBACK_DIR / "corrected"

for directory in (CLIPS_DIR, FEEDBACK_DIR, NEEDS_REVIEW_DIR, CONFIRMED_DIR, REJECTED_DIR, CORRECTED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="MammalRadar Review API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
SAFE_PATH_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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
            rel_path = str(wav.relative_to(NEEDS_REVIEW_DIR))
            data["audio_url"] = f"/api/audio/needs_review/{rel_path}"
            data["source"] = "needs_review"
            data["clip_path"] = rel_path
            items.append(data)
    return items


def _list_confirmed_clips(limit: int = 100) -> list[dict]:
    all_clips = _read_jsonl(CLIPS_DIR / "index.jsonl")
    clips = list(reversed(all_clips))[:limit]
    for clip in clips:
        filename = str(clip.get("filename", "")).strip()
        if not filename:
            continue
        clip["clip_path"] = f"clips/{filename}"
        wav = CLIPS_DIR / filename
        if wav.exists():
            clip["audio_url"] = f"/api/audio/{clip['clip_path']}"
            clip["source"] = "clips"
        else:
            clip["source"] = clip.get("source", "clips")
    return clips


@app.get("/api/detections")
def get_detections(limit: int = 50) -> list[dict]:
    needs_review = _list_needs_review()
    clips = _list_confirmed_clips(limit=limit)
    combined = needs_review + clips
    combined.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return combined[:limit]


def _safe_relative_parts(rel_path: str) -> list[str]:
    normalized = rel_path.strip().replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if not parts:
        raise HTTPException(status_code=403, detail="Verboden pad")
    for part in parts:
        if part in {".", ".."} or not SAFE_PATH_PART.fullmatch(part):
            raise HTTPException(status_code=403, detail="Verboden pad")
    return parts


def _resolve_relative_path(base_dir: Path, rel_path: str) -> Path:
    full_path = base_dir.joinpath(*_safe_relative_parts(rel_path)).resolve()
    try:
        full_path.relative_to(base_dir)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Verboden pad") from exc
    return full_path


def _resolve_audio_path(rel_path: str) -> Path:
    if rel_path.startswith("clips/"):
        return _resolve_relative_path(CLIPS_DIR, rel_path[len("clips/") :])
    if rel_path.startswith("needs_review/"):
        return _resolve_relative_path(NEEDS_REVIEW_DIR, rel_path[len("needs_review/") :])
    if rel_path.startswith("confirmed/"):
        return _resolve_relative_path(CONFIRMED_DIR, rel_path[len("confirmed/") :])
    if rel_path.startswith("rejected/"):
        return _resolve_relative_path(REJECTED_DIR, rel_path[len("rejected/") :])
    if rel_path.startswith("corrected/"):
        return _resolve_relative_path(CORRECTED_DIR, rel_path[len("corrected/") :])
    return _resolve_relative_path(FEEDBACK_DIR, rel_path)


@app.get("/api/audio/{rel_path:path}")
def get_audio(rel_path: str) -> FileResponse:
    full_path = _resolve_audio_path(rel_path)

    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Bestand niet gevonden")
    if full_path.suffix.lower() != ".wav":
        raise HTTPException(status_code=400, detail="Alleen WAV-bestanden toegestaan")

    return FileResponse(str(full_path), media_type="audio/wav")


class ReviewRequest(BaseModel):
    clip_path: str


class CorrectRequest(ReviewRequest):
    corrected_species: str


def _species_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower().replace(" ", "_"))
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        raise HTTPException(status_code=400, detail="Ongeldige soort")
    return slug


def _read_species_config() -> list[dict[str, Any]]:
    if not SPECIES_CONFIG_PATH.exists():
        return []
    try:
        with open(SPECIES_CONFIG_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="species_config.json is ongeldig") from exc
    return list(data.get("species", []))


def _move_clip_and_sidecar(
    clip: Path,
    destination_root: Path,
    status: str,
    *,
    destination_species: str | None = None,
    metadata_updates: dict[str, Any] | None = None,
) -> Path:
    try:
        clip.relative_to(NEEDS_REVIEW_DIR)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Verboden pad") from exc

    species = _species_slug(destination_species or clip.parent.name)
    clip_name = _safe_relative_parts(clip.name)[0]
    target = _resolve_relative_path(destination_root, f"{species}/{clip_name}")
    target_dir = target.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(clip), str(target))

    sidecar = clip.with_suffix(".json")
    if sidecar.exists():
        sidecar_name = Path(clip_name).with_suffix(".json").name
        sidecar_target = _resolve_relative_path(destination_root, f"{species}/{sidecar_name}")
        shutil.move(str(sidecar), str(sidecar_target))
        try:
            with open(sidecar_target, encoding="utf-8") as fh:
                data = json.load(fh)
            data["review_status"] = status
            if metadata_updates:
                data.update(metadata_updates)
            with open(sidecar_target, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
        except (OSError, json.JSONDecodeError):
            pass
    return target


@app.get("/api/species")
def get_species() -> list[dict[str, str]]:
    species = []
    for item in _read_species_config():
        scientific = str(item.get("scientific", "")).strip()
        nl_name = str(item.get("nl", "")).strip()
        if scientific and nl_name:
            species.append(
                {
                    "scientific": scientific,
                    "nl": nl_name,
                    "slug": _species_slug(scientific),
                }
            )
    return species


@app.post("/api/confirm")
def confirm_detection(req: ReviewRequest) -> dict[str, Any]:
    clip = _resolve_relative_path(NEEDS_REVIEW_DIR, req.clip_path)
    if not clip.exists():
        raise HTTPException(status_code=404, detail="Clip niet gevonden")
    target = _move_clip_and_sidecar(clip, CONFIRMED_DIR, "confirmed")
    return {"status": "confirmed", "path": str(target)}


@app.post("/api/reject")
def reject_detection(req: ReviewRequest) -> dict[str, Any]:
    clip = _resolve_relative_path(NEEDS_REVIEW_DIR, req.clip_path)
    if not clip.exists():
        raise HTTPException(status_code=404, detail="Clip niet gevonden")
    target = _move_clip_and_sidecar(clip, REJECTED_DIR, "rejected")
    return {"status": "rejected", "path": str(target)}


@app.post("/api/correct")
def correct_detection(req: CorrectRequest) -> dict[str, Any]:
    clip = _resolve_relative_path(NEEDS_REVIEW_DIR, req.clip_path)
    if not clip.exists():
        raise HTTPException(status_code=404, detail="Clip niet gevonden")

    corrected_species = _species_slug(req.corrected_species)
    original_species = clip.parent.name
    target = _move_clip_and_sidecar(
        clip,
        CORRECTED_DIR,
        "corrected",
        destination_species=corrected_species,
        metadata_updates={
            "original_species": original_species,
            "corrected_species": corrected_species,
        },
    )
    return {"status": "corrected", "path": str(target), "corrected_species": corrected_species}


@app.get("/api/stats")
def get_stats() -> dict[str, Any]:
    confirmed = sum(1 for _ in CONFIRMED_DIR.rglob("*.wav")) if CONFIRMED_DIR.exists() else 0
    rejected = sum(1 for _ in REJECTED_DIR.rglob("*.wav")) if REJECTED_DIR.exists() else 0
    corrected = sum(1 for _ in CORRECTED_DIR.rglob("*.wav")) if CORRECTED_DIR.exists() else 0
    needs_review = sum(1 for _ in NEEDS_REVIEW_DIR.rglob("*.wav")) if NEEDS_REVIEW_DIR.exists() else 0
    clips_total = len(_read_jsonl(CLIPS_DIR / "index.jsonl"))
    return {
        "needs_review": needs_review,
        "confirmed": confirmed,
        "rejected": rejected,
        "corrected": corrected,
        "clips_total": clips_total,
    }
