from __future__ import annotations

import importlib.util
import io
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "dataset" / "download_naturelm.py"
    spec = importlib.util.spec_from_file_location("download_naturelm", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_slug_normalizes_scientific_name() -> None:
    module = _load_module()
    assert module._slug("Vulpes vulpes") == "vulpes_vulpes"
    assert module._slug("  Canis   lupus  ") == "canis_lupus"


def test_save_audio_as_wav_decodes_raw_duckdb_struct(tmp_path) -> None:
    """DuckDB/Parquet geeft audio terug als {'bytes': <encoded>, 'path': ...} —
    dit is een ander formaat dan de HF datasets Audio-feature ({'array', 'sampling_rate'})
    en moet apart gedecodeerd worden."""
    module = _load_module()
    if not module.HAS_SOUNDFILE:
        return

    import numpy as np
    import soundfile as sf

    tone = np.zeros(1600, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, tone, 16000, format="WAV", subtype="PCM_16")
    raw_bytes = buf.getvalue()

    dest = tmp_path / "clip.wav"
    ok = module._save_audio_as_wav({"bytes": raw_bytes, "path": "orig.wav"}, dest)

    assert ok
    assert dest.exists()
    array, sr = sf.read(str(dest))
    assert sr == 16000
    assert len(array) == 1600


def test_save_audio_as_wav_still_handles_hf_decoded_dict(tmp_path) -> None:
    """Oude vorm (HF datasets Audio-feature, al gedecodeerd) moet blijven werken."""
    module = _load_module()
    if not module.HAS_SOUNDFILE:
        return

    dest = tmp_path / "clip.wav"
    ok = module._save_audio_as_wav(
        {"array": [0.0] * 1600, "sampling_rate": 16000}, dest
    )
    assert ok
    assert dest.exists()


def test_plan_downloads_dedupes_same_source_recording_across_task_types() -> None:
    """Eenzelfde brondata-opname (file_name) verschijnt vaak onder meerdere
    task-types in de index — moet maar één keer ingepland worden."""
    module = _load_module()

    entries = [
        {"id": "a", "species_slug": "vulpes_vulpes", "file_name": "rec1.wav",
         "parquet_url": "shard0.parquet", "task": "taxonomic-classification"},
        {"id": "b", "species_slug": "vulpes_vulpes", "file_name": "rec1.wav",
         "parquet_url": "shard0.parquet", "task": "genus-detection"},
        {"id": "c", "species_slug": "vulpes_vulpes", "file_name": "rec2.wav",
         "parquet_url": "shard1.parquet", "task": "caption-common"},
    ]

    plan = module._plan_downloads(
        entries, target_slugs={"vulpes_vulpes"}, counters={"vulpes_vulpes": 0}, max_per_species=50
    )

    planned_ids = {e["id"] for entries_ in plan.values() for e in entries_}
    assert planned_ids == {"a", "c"}  # "b" is dezelfde opname als "a", overgeslagen


def test_plan_downloads_skips_species_already_at_quota() -> None:
    module = _load_module()

    entries = [
        {"id": "a", "species_slug": "vulpes_vulpes", "file_name": "rec1.wav",
         "parquet_url": "shard0.parquet", "task": "taxonomic-classification"},
    ]

    plan = module._plan_downloads(
        entries, target_slugs={"vulpes_vulpes"}, counters={"vulpes_vulpes": 50}, max_per_species=50
    )

    assert plan == {}


def test_plan_downloads_ignores_non_target_species() -> None:
    module = _load_module()

    entries = [
        {"id": "a", "species_slug": "sus_scrofa", "file_name": "rec1.wav",
         "parquet_url": "shard0.parquet", "task": "taxonomic-classification"},
    ]

    plan = module._plan_downloads(
        entries, target_slugs={"vulpes_vulpes"}, counters={"vulpes_vulpes": 0}, max_per_species=50
    )

    assert plan == {}


def test_plan_downloads_skips_filenames_already_downloaded_in_earlier_run() -> None:
    """Cross-run dedupe: als een vorige run 'rec1.wav' al opsloeg (onder een
    ander id/task-type), mag een latere run 'm niet nogmaals inplannen."""
    module = _load_module()

    entries = [
        {"id": "a", "species_slug": "vulpes_vulpes", "file_name": "rec1.wav",
         "parquet_url": "shard0.parquet", "task": "taxonomic-classification"},
        {"id": "b", "species_slug": "vulpes_vulpes", "file_name": "rec2.wav",
         "parquet_url": "shard1.parquet", "task": "caption-common"},
    ]

    plan = module._plan_downloads(
        entries,
        target_slugs={"vulpes_vulpes"},
        counters={"vulpes_vulpes": 0},
        max_per_species=50,
        already_downloaded={"vulpes_vulpes": {"rec1.wav"}},
    )

    planned_ids = {e["id"] for entries_ in plan.values() for e in entries_}
    assert planned_ids == {"b"}  # "a" (rec1.wav) is al gedownload in een eerdere run


def test_downloaded_filenames_roundtrip(tmp_path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "DOWNLOADED_FILENAMES_FILE", tmp_path / "downloaded.json")

    assert module._load_downloaded_filenames() == {}

    module._save_downloaded_filenames({"vulpes_vulpes": {"rec1.wav", "rec2.wav"}})
    loaded = module._load_downloaded_filenames()

    assert loaded == {"vulpes_vulpes": {"rec1.wav", "rec2.wav"}}
