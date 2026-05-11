from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "dataset" / "prepare_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_dataset", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_species_supports_json_and_yaml(tmp_path: Path) -> None:
    module = _load_module()

    json_file = tmp_path / "species_config.json"
    json_file.write_text(
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

    yaml_file = tmp_path / "species_targets.yaml"
    yaml_file.write_text(
        "species:\n"
        "  - scientific: Vulpes vulpes\n"
        "    nl: vos\n",
        encoding="utf-8",
    )

    json_species = module._load_species(json_file)
    yaml_species = module._load_species(yaml_file)

    assert json_species["canis_lupus"]["nl"] == "wolf"
    assert json_species["lynx_lynx"]["scientific"] == "Lynx lynx"
    assert yaml_species["vulpes_vulpes"]["nl"] == "vos"


def test_audio_files_picks_supported_extensions(tmp_path: Path) -> None:
    module = _load_module()
    species_dir = tmp_path / "canis_lupus"
    species_dir.mkdir()

    for filename in [
        "inat_1.mp3",
        "inat_2.wav",
        "inat_3.ogg",
        "inat_4.flac",
        "inat_5.m4a",
        "inat_6.aac",
        "inat_7.opus",
        "gbif_8.bin",
        "notes.txt",
    ]:
        (species_dir / filename).write_bytes(b"dummy")

    audio_files = module._audio_files(species_dir)
    names = {p.name for p in audio_files}

    assert names == {
        "inat_1.mp3",
        "inat_2.wav",
        "inat_3.ogg",
        "inat_4.flac",
        "inat_5.m4a",
        "inat_6.aac",
        "inat_7.opus",
    }


def test_detect_source_uses_gbif_prefix() -> None:
    module = _load_module()

    assert module._detect_source(Path("gbif_123.mp3")) == "gbif"
    assert module._detect_source(Path("inat_123.mp3")) == "inaturalist"
    assert module._detect_source(Path("recording.wav")) == "inaturalist"
