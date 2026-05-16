from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "dataset" / "download_gbif.py"
    spec = importlib.util.spec_from_file_location("download_gbif", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_species_reads_gbif_taxon_keys_from_config() -> None:
    module = _load_module()
    repo_root = Path(__file__).resolve().parents[1]

    species = module._load_species(repo_root / "species_config.json")
    assert species
    assert len(species) == 15

    by_scientific = {s["scientific"]: s["gbif_taxon_key"] for s in species}
    assert "Canis lupus" in by_scientific
    assert by_scientific["Vulpes vulpes"] == 5219408
    assert by_scientific["Lynx lynx"] == 5219072
    assert by_scientific["Felis silvestris"] == 2435098


def test_collect_sound_media_filters_non_sound_and_missing_identifier() -> None:
    module = _load_module()

    occurrences = [
        {
            "occurrenceID": "occ-1",
            "taxonKey": 5219408,
            "license": "CC-BY",
            "media": [
                {"type": "StillImage", "identifier": "https://example.org/image.jpg"},
                {"type": "Sound", "identifier": "https://example.org/a.mp3", "format": "audio/mpeg"},
                {"type": "SOUND", "identifier": "https://example.org/b.wav", "format": "audio/wav"},
                {"type": "Sound", "format": "audio/wav"},
            ],
        }
    ]

    sounds = module._collect_sound_media(occurrences)
    assert len(sounds) == 2
    assert sounds[0]["occurrence_id"] == "occ-1"
    assert sounds[0]["url"] == "https://example.org/a.mp3"
    assert sounds[0]["format"] == "audio/mpeg"
    assert sounds[1]["url"] == "https://example.org/b.wav"


def test_guess_extension_uses_format_when_url_has_no_suffix() -> None:
    module = _load_module()

    assert module._guess_extension("https://example.org/download", "audio/mpeg") == "mp3"
    assert module._guess_extension("https://example.org/download", "audio/wav") == "wav"
    assert module._guess_extension("https://example.org/download", None) == "bin"
