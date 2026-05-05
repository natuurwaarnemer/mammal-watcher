"""
Tests voor mammal-watcher — rook-testen die de plumbing controleren.

Voert geen echte audio-files in en maakt geen netwerk-calls.
"""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------

class TestStubClassifier:
    """Tests voor de StubClassifier stub-implementatie."""

    def test_import(self) -> None:
        """Importeer de classifier module zonder fouten."""
        from classifier import StubClassifier  # noqa: F401

    def test_returns_required_keys(self) -> None:
        """classify() moet alle verplichte sleutels bevatten."""
        from classifier import StubClassifier

        clf = StubClassifier()
        audio = np.zeros(48000, dtype=np.float32)
        result = clf.classify(audio, sr=48000)

        required = {
            "species_scientific",
            "species_nl",
            "confidence",
            "tier",
            "model_version",
        }
        assert required.issubset(result.keys()), (
            f"Ontbrekende sleutels: {required - result.keys()}"
        )

    def test_confidence_in_range(self) -> None:
        """confidence moet tussen 0 en 1 liggen."""
        from classifier import StubClassifier

        clf = StubClassifier()
        for amplitude in [0.0, 0.01, 0.05, 0.10, 0.50, 1.0]:
            audio = np.full(4800, amplitude, dtype=np.float32)
            result = clf.classify(audio, sr=48000)
            assert 0.0 <= result["confidence"] <= 1.0, (
                f"confidence {result['confidence']} buiten bereik voor amplitude {amplitude}"
            )

    def test_tier_is_int(self) -> None:
        """tier moet een int zijn (1, 2 of 3)."""
        from classifier import StubClassifier

        clf = StubClassifier()
        audio = np.random.default_rng(42).random(48000).astype(np.float32)
        result = clf.classify(audio, sr=48000)
        assert isinstance(result["tier"], int)
        assert result["tier"] in (1, 2, 3)

    def test_deterministic(self) -> None:
        """Zelfde audio → zelfde resultaat."""
        from classifier import StubClassifier

        clf = StubClassifier()
        audio = np.full(8000, 0.05, dtype=np.float32)
        r1 = clf.classify(audio, sr=48000)
        r2 = clf.classify(audio, sr=48000)
        assert r1 == r2

    def test_base_classifier_is_abstract(self) -> None:
        """BaseClassifier mag niet direct geïnstantieerd worden."""
        from classifier import BaseClassifier

        with pytest.raises(TypeError):
            BaseClassifier()  # type: ignore[abstract]

    def test_model_version_is_stub(self) -> None:
        """StubClassifier moet 'stub' in de model_version string hebben."""
        from classifier import StubClassifier

        clf = StubClassifier()
        audio = np.zeros(4800, dtype=np.float32)
        result = clf.classify(audio, sr=48000)
        assert "stub" in result["model_version"].lower()


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfigLoading:
    """Tests voor het laden van config.yaml."""

    def test_load_default_config(self) -> None:
        """config.yaml in de repo-root moet foutloos laden."""
        from mammal_watcher import load_config

        cfg = load_config("config.yaml")
        assert "watch" in cfg
        assert "n8n" in cfg
        assert "classifier" in cfg

    def test_load_custom_config(self) -> None:
        """Een custom config-dict wordt correct ingelezen."""
        from mammal_watcher import load_config

        data = {
            "watch": {"snippet_dir": "/tmp/test", "file_extensions": [".wav"]},
            "n8n": {"webhook_url": "http://localhost:5678/test", "timeout_s": 5},
            "classifier": {"model": "stub", "min_confidence": 0.5},
            "species_csv": "./species_mammals_nl.csv",
            "logging": {"level": "DEBUG"},
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        ) as fh:
            yaml.dump(data, fh)
            tmp_path = fh.name

        try:
            cfg = load_config(tmp_path)
            assert cfg["watch"]["snippet_dir"] == "/tmp/test"
            assert cfg["n8n"]["timeout_s"] == 5
        finally:
            os.unlink(tmp_path)

    def test_config_has_snippet_dir(self) -> None:
        """snippet_dir moet aanwezig zijn in de config."""
        from mammal_watcher import load_config

        cfg = load_config("config.yaml")
        assert cfg["watch"]["snippet_dir"]

    def test_config_has_webhook_url(self) -> None:
        """webhook_url moet aanwezig zijn in de config."""
        from mammal_watcher import load_config

        cfg = load_config("config.yaml")
        assert cfg["n8n"]["webhook_url"].startswith("http")


# ---------------------------------------------------------------------------
# Payload building tests
# ---------------------------------------------------------------------------

class TestBuildPayload:
    """Tests voor het samenstellen van het n8n-payload."""

    def test_payload_schema(self) -> None:
        """build_payload() moet alle verplichte sleutels bevatten."""
        from mammal_watcher import build_payload

        audio = np.zeros(48000, dtype=np.float32)
        prediction = {
            "species_scientific": "Vulpes vulpes",
            "species_nl": "vos",
            "confidence": 0.82,
            "tier": 2,
            "model_version": "stub-0.1",
        }
        payload = build_payload("/tmp/test.wav", audio, 48000, prediction, {})

        required = {
            "timestamp",
            "audio_path",
            "duration_s",
            "rms",
            "species_scientific",
            "species_nl",
            "confidence",
            "tier",
            "model_version",
        }
        assert required.issubset(payload.keys())

    def test_duration_calculation(self) -> None:
        """duration_s moet gelijk zijn aan len(audio) / sr."""
        from mammal_watcher import build_payload

        sr = 48000
        seconds = 3.0
        audio = np.zeros(int(sr * seconds), dtype=np.float32)
        prediction = {
            "species_scientific": "Vulpes vulpes",
            "species_nl": "vos",
            "confidence": 0.5,
            "tier": 2,
            "model_version": "stub-0.1",
        }
        payload = build_payload("/tmp/test.wav", audio, sr, prediction, {})
        assert abs(payload["duration_s"] - seconds) < 0.01


# ---------------------------------------------------------------------------
# Species CSV tests
# ---------------------------------------------------------------------------

class TestSpeciesCSV:
    """Tests voor het inlezen van de soorten-CSV."""

    def test_csv_loads(self) -> None:
        """species_mammals_nl.csv moet foutloos ingelezen worden."""
        from mammal_watcher import load_species_index

        index = load_species_index("./species_mammals_nl.csv")
        assert len(index) > 0

    def test_csv_has_required_columns(self) -> None:
        """Elke rij in de CSV moet de verplichte kolommen bevatten."""
        required_columns = {
            "scientific_name",
            "nl_name",
            "en_name",
            "category",
            "tier",
            "emoji",
            "realistic_for_field_edge",
            "notes",
        }
        with open("./species_mammals_nl.csv", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            columns = set(reader.fieldnames or [])
        assert required_columns.issubset(columns)

    def test_bats_marked_not_realistic(self) -> None:
        """Vleermuizen mogen niet als realistic_for_field_edge=true staan."""
        with open("./species_mammals_nl.csv", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row["category"] == "chiroptera":
                    assert row["realistic_for_field_edge"].lower() == "false", (
                        f"{row['scientific_name']} is een vleermuis maar staat als realistisch"
                    )

    def test_tier1_species_present(self) -> None:
        """Er moeten Tier-1 soorten in de CSV staan."""
        tier1_found = False
        with open("./species_mammals_nl.csv", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row["tier"] == "1":
                    tier1_found = True
                    break
        assert tier1_found, "Geen Tier-1 soorten gevonden in CSV"
