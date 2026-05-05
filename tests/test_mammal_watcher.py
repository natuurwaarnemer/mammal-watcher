"""
Tests voor mammal-watcher — rook-testen die de plumbing controleren.

Voert geen echte audio-files in en maakt geen netwerk-calls.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from datetime import datetime, timezone
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
            "species_en",
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

    def test_model_version_is_stub_02(self) -> None:
        """StubClassifier model_version moet stub-0.2 zijn."""
        from classifier import StubClassifier

        clf = StubClassifier()
        assert clf.MODEL_VERSION == "stub-0.2"

    def test_tier1_rewilding_bias(self) -> None:
        """~10% van inputs geeft een tier-1 rewilding soort terug."""
        from classifier import StubClassifier

        clf = StubClassifier()
        tier1_rewilding = {s for s, _, _ in clf._TIER1_REWILDING}
        tier1_count = 0
        total = 100
        for i in range(total):
            # Varieer de audio per iteratie voor diverse hashes
            audio = np.full(4800, 0.05, dtype=np.float32)
            audio[0] = float(i) / total  # lichte variatie
            result = clf.classify(audio, sr=48000)
            if result["species_scientific"] in tier1_rewilding:
                tier1_count += 1
        # Verwacht ~10%, accepteer 3–25%
        assert 3 <= tier1_count <= 25, (
            f"tier-1 rewilding bias buiten verwacht bereik: {tier1_count}/100"
        )


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfigLoading:
    """Tests voor het laden van config.yaml."""

    def test_load_default_config(self) -> None:
        """config.yaml in de repo-root moet foutloos laden."""
        from mammal_watcher import load_config

        cfg = load_config("config.yaml")
        assert "rtsp" in cfg
        assert "mqtt" in cfg
        assert "classifier" in cfg

    def test_load_custom_config(self) -> None:
        """Een custom config-dict wordt correct ingelezen."""
        from mammal_watcher import load_config

        data = {
            "rtsp": {"url": "rtsp://localhost:8554/mic"},
            "mqtt": {
                "enabled": False,
                "broker": "localhost",
                "port": 1883,
            },
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
            assert cfg["rtsp"]["url"] == "rtsp://localhost:8554/mic"
            assert cfg["mqtt"]["port"] == 1883
        finally:
            os.unlink(tmp_path)

    def test_config_has_rtsp_url(self) -> None:
        """rtsp.url moet aanwezig zijn in de config."""
        from mammal_watcher import load_config

        cfg = load_config("config.yaml")
        assert cfg["rtsp"]["url"].startswith("rtsp://")

    def test_config_has_mqtt_broker(self) -> None:
        """mqtt.broker moet aanwezig zijn in de config."""
        from mammal_watcher import load_config

        cfg = load_config("config.yaml")
        assert cfg["mqtt"]["broker"]


# ---------------------------------------------------------------------------
# Payload building tests
# ---------------------------------------------------------------------------

class TestBuildPayload:
    """Tests voor het samenstellen van het MQTT-payload."""

    def test_payload_schema(self) -> None:
        """build_payload() moet alle verplichte sleutels bevatten."""
        from mammal_watcher import build_payload

        audio = np.zeros(48000, dtype=np.float32)
        prediction = {
            "species_scientific": "Vulpes vulpes",
            "species_nl": "vos",
            "species_en": "red fox",
            "confidence": 0.82,
            "tier": 2,
            "model_version": "stub-0.2",
        }
        payload = build_payload(
            "rtsp://localhost:8554/mic", audio, 48000, prediction, {}
        )

        required = {
            "timestamp",
            "source",
            "duration_s",
            "rms",
            "species_scientific",
            "species_nl",
            "species_en",
            "confidence",
            "tier",
            "model_version",
        }
        assert required.issubset(payload.keys())

    def test_source_field(self) -> None:
        """build_payload() moet de RTSP source URL opslaan in 'source'."""
        from mammal_watcher import build_payload

        audio = np.zeros(1000, dtype=np.float32)
        prediction = {
            "species_scientific": "Vulpes vulpes",
            "species_nl": "vos",
            "species_en": "red fox",
            "confidence": 0.5,
            "tier": 2,
            "model_version": "stub-0.2",
        }
        url = "rtsp://localhost:8554/mic"
        payload = build_payload(url, audio, 48000, prediction, {})
        assert payload["source"] == url

    def test_duration_calculation(self) -> None:
        """duration_s moet gelijk zijn aan len(audio) / sr."""
        from mammal_watcher import build_payload

        sr = 48000
        seconds = 3.0
        audio = np.zeros(int(sr * seconds), dtype=np.float32)
        prediction = {
            "species_scientific": "Vulpes vulpes",
            "species_nl": "vos",
            "species_en": "red fox",
            "confidence": 0.5,
            "tier": 2,
            "model_version": "stub-0.2",
        }
        payload = build_payload(
            "rtsp://localhost:8554/mic", audio, sr, prediction, {}
        )
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

    def test_canis_lupus_in_csv(self) -> None:
        """Canis lupus (wolf) moet in de CSV staan."""
        found = False
        with open("./species_mammals_nl.csv", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row["scientific_name"] == "Canis lupus":
                    found = True
                    break
        assert found, "Canis lupus niet gevonden in CSV"

    def test_rewilding_species_in_csv(self) -> None:
        """Alle 7 nieuwe rewilding-soorten moeten in de CSV staan."""
        expected = {
            "Canis aureus",
            "Lynx lynx",
            "Felis silvestris",
            "Castor fiber",
            "Dama dama",
            "Mustela erminea",
            "Eliomys quercinus",
        }
        found: set[str] = set()
        with open("./species_mammals_nl.csv", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row["scientific_name"] in expected:
                    found.add(row["scientific_name"])
        missing = expected - found
        assert not missing, f"Ontbrekende rewilding-soorten in CSV: {missing}"


# ---------------------------------------------------------------------------
# MQTT publisher tests
# ---------------------------------------------------------------------------

class TestMQTTPublisher:
    """Tests voor MQTTPublisher met gemockte paho client."""

    def test_import(self) -> None:
        """Importeer mqtt_publisher module zonder fouten."""
        from mqtt_publisher import MQTTPublisher  # noqa: F401

    def test_publish_payload_format(self) -> None:
        """publish() moet een geldig JSON-payload sturen via de MQTT-client."""
        from mqtt_publisher import MQTTPublisher

        publisher = MQTTPublisher(
            broker="localhost",
            topic_detections="mammal/detection",
        )
        # Mock de interne client en verbindingsstatus
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.rc = 0
        mock_client.publish.return_value = mock_result
        publisher._client = mock_client
        publisher._connected = True

        payload = {
            "timestamp": "2026-05-05T18:02:13+00:00",
            "source": "rtsp://localhost:8554/mic",
            "duration_s": 5.0,
            "rms": 0.031,
            "species_scientific": "Vulpes vulpes",
            "species_nl": "vos",
            "species_en": "red fox",
            "confidence": 0.82,
            "tier": 2,
            "emoji": "🦊",
            "model_version": "stub-0.2",
        }
        publisher.publish(payload)

        mock_client.publish.assert_called_once()
        call_args = mock_client.publish.call_args
        topic = call_args[0][0]
        message = call_args[0][1]
        assert topic == "mammal/detection"
        parsed = json.loads(message)
        assert parsed["species_scientific"] == "Vulpes vulpes"
        assert parsed["source"] == "rtsp://localhost:8554/mic"

    def test_publish_skipped_when_disconnected(self) -> None:
        """publish() mag niks doen als de MQTT-client niet verbonden is."""
        from mqtt_publisher import MQTTPublisher

        publisher = MQTTPublisher(broker="localhost")
        publisher._client = None
        publisher._connected = False
        # Moet geen exception gooien
        publisher.publish({"species_nl": "vos"})

    def test_ha_discovery_config_keys(self) -> None:
        """HA discovery config moet verplichte sleutels bevatten."""
        from mqtt_publisher import MQTTPublisher

        publisher = MQTTPublisher(broker="localhost", ha_discovery=True)
        mock_client = MagicMock()
        mock_client.publish.return_value = MagicMock(rc=0)
        publisher._publish_ha_discovery(mock_client)

        mock_client.publish.assert_called_once()
        call_args = mock_client.publish.call_args
        config = json.loads(call_args[0][1])
        assert "state_topic" in config
        assert "device" in config
        assert config["device"]["manufacturer"] == "natuurwaarnemer"


# ---------------------------------------------------------------------------
# RTSP consumer tests
# ---------------------------------------------------------------------------

class TestRTSPConsumer:
    """Tests voor RTSPConsumer met gemockte av.open."""

    def test_import(self) -> None:
        """Importeer rtsp_consumer module zonder fouten."""
        from rtsp_consumer import RTSPConsumer  # noqa: F401

    def test_stop_flag(self) -> None:
        """stop() moet de consumer-vlag zetten."""
        from rtsp_consumer import RTSPConsumer

        consumer = RTSPConsumer(url="rtsp://localhost:8554/mic")
        assert not consumer._stop
        consumer.stop()
        assert consumer._stop

    def test_reconnect_backoff(self) -> None:
        """Bij falen wordt de herverbindingsvertraging verdubbeld tot reconnect_max_s."""
        from rtsp_consumer import RTSPConsumer

        consumer = RTSPConsumer(
            url="rtsp://localhost:8554/mic",
            reconnect_initial_s=1.0,
            reconnect_max_s=8.0,
        )

        call_count = 0

        def fake_open(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count >= 3:
                consumer.stop()
            raise ConnectionError("Geen verbinding")

        delays: list[float] = []

        def fake_sleep(d: float) -> None:
            delays.append(d)

        with (
            patch("rtsp_consumer.time.sleep", fake_sleep),
            patch("av.open", fake_open),
        ):
            # iter_windows gebruikt _iter_windows_av intern
            list(consumer._iter_windows_av())

        # Controleer exponentiële backoff: 1 → 2 → (stop)
        assert len(delays) >= 1
        assert delays[0] == 1.0
        if len(delays) >= 2:
            assert delays[1] == 2.0

    def test_window_size(self) -> None:
        """window_size moet gelijk zijn aan target_sr * window_seconds."""
        from rtsp_consumer import RTSPConsumer

        consumer = RTSPConsumer(
            url="rtsp://localhost:8554/mic",
            target_sr=48000,
            window_seconds=5.0,
        )
        assert consumer._window_size == 240000  # 48000 * 5
