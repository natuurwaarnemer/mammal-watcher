"""
Tests voor mammal-watcher — rook-testen die de plumbing controleren.

Voert geen echte audio-files in en maakt geen netwerk-calls.
"""

from __future__ import annotations

import argparse
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

def test_base_classifier_is_abstract() -> None:
    """BaseClassifier mag niet direct geïnstantieerd worden."""
    from classifier import BaseClassifier

    with pytest.raises(TypeError):
        BaseClassifier()  # type: ignore[abstract]


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
            "classifier": {"model": "mammal_cnn", "min_confidence": 0.5},
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
# Main processing tests
# ---------------------------------------------------------------------------

def test_main_skips_background_detection_without_logging_or_publishing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import mammal_watcher

    fake_classifier = MagicMock()
    fake_classifier.classify.return_value = {
        "species_scientific": "Background",
        "species_nl": "Background",
        "species_en": "Background",
        "confidence": 0.99,
        "model_version": "mammal-cnn-1.0",
    }
    publisher = MagicMock()
    args = argparse.Namespace(
        config="config.yaml",
        dry_run=False,
        no_rtsp=True,
    )
    cfg = {
        "logging": {"level": "INFO"},
        "classifier": {
            "model": "mammal_cnn",
            "min_confidence": 0.4,
            "tier1_threshold": 0.75,
            "tier2_threshold": 0.5,
        },
        "species_csv": "./species_mammals_nl.csv",
        "rtsp": {"url": "rtsp://localhost:8554/mic"},
        "clips": {"clips_dir": str(tmp_path / "clips"), "enabled": True},
        "feedback": {"enabled": False, "feedback_dir": str(tmp_path / "feedback")},
        "mqtt": {"enabled": True, "broker": "localhost", "port": 1883},
        "n8n": {"enabled": False},
    }

    with (
        patch.object(mammal_watcher, "parse_args", return_value=args),
        patch.object(mammal_watcher, "load_config", return_value=cfg),
        patch.object(mammal_watcher, "_setup_logging"),
        patch.object(mammal_watcher, "load_species_index", return_value={}),
        patch.object(mammal_watcher, "MammalCNNClassifier", return_value=fake_classifier),
        patch("mqtt_publisher.MQTTPublisher", return_value=publisher),
        patch.object(mammal_watcher, "build_payload") as build_payload,
        patch.object(mammal_watcher.ClipSaver, "save") as save_clip,
        patch.object(mammal_watcher.logger, "info") as log_info,
    ):
        mammal_watcher.main()

    fake_classifier.classify.assert_called_once()
    build_payload.assert_not_called()
    save_clip.assert_not_called()
    publisher.publish.assert_not_called()
    publisher.publish_pending.assert_not_called()
    assert all("Gedetecteerd:" not in call.args[0] for call in log_info.call_args_list)
    assert capsys.readouterr().out == ""


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

        assert mock_client.publish.call_count == 2
        first_call = mock_client.publish.call_args_list[0]
        config = json.loads(first_call[0][1])
        assert "state_topic" in config
        assert "device" in config
        assert config["device"]["manufacturer"] == "natuurwaarnemer"

        pending_call = mock_client.publish.call_args_list[1]
        assert (
            pending_call[0][0]
            == "homeassistant/sensor/mammal_watcher_pending/config"
        )
        pending_config = json.loads(pending_call[0][1])
        assert pending_config["state_topic"] == "mammal/pending"

    def test_publish_pending_payload_format(self) -> None:
        """publish_pending() moet een geldig JSON-payload sturen via pending topic."""
        from mqtt_publisher import MQTTPublisher

        publisher = MQTTPublisher(
            broker="localhost",
            topic_pending="mammal/pending",
        )
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.rc = 0
        mock_client.publish.return_value = mock_result
        publisher._client = mock_client
        publisher._connected = True

        payload = {
            "species_scientific": "meles_meles",
            "species_nl": "das",
            "review_status": "needs_review",
            "review_message": "Mogelijk das — ter beoordeling",
        }
        publisher.publish_pending(payload)

        mock_client.publish.assert_called_once()
        call_args = mock_client.publish.call_args
        assert call_args[0][0] == "mammal/pending"
        parsed = json.loads(call_args[0][1])
        assert parsed["review_status"] == "needs_review"


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

    def test_iter_windows_av_peak_normalizes_low_float_audio(self) -> None:
        """PyAV-pad moet lage float-amplitude per frame piek-normaliseren."""
        from rtsp_consumer import RTSPConsumer

        consumer = RTSPConsumer(
            url="rtsp://localhost:8554/mic",
            target_sr=4,
            window_seconds=1.0,
            hop_seconds=1.0,
        )

        class DummyFrame:
            sample_rate = 4

            def to_ndarray(self) -> np.ndarray:
                return np.array([0.01, -0.02, 0.05, -0.04], dtype=np.float32)

        class DummyContainer:
            def decode(self, audio: int = 0):  # noqa: ARG002
                return iter([DummyFrame()])

            def close(self) -> None:
                return None

        with patch("av.open", return_value=DummyContainer()):
            gen = consumer._iter_windows_av()
            window, sr, _ = next(gen)
            consumer.stop()
            gen.close()

        assert sr == 4
        assert np.isclose(np.abs(window).max(), 1.0)
        assert window.dtype == np.float32

    def test_iter_windows_ffmpeg_peak_normalizes_chunks(self) -> None:
        """ffmpeg-fallback moet chunks na s16 normalisatie ook piek-normaliseren."""
        from rtsp_consumer import RTSPConsumer

        consumer = RTSPConsumer(
            url="rtsp://localhost:8554/mic",
            target_sr=4,
            window_seconds=1.0,
            hop_seconds=1.0,
        )

        raw_chunk = np.array([1000, -2000, 500, 0], dtype=np.int16).tobytes()

        class DummyStdout:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload
                self._read_count = 0

            def read(self, _size: int) -> bytes:
                if self._read_count == 0:
                    self._read_count += 1
                    return self._payload
                return b""

        class DummyProc:
            def __init__(self, payload: bytes) -> None:
                self.pid = 1234
                self.stdout = DummyStdout(payload)

            def terminate(self) -> None:
                return None

            def wait(self, timeout: float | None = None) -> int:  # noqa: ARG002
                return 0

        with patch("rtsp_consumer.subprocess.Popen", return_value=DummyProc(raw_chunk)):
            gen = consumer._iter_windows_ffmpeg()
            window, sr, _ = next(gen)
            consumer.stop()
            gen.close()

        assert sr == 4
        assert np.isclose(np.abs(window).max(), 1.0)
        assert window.dtype == np.float32
