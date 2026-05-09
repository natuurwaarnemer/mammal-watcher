"""
mammal_watcher.py — Hoofd-entry-point voor de mammal-watcher service.

Leest een RTSP-stroom via MediaMTX relay, classificeert 5-seconden
audio-vensters op zoogdiergeluiden en publiceert detecties via MQTT
(en optioneel via een n8n webhook).

Gebruik:
    python mammal_watcher.py --config config.yaml
    python mammal_watcher.py --dry-run         # print payloads, geen MQTT/HTTP
    python mammal_watcher.py --no-rtsp --dry-run  # CI smoke-test, exit 0

Graceful shutdown via SIGTERM (Docker-veilig).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import soundfile as sf

from classifier import BaseClassifier, StubClassifier, YAMNetClassifier

__version__ = "0.2.0"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    """Configureer de root logger met tijdstempelformat."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


logger = logging.getLogger(__name__)


class ClipSaver:
    """Sla detectie-audio op als WAV met JSONL-index voor latere training."""

    def __init__(
        self,
        clips_dir: str,
        enabled: bool = True,
        save_uncertain: bool = True,
        uncertain_threshold: float = 0.5,
        max_clips_per_day: int = 500,
    ) -> None:
        self.enabled = enabled
        self.save_uncertain = save_uncertain
        self.uncertain_threshold = float(uncertain_threshold)
        self.max_clips_per_day = int(max_clips_per_day)
        self.clips_dir = Path(clips_dir)
        self.confirmed_dir = self.clips_dir / "confirmed"
        self.uncertain_dir = self.clips_dir / "uncertain"
        self.index_path = self.clips_dir / "index.jsonl"
        if self.enabled:
            self.confirmed_dir.mkdir(parents=True, exist_ok=True)
            self.uncertain_dir.mkdir(parents=True, exist_ok=True)
            self.index_path.touch(exist_ok=True)

    def _slug(self, value: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
        return slug or "unknown"

    def _count_clips_today(self, day_prefix: str) -> int:
        if not self.index_path.exists():
            return 0
        count = 0
        with open(self.index_path, encoding="utf-8") as fh:
            for line in fh:
                if f"\"timestamp\": \"{day_prefix}" in line:
                    count += 1
        return count

    def save(self, audio: np.ndarray, sr: int, payload: dict[str, Any]) -> str | None:
        if not self.enabled:
            return None

        timestamp = str(payload.get("timestamp", datetime.now(tz=timezone.utc).isoformat()))
        day_prefix = timestamp[:10]
        if self._count_clips_today(day_prefix) >= self.max_clips_per_day:
            logger.warning("Cliplimiet voor vandaag bereikt (%s)", self.max_clips_per_day)
            return None

        confidence = float(payload.get("confidence", 0.0))
        is_uncertain = confidence < self.uncertain_threshold
        if is_uncertain and not self.save_uncertain:
            return None

        target_dir = self.uncertain_dir if is_uncertain else self.confirmed_dir
        ts_safe = timestamp.replace(":", "-").replace("+00:00", "Z").replace(".", "-")
        species_slug = self._slug(str(payload.get("species_nl", "unknown")))
        filename = f"{ts_safe}_{species_slug}_conf{confidence:.2f}.wav"
        wav_path = target_dir / filename

        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        sf.write(str(wav_path), samples, sr, subtype="PCM_16")

        metadata = {
            "timestamp": timestamp,
            "filename": str((target_dir.name + "/" + filename)),
            "species_scientific": payload.get("species_scientific", ""),
            "species_nl": payload.get("species_nl", ""),
            "confidence": confidence,
            "tier": int(payload.get("tier", 3)),
            "model_version": payload.get("model_version", ""),
            "duration_s": float(payload.get("duration_s", len(samples) / sr if sr > 0 else 0.0)),
            "rms": float(payload.get("rms", np.sqrt(np.mean(samples.astype(np.float64) ** 2)))),
        }
        with open(self.index_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        return str(wav_path)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """Laad en geef het YAML-configuratiebestand terug."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Species lookup
# ---------------------------------------------------------------------------

def load_species_index(csv_path: str) -> dict[str, dict]:
    """Laad de soorten-CSV in een dict op scientific_name.

    Geeft een lege dict terug als het bestand niet bestaat.
    """
    import csv

    index: dict[str, dict] = {}
    p = Path(csv_path)
    if not p.exists():
        logger.warning("Soorten-CSV niet gevonden: %s", csv_path)
        return index

    with open(p, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            index[row["scientific_name"]] = row
    return index


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------

def build_payload(
    source: str,
    audio: np.ndarray,
    sr: int,
    prediction: dict,
    species_index: dict[str, dict],
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Combineer audio-metadata en classifier-output tot een MQTT-payload."""
    if timestamp is None:
        timestamp = datetime.now(tz=timezone.utc)

    duration_s = len(audio) / sr if sr > 0 else 0.0
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

    species_info = species_index.get(prediction["species_scientific"], {})

    return {
        "timestamp": timestamp.isoformat(),
        "source": source,
        "duration_s": round(duration_s, 3),
        "rms": round(rms, 6),
        "species_scientific": prediction["species_scientific"],
        "species_nl": prediction.get("species_nl", species_info.get("nl_name", "")),
        "species_en": prediction.get("species_en", species_info.get("en_name", "")),
        "confidence": prediction["confidence"],
        "tier": prediction["tier"],
        "emoji": prediction.get("emoji", species_info.get("emoji", "")),
        "model_version": prediction["model_version"],
    }


# ---------------------------------------------------------------------------
# n8n integratie (optioneel)
# ---------------------------------------------------------------------------

def post_to_n8n(payload: dict, webhook_url: str, timeout: int) -> None:
    """POST een detectie-payload naar de n8n webhook."""
    import requests

    resp = requests.post(webhook_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    logger.debug("n8n response %s", resp.status_code)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Verwerk command-line argumenten."""
    parser = argparse.ArgumentParser(
        description="mammal-watcher — zoogdier-detectie via RTSP"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Pad naar config.yaml (standaard: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print payloads naar stdout in plaats van te publiceren",
    )
    parser.add_argument(
        "--no-rtsp",
        action="store_true",
        help="CI-modus: verwerk één synthetisch buffer en sluit af met exitcode 0",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry-point: start de RTSP-consumer en verwerk audio-vensters."""
    args = parse_args()
    cfg = load_config(args.config)

    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    # Bouw classifier
    model_name = cfg.get("classifier", {}).get("model", "stub")
    if model_name == "stub":
        classifier: BaseClassifier = StubClassifier()
    elif model_name == "yamnet":
        try:
            classifier = YAMNetClassifier(
                min_score=cfg.get("classifier", {}).get("yamnet_min_score", 0.1)
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("YAMNet laden mislukt (%s), gebruik stub fallback", exc)
            classifier = StubClassifier()
    else:
        logger.warning("Onbekend model '%s', gebruik stub als fallback", model_name)
        classifier = StubClassifier()

    # Laad soorten-index
    species_csv = cfg.get("species_csv", "./species_mammals_nl.csv")
    species_index = load_species_index(species_csv)

    rtsp_url: str = cfg.get("rtsp", {}).get("url", "rtsp://localhost:8554/mic")
    min_confidence: float = cfg.get("classifier", {}).get("min_confidence", 0.4)
    tier1_threshold: float = cfg.get("classifier", {}).get("tier1_threshold", 0.75)
    tier2_threshold: float = cfg.get("classifier", {}).get("tier2_threshold", 0.5)
    clips_cfg = cfg.get("clips", {})
    clip_saver = ClipSaver(
        clips_dir=clips_cfg.get("clips_dir", "./clips"),
        enabled=clips_cfg.get("enabled", True),
        save_uncertain=clips_cfg.get("save_uncertain", True),
        uncertain_threshold=clips_cfg.get("uncertain_threshold", 0.5),
        max_clips_per_day=clips_cfg.get("max_clips_per_day", 500),
    )

    n8n_cfg = cfg.get("n8n", {})
    n8n_enabled: bool = n8n_cfg.get("enabled", False)
    n8n_webhook_url: str = n8n_cfg.get("webhook_url", "")
    n8n_timeout: int = n8n_cfg.get("timeout_s", 10)

    mqtt_cfg = cfg.get("mqtt", {})
    mqtt_enabled: bool = mqtt_cfg.get("enabled", True) and not args.dry_run

    if args.dry_run:
        logger.info("DRY-RUN modus — geen MQTT/HTTP publicatie")

    # MQTT opstarten
    publisher = None
    if mqtt_enabled:
        from mqtt_publisher import MQTTPublisher

        publisher = MQTTPublisher(
            broker=mqtt_cfg.get("broker", "homeassistant"),
            port=mqtt_cfg.get("port", 1883),
            username=mqtt_cfg.get("username"),
            password=mqtt_cfg.get("password"),
            topic_detections=mqtt_cfg.get("topic_detections", "mammal/detection"),
            topic_status=mqtt_cfg.get("topic_status", "mammal/status"),
            ha_discovery=mqtt_cfg.get("ha_discovery", True),
        )
        publisher.connect()

    # Graceful shutdown op SIGTERM (Docker stop)
    _running = True

    def _shutdown(signum: int, frame: object) -> None:  # noqa: ARG001
        nonlocal _running
        logger.info("Signaal %d ontvangen, afsluiten …", signum)
        _running = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    def _process_window(
        audio: np.ndarray, sr: int, ts: datetime
    ) -> None:
        """Classificeer één audio-venster en publiceer indien relevant."""
        try:
            prediction = classifier.classify(audio, sr)
        except Exception as exc:  # noqa: BLE001
            logger.error("Classifier-fout: %s", exc)
            return
        if not prediction:
            return

        confidence = prediction.get("confidence", 0.0)
        if confidence < min_confidence:
            return

        if confidence >= tier1_threshold:
            prediction["tier"] = 1
        elif confidence >= tier2_threshold:
            prediction["tier"] = 2
        else:
            prediction["tier"] = 3

        payload = build_payload(
            rtsp_url, audio, sr, prediction, species_index, timestamp=ts
        )
        clip_saver.save(audio, sr, payload)

        logger.info(
            "Gedetecteerd: %s (%s) conf=%.2f tier=%s",
            payload["species_scientific"],
            payload["species_nl"],
            confidence,
            payload["tier"],
        )

        if args.dry_run:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            if publisher is not None:
                publisher.publish(payload)
            if n8n_enabled and n8n_webhook_url:
                try:
                    post_to_n8n(payload, n8n_webhook_url, n8n_timeout)
                except Exception as exc:  # noqa: BLE001
                    logger.error("n8n POST mislukt: %s", exc)

    try:
        if args.no_rtsp:
            # CI-modus: verwerk één synthetisch buffer en sluit af
            logger.info("--no-rtsp: synthetisch buffer verwerken")
            rng = np.random.default_rng(42)
            audio = rng.random(5 * 48000).astype(np.float32) * 0.1
            ts = datetime.now(tz=timezone.utc)
            _process_window(audio, 48000, ts)
            logger.info("--no-rtsp modus: klaar, exitcode 0")
            return

        # Normale RTSP-modus
        from rtsp_consumer import RTSPConsumer

        audio_cfg = cfg.get("audio", {})
        rtsp_cfg = cfg.get("rtsp", {})

        consumer = RTSPConsumer(
            url=rtsp_url,
            target_sr=audio_cfg.get("target_sample_rate", 48000),
            window_seconds=audio_cfg.get("window_seconds", 5.0),
            hop_seconds=audio_cfg.get("hop_seconds", 5.0),
            reconnect_initial_s=rtsp_cfg.get("reconnect_initial_s", 1.0),
            reconnect_max_s=rtsp_cfg.get("reconnect_max_s", 30.0),
        )

        try:
            for audio, sr, ts in consumer.iter_windows():
                if not _running:
                    break
                _process_window(audio, sr, ts)
        finally:
            consumer.stop()

    finally:
        if publisher is not None:
            publisher.disconnect()
        logger.info("mammal-watcher gestopt")


if __name__ == "__main__":
    main()
