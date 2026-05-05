"""
mammal_watcher.py — Hoofd-entry-point voor de mammal-watcher service.

Bewaakt een map op nieuwe audio-snippets, laat ze door de classifier lopen
en stuurt detecties door naar n8n via een webhook.

Gebruik:
    python mammal_watcher.py
    python mammal_watcher.py --dry-run   # print payloads, POST niks

Graceful shutdown via SIGTERM (Docker-veilig).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests
import soundfile as sf
import yaml
from watchdog.events import FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from classifier import BaseClassifier, StubClassifier


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(level: str) -> None:
    """Configure root logger with timestamp format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """Load and return the YAML configuration file."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Species lookup
# ---------------------------------------------------------------------------

def load_species_index(csv_path: str) -> dict[str, dict]:
    """Load species CSV into a dict keyed by scientific_name.

    Returns an empty dict if the file does not exist.
    """
    index: dict[str, dict] = {}
    p = Path(csv_path)
    if not p.exists():
        logger.warning("Species CSV not found: %s", csv_path)
        return index

    import csv
    with open(p, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            index[row["scientific_name"]] = row
    return index


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------

def build_payload(
    audio_path: str,
    audio: np.ndarray,
    sr: int,
    prediction: dict,
    species_index: dict[str, dict],
) -> dict[str, Any]:
    """Combine audio metadata and classifier output into an n8n payload."""
    duration_s = len(audio) / sr if sr > 0 else 0.0
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

    species_info = species_index.get(prediction["species_scientific"], {})

    return {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "audio_path": audio_path,
        "duration_s": round(duration_s, 3),
        "rms": round(rms, 6),
        "species_scientific": prediction["species_scientific"],
        "species_nl": prediction.get("species_nl", species_info.get("nl_name", "")),
        "confidence": prediction["confidence"],
        "tier": prediction["tier"],
        "model_version": prediction["model_version"],
    }


# ---------------------------------------------------------------------------
# n8n integration
# ---------------------------------------------------------------------------

def post_to_n8n(payload: dict, webhook_url: str, timeout: int) -> None:
    """POST a detection payload to the n8n webhook."""
    resp = requests.post(webhook_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    logger.debug("n8n response %s", resp.status_code)


# ---------------------------------------------------------------------------
# File event handler
# ---------------------------------------------------------------------------

class AudioEventHandler(FileSystemEventHandler):
    """Handle new audio files dropped into the watch directory."""

    def __init__(
        self,
        cfg: dict,
        classifier: BaseClassifier,
        species_index: dict[str, dict],
        dry_run: bool,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._classifier = classifier
        self._species_index = species_index
        self._dry_run = dry_run
        self._extensions: list[str] = cfg["watch"].get(
            "file_extensions", [".wav", ".flac"]
        )
        self._min_confidence: float = cfg["classifier"].get("min_confidence", 0.4)
        self._tier1_threshold: float = cfg["classifier"].get("tier1_threshold", 0.75)
        self._tier2_threshold: float = cfg["classifier"].get("tier2_threshold", 0.5)
        self._webhook_url: str = cfg["n8n"]["webhook_url"]
        self._timeout: int = cfg["n8n"].get("timeout_s", 10)

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        """Process a newly created file if it is an audio file."""
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in self._extensions:
            return

        logger.info("New file: %s", path)
        self._process(str(path))

    def _process(self, audio_path: str) -> None:
        """Load audio, classify, and forward to n8n if confidence is sufficient."""
        try:
            audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        except Exception as exc:  # noqa: BLE001
            logger.error("Cannot read %s: %s", audio_path, exc)
            return

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        try:
            prediction = self._classifier.classify(audio, sr)
        except Exception as exc:  # noqa: BLE001
            logger.error("Classifier error for %s: %s", audio_path, exc)
            return

        confidence = prediction.get("confidence", 0.0)
        if confidence < self._min_confidence:
            logger.debug(
                "Below threshold (%.2f < %.2f): %s",
                confidence,
                self._min_confidence,
                audio_path,
            )
            return

        # Override tier based on confidence thresholds from config.
        # This allows fine-tuning alert sensitivity without changing the classifier.
        if confidence >= self._tier1_threshold:
            prediction["tier"] = 1
        elif confidence >= self._tier2_threshold:
            prediction["tier"] = 2
        else:
            prediction["tier"] = 3

        payload = build_payload(audio_path, audio, sr, prediction, self._species_index)

        tier = payload.get("tier", "-")
        logger.info(
            "Detected: %s (%s) conf=%.2f tier=%s",
            payload["species_scientific"],
            payload["species_nl"],
            confidence,
            tier,
        )

        if self._dry_run:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            try:
                post_to_n8n(payload, self._webhook_url, self._timeout)
            except requests.RequestException as exc:
                logger.error("n8n POST failed: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="mammal-watcher — zoogdier-detectie naast BirdNET-Pi"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Pad naar config.yaml (standaard: config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print payloads naar stdout in plaats van naar n8n te posten",
    )
    return parser.parse_args()


def main() -> None:
    """Entry-point: start the file watcher and block until SIGTERM/SIGINT."""
    args = parse_args()
    cfg = load_config(args.config)

    _setup_logging(cfg.get("logging", {}).get("level", "INFO"))

    # Build classifier
    model_name = cfg.get("classifier", {}).get("model", "stub")
    if model_name == "stub":
        classifier: BaseClassifier = StubClassifier()
    else:
        logger.warning("Unknown model '%s', falling back to stub", model_name)
        classifier = StubClassifier()

    # Load species index
    species_csv = cfg.get("species_csv", "./species_mammals_nl.csv")
    species_index = load_species_index(species_csv)

    snippet_dir = cfg["watch"]["snippet_dir"]

    if args.dry_run:
        logger.info("DRY-RUN mode — geen POSTs naar n8n")

    logger.info("Watching %s", snippet_dir)

    # Ensure the watch directory exists.  If it doesn't (e.g. during testing),
    # try to create it; if that's not possible, exit with a clear message.
    try:
        os.makedirs(snippet_dir, exist_ok=True)
    except OSError:
        pass  # Directory may exist but be read-only (mounted volume); that's fine.

    if not os.path.isdir(snippet_dir):
        logger.error(
            "Watch directory does not exist and cannot be created: %s",
            snippet_dir,
        )
        sys.exit(1)

    handler = AudioEventHandler(cfg, classifier, species_index, dry_run=args.dry_run)
    observer = Observer()
    observer.schedule(handler, snippet_dir, recursive=False)
    observer.start()

    # Graceful shutdown on SIGTERM (Docker stop)
    def _shutdown(signum: int, frame: object) -> None:  # noqa: ARG001
        logger.info("Received signal %d, shutting down …", signum)
        observer.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        while observer.is_alive():
            time.sleep(1)
    finally:
        observer.join()
        logger.info("mammal-watcher gestopt")


if __name__ == "__main__":
    main()
