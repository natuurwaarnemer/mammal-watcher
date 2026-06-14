"""
rtsp_consumer.py — RTSP audio consumer voor mammal-watcher.

Leest een live RTSP-stroom (bijv. via MediaMTX relay) met PyAV,
resamples naar 48 kHz mono, en levert 5-seconden vensters.

Fallback: als av niet beschikbaar is, wordt ffmpeg als subprocess gebruikt.
"""

from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime, timezone
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)


class RTSPConsumer:
    """Lees een RTSP-stream en lever overlappende audio-vensters.

    Parameters
    ----------
    url:
        RTSP-URL, bijv. ``rtsp://localhost:8554/mic``.
    target_sr:
        Doel-samplerate in Hz (standaard 48000).
    window_seconds:
        Lengte van elk venster in seconden (standaard 5.0).
    hop_seconds:
        Stapgrootte tussen vensters in seconden (standaard 5.0).
    reconnect_initial_s:
        Eerste wachttijd bij herverbinding in seconden.
    reconnect_max_s:
        Maximale wachttijd bij herverbinding in seconden.
    """

    def __init__(
        self,
        url: str,
        target_sr: int = 48000,
        window_seconds: float = 5.0,
        hop_seconds: float = 5.0,
        reconnect_initial_s: float = 1.0,
        reconnect_max_s: float = 30.0,
    ) -> None:
        self._url = url
        self._target_sr = target_sr
        self._window_size = int(target_sr * window_seconds)
        self._hop_size = int(target_sr * hop_seconds)
        self._reconnect_initial_s = reconnect_initial_s
        self._reconnect_max_s = reconnect_max_s
        self._stop = False

    def stop(self) -> None:
        """Signaleer de consumer om te stoppen."""
        self._stop = True

    def iter_windows(self) -> Iterator[tuple[np.ndarray, int, datetime]]:
        """Levert ``(audio, sample_rate, timestamp)`` vensters uit de RTSP-stroom.

        Probeert eerst PyAV; valt terug op ffmpeg-subprocess als av niet
        beschikbaar is.
        """
        try:
            import av  # noqa: F401
            yield from self._iter_windows_av()
        except ImportError:
            logger.warning("PyAV niet beschikbaar, gebruik ffmpeg subprocess als fallback")
            yield from self._iter_windows_ffmpeg()

    # ------------------------------------------------------------------
    # PyAV-implementatie
    # ------------------------------------------------------------------

    def _iter_windows_av(self) -> Iterator[tuple[np.ndarray, int, datetime]]:
        """Levert vensters via PyAV."""
        import av

        buffer: np.ndarray = np.empty(0, dtype=np.float32)
        delay = self._reconnect_initial_s

        while not self._stop:
            container = None
            try:
                logger.info("Verbinding maken met RTSP-stroom: %s", self._url)
                container = av.open(
                    self._url,
                    options={"rtsp_transport": "tcp", "timeout": "10000000"},
                )
                delay = self._reconnect_initial_s  # reset backoff bij succes
                logger.info("Verbonden met RTSP-stroom")

                for frame in container.decode(audio=0):
                    if self._stop:
                        break

                    samples = frame.to_ndarray()
                    # shape kan (channels, samples) of (samples,) zijn
                    if samples.ndim > 1:
                        samples = samples.mean(axis=0)
                    samples = samples.astype(np.float32)

                    # Resample als samplerate afwijkt van doelrate
                    source_sr = frame.sample_rate
                    if source_sr and source_sr != self._target_sr:
                        new_len = max(1, int(len(samples) * self._target_sr / source_sr))
                        samples = np.interp(
                            np.linspace(0, len(samples) - 1, new_len),
                            np.arange(len(samples)),
                            samples,
                        ).astype(np.float32)

                    buffer = np.concatenate([buffer, samples])

                    while len(buffer) >= self._window_size:
                        window = buffer[: self._window_size]
                        buffer = buffer[self._hop_size :]
                        ts = datetime.now(tz=timezone.utc)
                        yield window, self._target_sr, ts

                    logger.debug(
                        "Frame: %d samples, buffer: %d", len(samples), len(buffer)
                    )

            except Exception as exc:  # noqa: BLE001
                if self._stop:
                    break
                logger.info(
                    "RTSP-fout: %s — herverbinden over %.1fs", exc, delay
                )
                time.sleep(delay)
                delay = min(delay * 2, self._reconnect_max_s)
            finally:
                if container is not None:
                    try:
                        container.close()
                    except Exception:  # noqa: BLE001
                        pass

    # ------------------------------------------------------------------
    # ffmpeg-subprocess fallback
    # ------------------------------------------------------------------

    def _iter_windows_ffmpeg(self) -> Iterator[tuple[np.ndarray, int, datetime]]:
        """Levert vensters via ffmpeg subprocess (fallback)."""
        SAMPLE_BYTES = 2  # s16le = 2 bytes per sample
        buffer: np.ndarray = np.empty(0, dtype=np.float32)
        delay = self._reconnect_initial_s
        chunk_samples = self._target_sr  # 1 seconde per lees-call

        while not self._stop:
            proc = None
            try:
                logger.info(
                    "Verbinding maken via ffmpeg subprocess: %s", self._url
                )
                cmd = [
                    "ffmpeg",
                    "-rtsp_transport", "tcp",
                    "-i", self._url,
                    "-f", "s16le",
                    "-ar", str(self._target_sr),
                    "-ac", "1",
                    "-",
                ]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                delay = self._reconnect_initial_s
                logger.info("ffmpeg process gestart (PID %d)", proc.pid)

                while not self._stop:
                    raw = proc.stdout.read(chunk_samples * SAMPLE_BYTES)  # type: ignore[union-attr]
                    if not raw:
                        break
                    samples = (
                        np.frombuffer(raw, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    buffer = np.concatenate([buffer, samples])

                    while len(buffer) >= self._window_size:
                        window = buffer[: self._window_size]
                        buffer = buffer[self._hop_size :]
                        ts = datetime.now(tz=timezone.utc)
                        yield window, self._target_sr, ts

            except Exception as exc:  # noqa: BLE001
                if self._stop:
                    break
                logger.info(
                    "ffmpeg-fout: %s — herverbinden over %.1fs", exc, delay
                )
                time.sleep(delay)
                delay = min(delay * 2, self._reconnect_max_s)
            finally:
                if proc is not None:
                    try:
                        proc.terminate()
                        proc.wait(timeout=5)
                    except Exception:  # noqa: BLE001
                        pass
