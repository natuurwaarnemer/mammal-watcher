"""
mqtt_publisher.py — MQTT-uitgever voor mammal-watcher detecties.

Gebruikt paho-mqtt v2 sync client met automatische herverbinding.
Publiceert Home Assistant MQTT-discovery config bij verbinding.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

__version__ = "0.2.0"


class MQTTPublisher:
    """Publiceert detecties naar een MQTT-broker.

    Parameters
    ----------
    broker:
        Hostnaam of IP-adres van de MQTT-broker.
    port:
        TCP-poort van de broker (standaard 1883).
    username:
        Optionele gebruikersnaam voor authenticatie.
    password:
        Optioneel wachtwoord voor authenticatie.
    topic_detections:
        MQTT-topic voor detectie-payloads.
    topic_status:
        MQTT-topic voor service-status (LWT).
    ha_discovery:
        Publiceert HA MQTT-discovery config bij verbinding.
    """

    def __init__(
        self,
        broker: str,
        port: int = 1883,
        username: str | None = None,
        password: str | None = None,
        topic_detections: str = "mammal/detection",
        topic_status: str = "mammal/status",
        ha_discovery: bool = True,
    ) -> None:
        self._broker = broker
        self._port = port
        self._username = username
        self._password = password
        self._topic_detections = topic_detections
        self._topic_status = topic_status
        self._ha_discovery = ha_discovery
        self._client: Any = None
        self._connected = False

    def connect(self) -> None:
        """Maak verbinding met de MQTT-broker."""
        import paho.mqtt.client as mqtt

        client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="mammal-watcher",
        )

        if self._username:
            client.username_pw_set(self._username, self._password)

        # Last Will and Testament: stuur 'offline' als de verbinding wegvalt
        client.will_set(self._topic_status, "offline", qos=1, retain=True)

        def on_connect(
            client: Any,
            userdata: Any,
            flags: Any,
            reason_code: Any,
            properties: Any,
        ) -> None:
            if reason_code == 0:
                logger.info(
                    "MQTT verbonden met %s:%d", self._broker, self._port
                )
                self._connected = True
                client.publish(self._topic_status, "online", qos=1, retain=True)
                if self._ha_discovery:
                    self._publish_ha_discovery(client)
            else:
                logger.warning("MQTT verbinding mislukt: %s", reason_code)
                self._connected = False

        def on_disconnect(
            client: Any,
            userdata: Any,
            disconnect_flags: Any,
            reason_code: Any,
            properties: Any,
        ) -> None:
            logger.info("MQTT verbroken: %s", reason_code)
            self._connected = False

        client.on_connect = on_connect
        client.on_disconnect = on_disconnect

        client.connect(self._broker, self._port, keepalive=60)
        client.loop_start()
        self._client = client

    def _publish_ha_discovery(self, client: Any) -> None:
        """Publiceert Home Assistant MQTT-discovery configuratie."""
        discovery_topic = (
            "homeassistant/sensor/mammal_watcher_last_detection/config"
        )
        config = {
            "name": "Mammal Watcher Last Detection",
            "state_topic": self._topic_detections,
            "value_template": "{{ value_json.species_nl }}",
            "json_attributes_topic": self._topic_detections,
            "unique_id": "mammal_watcher_last_detection",
            "device": {
                "identifiers": ["mammal_watcher"],
                "name": "Mammal Watcher",
                "manufacturer": "natuurwaarnemer",
                "model": "mammal-watcher",
                "sw_version": __version__,
            },
        }
        client.publish(
            discovery_topic, json.dumps(config), qos=1, retain=True
        )
        logger.debug("HA discovery config gepubliceerd")

    def publish(self, payload: dict[str, Any]) -> None:
        """Publiceer een detectie-payload naar het detections-topic."""
        if self._client is None or not self._connected:
            logger.warning("MQTT niet verbonden, publicatie overgeslagen")
            return
        msg = json.dumps(payload, ensure_ascii=False)
        result = self._client.publish(
            self._topic_detections, msg, qos=1, retain=False
        )
        logger.debug("MQTT publicatie resultaat: %s", result.rc)

    def disconnect(self) -> None:
        """Verbreek de verbinding met de MQTT-broker."""
        if self._client is not None:
            try:
                self._client.publish(
                    self._topic_status, "offline", qos=1, retain=True
                )
            except Exception:  # noqa: BLE001
                pass
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None
            self._connected = False
