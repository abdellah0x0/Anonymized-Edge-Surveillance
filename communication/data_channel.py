"""
communication/data_channel.py
Gestion centralisée des DataChannels WebRTC.

Responsabilités:
  - Enregistrer / désenregistrer les canaux actifs
  - Broadcaster des alertes (étape 4) à tous les clients connectés
  - Envoyer des changements de mode confirmés
"""

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class DataChannelManager:
    """
    Singleton gérant l'ensemble des DataChannels ouverts.
    Thread-safe via asyncio (pas de threads OS).
    """

    def __init__(self):
        # {pc_id: channel_object}
        self._channels: dict[str, Any] = {}
        # Compteur d'alertes pour dédupliquer (même seconde)
        self._last_alert_time: float = 0.0
        self._alert_cooldown: float = 2.0  # secondes entre deux alertes identiques

    # ── Gestion des canaux ────────────────────────────────────────────────────

    def register(self, pc_id: str, channel) -> None:
        self._channels[pc_id] = channel
        logger.info(f"[DataChannel] Canal enregistré pour {pc_id} (total: {len(self._channels)})")

    def unregister(self, pc_id: str) -> None:
        self._channels.pop(pc_id, None)
        logger.info(f"[DataChannel] Canal supprimé pour {pc_id} (total: {len(self._channels)})")

    # ── Envoi de messages ────────────────────────────────────────────────────

    def send_to_all(self, message: str) -> None:
        """Envoie un message à tous les canaux ouverts (synchrone)."""
        dead = []
        for pc_id, ch in self._channels.items():
            try:
                ch.send(message)
            except Exception as e:
                logger.warning(f"[DataChannel] Échec envoi à {pc_id}: {e}")
                dead.append(pc_id)
        for pc_id in dead:
            self.unregister(pc_id)

    async def async_send_to_all(self, message: str) -> None:
        """Envoie asynchrone (à utiliser depuis une coroutine)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.send_to_all, message)

    # ── Alertes visage inconnu (étape 4) ─────────────────────────────────────

    def broadcast_alerts(self, alerts: list[str]) -> None:
        """
        Diffuse les alertes générées par apply_selective_anonymization.
        Applique un cooldown pour éviter le spam.
        """
        if not alerts:
            return

        now = time.time()
        if now - self._last_alert_time < self._alert_cooldown:
            return  # trop tôt, on skip

        self._last_alert_time = now
        for alert in alerts:
            logger.info(f"[DataChannel] Alerte: {alert}")
            self.send_to_all(alert)

    # ── Changement de mode (étape 5) ─────────────────────────────────────────

    def send_mode_change(self, new_mode: str) -> None:
        """Notifie le navigateur du changement de mode."""
        msg = f"MODE_CHANGED|{new_mode}"
        self.send_to_all(msg)
        logger.info(f"[DataChannel] Mode changé → {new_mode}")

    # ── Handler entrant (côté serveur) ───────────────────────────────────────

    def handle_incoming(self, message: str, channel, pc_id: str) -> None:
        """
        Réagit aux messages reçus du navigateur.
        Protocole simple:
          - "Hello from Browser!"       → salutation
          - "SET_MODE|<mode>"            → demande de changement de mode
          - "PING"                       → réponse PONG
        """
        logger.debug(f"[DataChannel] ← {pc_id}: {message}")

        if message == "Hello from Browser!":
            channel.send("Hello from Python WebRTC Server!")

        elif message.startswith("SET_MODE|"):
            mode = message.split("|", 1)[1].strip()
            channel.send(f"MODE_ACK|{mode}")
            logger.info(f"[DataChannel] Changement de mode demandé: {mode}")
            # Le mode est géré par webrtc_server.py via current_mode

        elif message == "PING":
            channel.send("PONG")

        else:
            channel.send(f"Server Acknowledges: {message}")


# Singleton global
_manager_instance: DataChannelManager | None = None


def get_manager() -> DataChannelManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = DataChannelManager()
    return _manager_instance