"""
streaming/webrtc_server.py
Serveur WebRTC intégrant les étapes 4 (flou sélectif) et 5 (heatmap).
"""

import asyncio
import json
import logging
import os
import time
import threading

import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from av import VideoFrame

from anonymization.privacy_modes import process_frame, ALL_MODES
from communication.data_channel import get_manager
from config.settings import (
    DEFAULT_PRIVACY_MODE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)

logger = logging.getLogger(__name__)
ROOT   = os.path.dirname(os.path.dirname(__file__))

# Mode courant partagé entre toutes les connexions
current_mode: str = DEFAULT_PRIVACY_MODE


# ──────────────────────────────────────────────────────────────────────────────
# Track vidéo
# ──────────────────────────────────────────────────────────────────────────────

class PrivacyVideoStreamTrack(VideoStreamTrack):
    """
    Track vidéo qui applique le mode de confidentialité courant à chaque frame.
    Optimisé avec un thread de capture et du frame skipping.
    """
    kind = "video"

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        self.latest_frame = None
        self.running = True
        self.frame_count = 0
        
        # Thread de capture pour vider le buffer OpenCV en temps réel (Point 3)
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = frame
            else:
                time.sleep(0.01)

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()

        while self.latest_frame is None and self.running:
            await asyncio.sleep(0.005)

        if not self.running:
            frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        else:
            frame = self.latest_frame.copy()

        self.frame_count += 1
        timestamp_ms = int(time.time() * 1000)

        # Pipeline avec optimisation (Points 2 et 5 passés via frame_count)
        # process_frame va gérer le saut de frames pour la détection et reconnaissance
        frame_out, alerts = process_frame(frame, timestamp_ms, mode=current_mode, frame_count=self.frame_count)

        if alerts:
            get_manager().broadcast_alerts(alerts)

        video_frame = VideoFrame.from_ndarray(frame_out, format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame

    def stop(self):
        super().stop()
        self.running = False
        if self.cap.isOpened():
            self.cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# Routes HTTP / Signaling
# ──────────────────────────────────────────────────────────────────────────────

async def index(request: web.Request) -> web.Response:
    content = open(os.path.join(ROOT, "web", "index.html"), "r", encoding="utf-8").read()
    return web.Response(content_type="text/html", text=content)


async def offer(request: web.Request) -> web.Response:
    global current_mode

    try:
        params      = await request.json()
        sdp_offer   = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        pc    = RTCPeerConnection()
        pcs   = request.app["pcs"]
        pc_id = f"pc-{id(pc)}"
        pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info(f"[WebRTC] {pc_id} state → {pc.connectionState}")
            if pc.connectionState in ("failed", "closed"):
                get_manager().unregister(pc_id)
                await pc.close()
                pcs.discard(pc)

        # ── DataChannel (étapes 2, 4, 5) ─────────────────────────────────────
        @pc.on("datachannel")
        def on_datachannel(channel):
            logger.info(f"[WebRTC] DataChannel '{channel.label}' ouvert ({pc_id})")
            get_manager().register(pc_id, channel)

            @channel.on("message")
            def on_message(message: str):
                # Déléguer au gestionnaire centralisé
                get_manager().handle_incoming(message, channel, pc_id)

                # Changer le mode si demandé
                if message.startswith("SET_MODE|"):
                    global current_mode
                    new_mode = message.split("|", 1)[1].strip()
                    if new_mode in ALL_MODES:
                        current_mode = new_mode
                        logger.info(f"[WebRTC] Mode changé → {current_mode}")
                        get_manager().send_mode_change(current_mode)
                    else:
                        channel.send(f"ERROR|Mode inconnu: {new_mode}")

            @channel.on("close")
            def on_close():
                get_manager().unregister(pc_id)

        # ── Track vidéo ───────────────────────────────────────────────────────
        pc.addTrack(PrivacyVideoStreamTrack())

        await pc.setRemoteDescription(sdp_offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        return web.Response(
            content_type="application/json",
            text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
        )

    except Exception:
        import traceback
        traceback.print_exc()
        return web.Response(status=500, text="Erreur interne du serveur WebRTC")


async def set_mode(request: web.Request) -> web.Response:
    """
    Route HTTP alternative pour changer de mode (utile pour tests curl).
    POST /mode  { "mode": "selective" }
    """
    global current_mode
    try:
        body     = await request.json()
        new_mode = body.get("mode", "")
        if new_mode not in ALL_MODES:
            return web.Response(
                status=400,
                text=json.dumps({"error": f"Mode invalide. Modes valides: {list(ALL_MODES)}"}),
                content_type="application/json",
            )
        current_mode = new_mode
        get_manager().send_mode_change(current_mode)
        return web.Response(
            content_type="application/json",
            text=json.dumps({"mode": current_mode}),
        )
    except Exception as e:
        return web.Response(status=400, text=str(e))


async def on_shutdown(app: web.Application):
    coros = [pc.close() for pc in app["pcs"]]
    await asyncio.gather(*coros)
    app["pcs"].clear()


def create_app() -> web.Application:
    app = web.Application()
    app["pcs"] = set()
    app.router.add_get("/",       index)
    app.router.add_post("/offer", offer)
    app.router.add_post("/mode",  set_mode)
    app.on_shutdown.append(on_shutdown)
    return app