"""
Lance le serveur WebRTC (aiohttp) pour les tests end-to-end.
Usage:
    py -3 tools/run_webrtc_server.py --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aiohttp import web

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from streaming.webrtc_server import create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Run WebRTC server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    app = create_app()
    web.run_app(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
