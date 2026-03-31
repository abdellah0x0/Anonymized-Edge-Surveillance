# Anonymized-Edge-Surveillance

## Prerequisites

- Python `3.10` recommended
- Webcam
- Windows/Linux/macOS with virtual environment support

## Installation

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run

- Notebook workflow:

```bash
jupyter notebook
```

- WebRTC server workflow:

```bash
py -3 tools/run_webrtc_server.py --host 0.0.0.0 --port 8080
```

Then open [http://localhost:8080](http://localhost:8080) in your browser.
