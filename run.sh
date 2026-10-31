#!/usr/bin/env bash
# AeroCast AI - one command: install, run the leakage-safe pipeline, launch the dashboard.
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Installing dependencies"
python3 -m pip install -r requirements.txt

echo "==> Running pipeline (ingest -> enrich -> features -> train -> evaluate)"
python3 -m aerocast.pipeline

echo "==> Launching dashboard (Ctrl+C to stop)"
python3 -m streamlit run app.py
