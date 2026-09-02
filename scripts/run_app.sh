#!/usr/bin/env bash
# Launch the Streamlit dashboard. Honors STREAMLIT_SERVER_PORT from .env.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a
exec streamlit run app.py --server.port "${STREAMLIT_SERVER_PORT:-8501}"
