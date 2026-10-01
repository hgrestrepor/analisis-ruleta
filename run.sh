#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PORT="${1:-8080}"

echo "Analizador de Ruleta  ->  http://localhost:${PORT}"
exec .venv/bin/python -m uvicorn src.app:app --host 127.0.0.1 --port "$PORT" --no-access-log
