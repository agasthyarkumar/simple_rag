#!/usr/bin/env bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "==> Installing backend dependencies..."
pip install -r "$ROOT/backend/requirements.txt"

echo "==> Installing frontend dependencies..."
(cd "$ROOT/frontend" && npm install)

echo ""
echo "==> Starting backend  → http://localhost:8000"
echo "==> Starting frontend → http://localhost:5173"
echo ""

# Run both in background; trap Ctrl-C to kill both
cleanup() { kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null; exit; }
trap cleanup INT TERM

(cd "$ROOT/backend" && uvicorn main:app --reload --host 0.0.0.0 --port 8000) &
BACKEND_PID=$!

(cd "$ROOT/frontend" && npm run dev -- --host 0.0.0.0) &
FRONTEND_PID=$!

wait
