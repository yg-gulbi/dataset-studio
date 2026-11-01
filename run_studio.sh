#!/usr/bin/env bash
# ==============================================================================
# Universal Dataset Studio & Polygon Lab Launcher
# ==============================================================================
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

PORT=${PORT:-8765}
HOST=${HOST:-"0.0.0.0"}

PROJECT_ROOT="$DIR"
export WORKSPACE_ROOT="$DIR"
export PYTHONPATH="${DIR}:${PYTHONPATH}"

echo "=================================================================="
echo "  🚀 Universal Dataset Studio & Polygon Lab Starting..."
echo "  📂 Workspace: ${PROJECT_ROOT}"
echo "  🌐 URL: http://localhost:${PORT}"
echo "=================================================================="

# Check if port is already in use, kill previous instance if it was our server
EXISTING_PID=$(lsof -ti :${PORT} || true)
if [ -n "$EXISTING_PID" ]; then
  echo "⚠️  Port ${PORT} is occupied by PID ${EXISTING_PID}. Terminating..."
  kill -9 $EXISTING_PID 2>/dev/null || true
  sleep 1
fi

# Automatically open default browser after server binds
(
  sleep 1.2
  if which xdg-open >/dev/null 2>&1; then
    xdg-open "http://localhost:${PORT}" >/dev/null 2>&1 || true
  elif which google-chrome >/dev/null 2>&1; then
    google-chrome "http://localhost:${PORT}" >/dev/null 2>&1 || true
  elif which firefox >/dev/null 2>&1; then
    firefox "http://localhost:${PORT}" >/dev/null 2>&1 || true
  fi
) &

exec python3 server.py

