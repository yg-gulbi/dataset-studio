#!/usr/bin/env bash
# ==============================================================================
# Universal Dataset Studio - Desktop Application Launcher
# Usage:
#   ./launch_dataset_studio.sh                   (Opens with default / last folder)
#   ./launch_dataset_studio.sh /path/to/dataset  (Opens specific dataset directly)
# ==============================================================================
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PORT=${PORT:-8765}
HOST="127.0.0.1"
TARGET_DIR="$1"

# 1. Check if port is already running
PID=$(lsof -ti :${PORT} 2>/dev/null || true)
if [ -z "$PID" ]; then
  export WORKSPACE_ROOT="${DIR}"
  export PYTHONPATH="${DIR}:${PYTHONPATH}"
  nohup setsid python3 "${DIR}/server.py" > "${DIR}/studio_server.log" 2>&1 &

  # Wait for server ready (up to 5s)
  for i in {1..25}; do
    if curl -s "http://${HOST}:${PORT}/api/dataset/current" >/dev/null 2>&1; then
      break
    fi
    sleep 0.2
  done
fi

# 2. If target directory was provided as argument, switch to it immediately
if [ -n "$TARGET_DIR" ] && [ -d "$TARGET_DIR" ]; then
  ABS_TARGET=$(realpath "$TARGET_DIR")
  curl -s -X POST "http://${HOST}:${PORT}/api/open_folder" \
    -H "Content-Type: application/json" \
    -d "{\"path\": \"${ABS_TARGET}\", \"init_if_raw\": false}" >/dev/null 2>&1 || true
fi

# 3. Launch Desktop Application (Native GTK3+WebKitGTK App Window)
# Falls back to Chrome App Mode only if GTK environment is unavailable
NATIVE_APP="${DIR}/native_app.py"

if [ -f "$NATIVE_APP" ] && python3 -c "import gi; gi.require_version('Gtk', '3.0'); gi.require_version('WebKit2', '4.1')" >/dev/null 2>&1; then
  exec python3 "$NATIVE_APP"
elif command -v google-chrome >/dev/null 2>&1; then
  APP_URL="http://${HOST}:${PORT}"
  exec google-chrome --app="${APP_URL}" --window-size=1600,950
elif command -v chromium-browser >/dev/null 2>&1; then
  APP_URL="http://${HOST}:${PORT}"
  exec chromium-browser --app="${APP_URL}" --window-size=1600,950
elif command -v xdg-open >/dev/null 2>&1; then
  APP_URL="http://${HOST}:${PORT}"
  exec xdg-open "${APP_URL}"
else
  echo "앱 주소: http://${HOST}:${PORT}"
fi
