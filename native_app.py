#!/usr/bin/env python3
"""
Dataset Studio - Native Desktop Application Window
Powered by GTK 3 & WebKitGTK (No external Chrome/Browser dependency).
Runs as a fully independent desktop application with dedicated dock icon.
"""

import sys
import os
import time
import signal
import warnings
import subprocess
import urllib.request
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set GTK application ID & Program name BEFORE importing Gtk
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('WebKit2', '4.1')
from gi.repository import Gtk, WebKit2, GLib, GdkPixbuf

APP_ID = "dataset-studio"
APP_NAME = "Dataset Studio"
PORT = int(os.environ.get("PORT", "8765"))
HOST = "127.0.0.1"
URL = f"http://{HOST}:{PORT}"

DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DIR
ICON_PATH = DIR / "static" / "app_icon.png"

GLib.set_prgname(APP_ID)
GLib.set_application_name(APP_NAME)


def is_server_running():
    try:
        req = urllib.request.urlopen(f"{URL}/api/dataset/current", timeout=0.8)
        return req.getcode() == 200
    except Exception:
        return False


def start_backend_server():
    if is_server_running():
        return None

    env = os.environ.copy()
    env.setdefault("WORKSPACE_ROOT", str(PROJECT_ROOT))
    env["PYTHONPATH"] = f"{DIR}:{env.get('PYTHONPATH', '')}"

    server_script = DIR / "server.py"
    data_dir = Path(env.get("DATASET_STUDIO_DATA_DIR", str(PROJECT_ROOT)))
    data_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(data_dir / "studio_server.log", "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(server_script)],
        env=env,
        stdout=log_file,
        stderr=subprocess.STREQUAL if hasattr(subprocess, "STREQUAL") else log_file,
        start_new_session=True
    )

    # Wait up to 6 seconds for backend readiness
    for _ in range(30):
        time.sleep(0.2)
        if is_server_running():
            break

    return proc


class DatasetStudioApp(Gtk.Window):
    def __init__(self, backend_proc=None):
        super().__init__(title=f"{APP_NAME} & Polygon Lab")
        self.backend_proc = backend_proc

        self.set_wmclass(APP_ID, "DatasetStudio")
        self.set_default_size(1600, 950)
        self.set_position(Gtk.WindowPosition.CENTER)

        # Set Window Icon
        if ICON_PATH.exists():
            try:
                pixbuf = GdkPixbuf.Pixbuf.new_from_file(str(ICON_PATH))
                self.set_icon(pixbuf)
                Gtk.Window.set_default_icon(pixbuf)
            except Exception as e:
                print(f"[Warning] Failed to load icon: {e}", file=sys.stderr)

        # WebKit WebView Configuration
        self.webview = WebKit2.WebView()
        settings = self.webview.get_settings()
        settings.set_enable_webgl(True)
        settings.set_enable_accelerated_2d_canvas(True)
        settings.set_enable_developer_extras(True)
        settings.set_enable_smooth_scrolling(True)

        # Scrolled window container
        scrolled = Gtk.ScrolledWindow()
        scrolled.add(self.webview)
        self.add(scrolled)

        # Load application URL
        self.webview.load_uri(URL)

        # Connect window close handler
        self.connect("destroy", self.on_destroy)

    def on_destroy(self, widget):
        if self.backend_proc and self.backend_proc.poll() is None:
            try:
                self.backend_proc.terminate()
            except Exception:
                pass
        Gtk.main_quit()


def main():
    backend_proc = start_backend_server()
    app = DatasetStudioApp(backend_proc=backend_proc)
    app.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
