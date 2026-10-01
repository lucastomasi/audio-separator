#!/usr/bin/env python3
"""Native window for Audio Separator. Closing the window stops the app."""
import os
import socket
import subprocess
import sys
import threading
import time
import traceback

URL = "http://127.0.0.1:7860"
HOST = "127.0.0.1"
PORT = 7860
SPLASH = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Audio Separator</title>
</head>
<body>
  <div style="font-family:-apple-system,sans-serif;padding:2rem;text-align:center">
    <h1>Audio Separator</h1>
    <p>Arrancando… el primer inicio puede tardar uno o dos minutos.</p>
  </div>
</body>
</html>
"""


def port_open():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.4)
    try:
        sock.connect((HOST, PORT))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def start_server():
    from app import build_server, launch_kwargs
    demo = build_server()
    demo.launch(**launch_kwargs(prevent_thread_lock=True, inbrowser=False))


def wait_until_ready(timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_open():
            return True
        time.sleep(0.4)
    return False


def attach_when_ready(window):
    if not wait_until_ready():
        window.load_html("No se pudo arrancar Audio Separator.")
        return
    window.load_url(URL)


def _report_launch_failure():
    log_dir = os.path.expanduser("~/Library/Logs/Audio Separator")
    try:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "launch.log"), "a", encoding="utf-8") as handle:
            handle.write("\n")
            handle.write(traceback.format_exc())
    except OSError:
        pass
    if sys.platform != "darwin":
        return
    subprocess.run(
        [
            "osascript",
            "-e",
            'display dialog "Audio Separator no pudo abrir. El detalle está en ~/Library/Logs/Audio Separator/launch.log" buttons {"OK"} default button 1 with title "Audio Separator"',
        ],
        check=False,
    )


def main():
    import webview

    already = port_open()
    if not already:
        threading.Thread(target=start_server, daemon=True).start()
    window = webview.create_window(
        "Audio Separator",
        URL if already else None,
        html=None if already else SPLASH,
        width=1100,
        height=820,
        min_size=(720, 560),
        text_select=True,
    )
    webview.start(None if already else attach_when_ready, None if already else window)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _report_launch_failure()
        raise SystemExit(1)
