#!/usr/bin/env python3
"""Native window for Audio Separator. Closing the window stops the app."""
import socket
import sys
import threading
import time

import webview

from app_env import host as env_host, pick_port

HOST = env_host()
PORT = pick_port()
URL = f"http://{HOST}:{PORT}"
SPLASH = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Audio Separator</title>
  <style>
    body {
      margin: 0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Helvetica Neue", sans-serif;
      background: #F2F2F7;
      color: #1D1D1F;
      -webkit-font-smoothing: antialiased;
    }
    .card { text-align: center; padding: 2rem; }
    h1 { font-size: 1.45rem; margin: 0 0 .4rem; font-weight: 700; letter-spacing: -0.03em; }
    p { margin: 0; color: #86868B; font-size: 0.9rem; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Audio Separator</h1>
    <p>Abriendo la interfaz…</p>
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


BLOCK_DOWNLOAD_JS = """
(function () {
  function block(e) {
    var t = e.target;
    if (!t || !t.closest) return;
    var a = t.closest('a[download], a[href^="blob:"], a[href^="data:"], button[aria-label*="Download" i], button[aria-label*="Descargar" i], button[title*="Download" i]');
    if (a) {
      e.preventDefault();
      e.stopPropagation();
      return false;
    }
  }
  document.addEventListener('click', block, true);
})();
"""


def inject_download_guard(window):
    try:
        window.evaluate_js(BLOCK_DOWNLOAD_JS)
    except Exception:
        pass


def attach_when_ready(window):
    if not wait_until_ready():
        window.load_html(
            SPLASH.replace(
                "Arrancando… el primer inicio puede tardar uno o dos minutos.",
                "No se pudo arrancar Audio Separator.",
            )
        )
        return
    window.load_url(URL)


def main():
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

    def on_loaded():
        inject_download_guard(window)

    def on_closed():
        try:
            from vc_runner import stop_vc_worker

            stop_vc_worker()
        except Exception:
            pass

    try:
        window.events.loaded += on_loaded
    except Exception:
        pass
    try:
        window.events.closed += on_closed
    except Exception:
        pass

    webview.start(None if already else attach_when_ready, None if already else window)


if __name__ == "__main__":
    main()
