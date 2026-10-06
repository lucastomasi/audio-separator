#!/usr/bin/env python3
"""Native window for Audio Separator. Closing the window stops the app."""
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

HOST = "127.0.0.1"
TOKEN_ENV = "AUDIO_SEPARATOR_TOKEN"
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


def pick_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def start_server(port):
    from app import build_server, launch_kwargs

    demo = build_server()
    demo.launch(
        **launch_kwargs(
            prevent_thread_lock=True,
            inbrowser=False,
            server_name=HOST,
            server_port=port,
        )
    )


def our_server_ready(url, token, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        request = urllib.request.Request(
            url,
            headers={"Host": HOST, "x-audio-separator-token": token},
        )
        try:
            with urllib.request.urlopen(request, timeout=0.4) as response:
                if 200 <= response.status < 400:
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                return False
            time.sleep(0.4)
        except (OSError, urllib.error.URLError):
            time.sleep(0.4)
    return False


def attach_when_ready(window, url, token):
    if not our_server_ready(url, token):
        window.load_html("No se pudo arrancar Audio Separator.")
        return
    window.load_url(url)


def _report_launch_failure():
    log_dir = os.path.expanduser("~/Library/Logs/Audio Separator")
    try:
        os.makedirs(log_dir, exist_ok=True)
        os.chmod(log_dir, 0o700)
        log_path = os.path.join(log_dir, "launch.log")
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write("\n")
            handle.write(traceback.format_exc())
        os.chmod(log_path, 0o600)
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


def _offline_env():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def main():
    import webview

    _offline_env()
    token = secrets.token_urlsafe(32)
    os.environ[TOKEN_ENV] = token
    port = pick_free_port()
    url = f"http://{HOST}:{port}/?access_token={token}"
    threading.Thread(target=start_server, args=(port,), daemon=True).start()
    window = webview.create_window(
        "Audio Separator",
        None,
        html=SPLASH,
        width=1100,
        height=820,
        min_size=(720, 560),
        text_select=True,
    )
    webview.start(lambda opened: attach_when_ready(opened, url, token), window)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _report_launch_failure()
        raise SystemExit(1)
