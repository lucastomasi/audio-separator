#!/usr/bin/env python3
"""Native window for Audio Separator. Closing the window stops the app."""
import os
import secrets
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

from app_env import host as env_host, pick_port
from app_security import TOKEN_ENV, TOKEN_HEADER, TOKEN_QUERY, ensure_token

HOST = env_host()
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
    .bar {
      width: 120px; height: 3px; margin: 18px auto 0; overflow: hidden;
      border-radius: 3px; background: #E5E5EA;
    }
    .bar i {
      display: block; width: 40%; height: 100%; background: #007AFF;
      animation: slide 1.1s ease-in-out infinite;
    }
    @keyframes slide {
      0% { transform: translateX(-120%); }
      100% { transform: translateX(320%); }
    }
    @media (prefers-color-scheme: dark) {
      body { background: #1C1C1E; color: #F5F5F7; }
      p { color: #98989D; }
      .bar { background: #3A3A3C; }
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Audio Separator</h1>
    <p>Arrancando… la primera vez instala lo que falte y puede tardar varios minutos.</p>
    <div class="bar" aria-hidden="true"><i></i></div>
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


def port_open(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.4)
    try:
        sock.connect((HOST, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _attach_logs():
    if sys.stderr.isatty():
        return
    path = os.path.expanduser("~/Library/Logs/Audio Separator/launch.log")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = handle
    sys.stderr = handle


def _offline_env():
    """Bundled weights only. Do not reach Hugging Face at runtime."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def start_server(port):
    try:
        from app import launch_app

        launch_app(
            prevent_thread_lock=True,
            inbrowser=False,
            server_name=HOST,
            server_port=port,
        )
    except Exception:
        traceback.print_exc()


def warmup_vc_worker():
    """Create the conversion venv if needed and start the worker process."""
    try:
        from vc_runner import _get_worker, ensure_vc_engine

        ensure_vc_engine()
        _get_worker()
    except Exception:
        traceback.print_exc()


def our_server_ready(url, token, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        request = urllib.request.Request(
            url,
            headers={"Host": HOST, TOKEN_HEADER: token},
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


def wait_until_ready(port, token, timeout=300):
    url = f"http://{HOST}:{port}/?{TOKEN_QUERY}={token}"
    return our_server_ready(url, token, timeout=timeout)


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


def attach_when_ready(window, url, token):
    if not our_server_ready(url, token):
        window.load_html(
            SPLASH.replace(
                "Arrancando… la primera vez instala lo que falte y puede tardar varios minutos.",
                "No se pudo arrancar Audio Separator.",
            )
        )
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


def resolve_server(token, probe_timeout=1.5):
    """Pick a port we own. Never attach to a foreign local listener."""
    from app_env import pick_port

    for _ in range(32):
        port = pick_port()
        url = f"http://{HOST}:{port}/?{TOKEN_QUERY}={token}"
        if not port_open(port):
            threading.Thread(target=start_server, args=(port,), daemon=True).start()
            return port, url, False
        # Port answers: only reuse if it accepts our token.
        if our_server_ready(url, token, timeout=probe_timeout):
            return port, url, True
        os.environ["AUDIO_SEPARATOR_PORT"] = str(port + 1)
    raise RuntimeError("No hay puerto local libre para la app.")


def main():
    import webview

    _attach_logs()
    _offline_env()
    token = ensure_token()
    if not os.environ.get(TOKEN_ENV):
        os.environ[TOKEN_ENV] = secrets.token_urlsafe(32)
        token = os.environ[TOKEN_ENV]
    port, url, already = resolve_server(token)
    threading.Thread(target=warmup_vc_worker, daemon=True).start()

    window = webview.create_window(
        "Audio Separator",
        url if already else None,
        html=None if already else SPLASH,
        width=980,
        height=760,
        min_size=(760, 620),
        text_select=False,
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
            from runpod_train import stop_pod

            stop_pod()
        except Exception:
            pass
        os._exit(0)

    try:
        window.events.loaded += on_loaded
    except Exception:
        pass
    try:
        window.events.closed += on_closed
    except Exception:
        pass

    webview.start(
        None if already else (lambda opened: attach_when_ready(opened, url, token)),
        None if already else window,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        _report_launch_failure()
        raise SystemExit(1)
