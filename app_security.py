"""Local-only Gradio launch: loopback, token, tight allowed paths."""
from __future__ import annotations

import json
import os
import re
import secrets
from urllib.parse import parse_qs, urlparse

from app_env import data_dir, home, host, package_dir

TOKEN_ENV = "AUDIO_SEPARATOR_TOKEN"
TOKEN_COOKIE = "as_token"
TOKEN_QUERY = "access_token"
TOKEN_HEADER = "x-audio-separator-token"
MAX_UPLOAD = "200mb"
MAX_N_FFT = 16384
MAX_DIM_F = 8192
MAX_DIM_T = 2048
MAX_HOP = 4096

_HEAD_TOKEN_JS = """
<script>
(function () {
  var params = new URLSearchParams(window.location.search);
  var token = params.get("access_token");
  if (token) {
    document.cookie = "as_token=" + encodeURIComponent(token) + "; path=/; SameSite=Strict; Max-Age=31536000";
  }
})();
</script>
"""

_PATH_RE = re.compile(r"(?:/Users/|/home/|/var/|/tmp/|[A-Za-z]:\\\\)")


def _token_ok(provided, token):
    if not provided or not token or len(provided) != len(token):
        return False
    return secrets.compare_digest(provided, token)


def _loopback_host(request):
    host = request.headers.get("host") or ""
    if host.startswith("["):
        name = host.rsplit("]", 1)[0].lstrip("[")
    else:
        name = host.split(":")[0]
    return name.lower() in {"127.0.0.1", "localhost", "::1"}


def _token_from_referer(request, token):
    referer = request.headers.get("referer") or ""
    parsed = urlparse(referer)
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return None
    values = parse_qs(parsed.query).get(TOKEN_QUERY) or []
    if not values:
        return None
    return values[0] if _token_ok(values[0], token) else None


def _phone_access_enabled():
    """LAN is open only when the server was explicitly bound past loopback."""
    return host().strip().lower() not in {"", "127.0.0.1", "localhost", "::1"}


def auth_dependency(request):
    token = os.environ.get(TOKEN_ENV, "")
    if not token:
        return None
    if not _loopback_host(request) and not _phone_access_enabled():
        return None
    provided = (
        request.headers.get(TOKEN_HEADER)
        or request.query_params.get(TOKEN_QUERY)
        or request.cookies.get(TOKEN_COOKIE)
        or _token_from_referer(request, token)
    )
    if _token_ok(provided, token):
        return "local"
    return None


def _output_dirs():
    data = data_dir()
    downloads = os.path.join(
        os.path.expanduser("~"), "Downloads", "Audio Separator"
    )
    return [
        os.path.join(data, "Trabajos"),
        os.path.join(data, "clean_song_output"),
        os.path.join(data, "remix_output"),
        os.path.join(data, "rvc_output"),
        os.path.join(data, "downloads"),
        os.path.join(data, "Voces"),
        downloads,
    ]


def allowed_paths():
    paths = _output_dirs()
    for path in paths:
        os.makedirs(path, exist_ok=True)
    return paths


def blocked_paths():
    blocked = [
        os.path.expanduser("~/.ssh"),
        os.path.expanduser("~/.gnupg"),
        os.path.expanduser("~/.aws"),
        "/etc",
    ]
    code = package_dir()
    data = data_dir()
    if os.path.realpath(data) != os.path.realpath(code):
        blocked.append(code)
    else:
        blocked.extend(
            [
                os.path.join(code, ".git"),
                os.path.join(code, ".cursor"),
                os.path.join(code, ".env"),
            ]
        )
    app_home = home()
    if os.path.realpath(app_home) != os.path.realpath(data):
        blocked.append(app_home)
    return [path for path in blocked if path]


def _public_message(exc):
    text = str(exc).strip() or "Algo salió mal."
    if _PATH_RE.search(text):
        return "Algo salió mal."
    return text


def ui_error(exc):
    import gradio as gr

    if isinstance(exc, gr.Error):
        raise exc
    if isinstance(exc, ValueError):
        raise gr.Error(_public_message(exc)) from exc
    raise gr.Error("Algo salió mal.") from exc


def _positive_dim(shape, index):
    try:
        value = shape[index]
    except (IndexError, TypeError):
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def mdx_config(model_path, session):
    config = {"hop": 1024, "overlap": 0.5, "compensate": 1.0}
    sidecar = os.path.splitext(model_path)[0] + ".json"
    if os.path.isfile(sidecar):
        with open(sidecar, encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("El .json del modelo tiene que ser un objeto.")
        for key in ("hop", "overlap", "compensate", "dim_f", "dim_t", "n_fft"):
            if key in loaded:
                config[key] = loaded[key]
    shape = session.get_inputs()[0].shape
    if "dim_f" not in config:
        dim_f = _positive_dim(shape, 2)
        if not dim_f:
            raise ValueError(
                "No pude leer dim_f del ONNX. Dejá un .json al lado con n_fft, dim_f y dim_t."
            )
        config["dim_f"] = dim_f
    if "dim_t" not in config:
        dim_t = _positive_dim(shape, 3)
        if not dim_t:
            raise ValueError(
                "No pude leer dim_t del ONNX. Dejá un .json al lado con n_fft, dim_f y dim_t."
            )
        config["dim_t"] = dim_t
    if "n_fft" not in config:
        dim_f = int(config["dim_f"])
        config["n_fft"] = 7680 if dim_f == 3072 else dim_f * 2
    if int(config["n_fft"]) // 2 + 1 < int(config["dim_f"]):
        config["n_fft"] = int(config["dim_f"]) * 2
    n_fft = int(config["n_fft"])
    hop = int(config["hop"])
    dim_f = int(config["dim_f"])
    dim_t = int(config["dim_t"])
    if (
        n_fft > MAX_N_FFT
        or hop > MAX_HOP
        or dim_f > MAX_DIM_F
        or dim_t > MAX_DIM_T
        or hop < 1
        or dim_t < 2
        or dim_f < 1
        or n_fft < 2
    ):
        raise ValueError("La configuración del modelo ONNX no es válida.")
    return config


def ensure_token():
    if not os.environ.get(TOKEN_ENV):
        os.environ[TOKEN_ENV] = secrets.token_urlsafe(32)
    return os.environ[TOKEN_ENV]
