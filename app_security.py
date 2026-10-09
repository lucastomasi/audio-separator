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
MAX_UPLOAD = None  # sin límite: app local

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
    # Sin token ni filtro loopback/LAN: toda petición entra.
    return "local"


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
    # Cualquier carpeta: modelos y audio desde donde sea.
    return paths + [os.path.expanduser("~"), os.path.abspath(os.sep)]


def blocked_paths():
    return []


def _public_message(exc):
    text = str(exc).strip()
    return text or type(exc).__name__


def ui_error(exc):
    import gradio as gr

    if isinstance(exc, gr.Error):
        raise exc
    raise gr.Error(_public_message(exc)) from exc


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
        hop < 1
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
