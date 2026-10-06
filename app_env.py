"""Independent AUDIO_SEPARATOR_* environment. No user-home paths hardcoded."""
from __future__ import annotations

import os
import socket


def package_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def in_app_bundle() -> bool:
    return ".app/Contents/Resources" in package_dir().replace("\\", "/")


def home() -> str:
    value = os.environ.get("AUDIO_SEPARATOR_HOME")
    if value:
        return os.path.abspath(value)
    return package_dir()


def data_dir() -> str:
    """User data. Never the code tree, unless AUDIO_SEPARATOR_DATA overrides it."""
    value = os.environ.get("AUDIO_SEPARATOR_DATA")
    if value:
        path = os.path.abspath(value)
    else:
        path = os.path.join(
            os.path.expanduser("~"),
            "Library",
            "Application Support",
            "Audio Separator",
        )
    os.makedirs(path, exist_ok=True)
    return path


def host() -> str:
    return os.environ.get("AUDIO_SEPARATOR_HOST") or "127.0.0.1"


def preferred_port() -> int:
    raw = os.environ.get("AUDIO_SEPARATOR_PORT") or "7860"
    try:
        return int(raw)
    except ValueError:
        return 7860


def pick_port(start: int | None = None) -> int:
    """Bind the first free port and export AUDIO_SEPARATOR_PORT."""
    start = preferred_port() if start is None else start
    bind_host = host()
    for port in [start] + list(range(start + 1, start + 32)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((bind_host, port))
            sock.close()
            os.environ["AUDIO_SEPARATOR_PORT"] = str(port)
            return port
        except OSError:
            continue
        finally:
            try:
                sock.close()
            except OSError:
                pass
    raise RuntimeError("No hay puerto local libre para la app.")


def vc_python() -> str:
    override = os.environ.get("AUDIO_SEPARATOR_VC_PYTHON")
    if override:
        return override
    return os.path.join(home(), ".venv-vc", "bin", "python")


def vc_root() -> str:
    return os.path.join(home(), "third_party", "vc")
