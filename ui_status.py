"""Single status channel for the Gradio bar.

Jobs never interpolate exception text into the UI. Tracebacks go to
~/Library/Logs/audio-separator.log (and logger).
"""
from __future__ import annotations

import os
import traceback
from datetime import datetime

from utils import logger

IDLE = "1 Canción → 2 Extraer → 3 Resultado → 4 Voz → 5 Unir → 6 Texto."
RUN_SEPARATE = "Separando… en Intel puede tardar varios minutos. No cierres la ventana."
RUN_TRAIN = "Entrenando… podés cerrar la ventana; Convertir queda bloqueado."
READY = "Audio listo. Elegí qué extraer y pulsá Separar."

KIND_IDLE = "idle"
KIND_RUN = "run"
KIND_OK = "ok"
KIND_ERROR = "error"

MSG_SEPARATE = "No se pudo separar."
MSG_TRAIN = "Falló el entrenamiento."
MSG_CONVERT = "No se pudo convertir la voz."
MSG_TTS = "Falló texto→RVC. Hace falta internet para el TTS (Edge o ElevenLabs)."
MSG_REMIX = "No se pudo armar el remix."
MSG_REMUX = "No se pudo pegar el audio al video."
MSG_COVER = "No se pudo generar la portada."
MSG_INSTALL = "Falló la descarga. Pulsá Completar instalación de nuevo."


def log_file() -> str:
    return os.path.expanduser("~/Library/Logs/audio-separator.log")


def append_log(text: str) -> str:
    path = log_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {text.rstrip()}\n")
    return path


def log_exception(where: str, exc: BaseException) -> str:
    logger.error("%s: %s", where, exc, exc_info=exc)
    return append_log(f"{where}: {exc}\n{traceback.format_exc()}")


def _usable_valueerror(exc: BaseException) -> str | None:
    if not isinstance(exc, ValueError):
        return None
    msg = str(exc).strip()
    if not msg or len(msg) > 220:
        return None
    lowered = msg.lower()
    if "traceback" in lowered or "is not defined" in lowered:
        return None
    if any(tok in msg for tok in ("/", "File ", "ffmpeg", ".py")):
        return None
    return msg


def fail(where: str, exc: BaseException, user_msg: str) -> str:
    """Log the exception; return a short Spanish line for the status bar."""
    owned = _usable_valueerror(exc)
    log_exception(where, exc)
    return owned or user_msg


def status_update(kind: str, text: str):
    import gradio as gr

    classes = {
        KIND_ERROR: ["is-error"],
        KIND_RUN: ["is-run"],
        KIND_OK: ["is-ok"],
        KIND_IDLE: ["is-idle"],
    }.get(kind, [])
    return gr.update(value=text, elem_classes=classes)
