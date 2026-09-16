"""Text → Edge TTS → RVC using the app's stable convert path.

Does NOT use tts-with-rvc's bundled RVC (segfaults on Mac Intel with RMVPE).
Edge needs internet once; RVC convert is local via rvc_engine.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

from library import find_index_for_model, list_rvc_voices
from rvc_engine import convert_voice

APP_ROOT = Path(__file__).resolve().parent
WORK = APP_ROOT / "library" / "tts_rvc_work"

EDGE_VOICES = [
    ("es-AR-ElenaNeural", "Argentina — Elena"),
    ("es-AR-TomasNeural", "Argentina — Tomás"),
    ("es-MX-DaliaNeural", "México — Dalia"),
    ("es-MX-JorgeNeural", "México — Jorge"),
    ("es-ES-ElviraNeural", "España — Elvira"),
    ("es-ES-AlvaroNeural", "España — Álvaro"),
    ("es-US-PalomaNeural", "EE.UU. — Paloma"),
    ("es-US-AlonsoNeural", "EE.UU. — Alonso"),
]


def resolve_voice_model(voice_name_or_path: str | None) -> tuple[str, str | None]:
    voices = list_rvc_voices()
    if not voices:
        raise ValueError(
            "No hay modelos RVC en la biblioteca. Entrená uno o cargá un .pth."
        )
    path = voice_name_or_path
    if not path:
        path = voices[0]["path"]
    elif not os.path.isfile(str(path)):
        stem = Path(str(path)).stem
        match = next((v for v in voices if v["name"] == stem), None)
        if not match:
            raise ValueError(f"No está el modelo RVC «{path}».")
        path = match["path"]
    index = find_index_for_model(path)
    return os.path.abspath(path), (os.path.abspath(index) if index else None)


async def _edge_tts_to_file(text: str, voice: str, dest_mp3: Path) -> Path:
    import edge_tts

    dest_mp3.parent.mkdir(parents=True, exist_ok=True)
    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(str(dest_mp3))
    if not dest_mp3.is_file() or dest_mp3.stat().st_size == 0:
        raise RuntimeError("Edge TTS no produjo audio (¿hay internet?).")
    return dest_mp3


def _to_wav(src: Path, dest_wav: Path) -> Path:
    """Edge saves MPEG; convert to real WAV for RVC."""
    import subprocess

    dest_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "40000",
        str(dest_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not dest_wav.is_file():
        raise RuntimeError(
            "No se pudo pasar Edge TTS a WAV (¿ffmpeg instalado?): "
            + (proc.stderr.decode("utf-8", "ignore")[:200])
        )
    return dest_wav


def speak_with_rvc(
    text: str,
    model_path: str | None = None,
    *,
    edge_voice: str = "es-AR-ElenaNeural",
    pitch: int = 0,
    index_rate: float = 0.66,
) -> str:
    """Edge TTS → wav → library RVC convert → Downloads."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Escribí un texto.")
    # Gradio dropdown may pass label or path
    if isinstance(model_path, (list, tuple)) and model_path:
        model_path = model_path[0]
    pth, index = resolve_voice_model(model_path)
    WORK.mkdir(parents=True, exist_ok=True)
    edge_mp3 = WORK / "edge_tmp.mp3"
    edge_wav = WORK / "edge_tmp.wav"
    asyncio.run(
        _edge_tts_to_file(text, edge_voice or "es-AR-ElenaNeural", edge_mp3)
    )
    _to_wav(edge_mp3, edge_wav)
    if not edge_wav.is_file():
        raise ValueError("Falta el audio de Edge TTS tras convertir a WAV.")

    out = convert_voice(
        str(edge_wav),
        pth,
        index,
        pitch=int(pitch),
        index_rate=float(index_rate),
        f0_method="rmvpe",
        copy_downloads=True,
    )
    # Rename downloads copy to a clearer name when possible
    if out and os.path.isfile(out):
        nicer = Path(out).with_name("voz_tts_rvc.wav")
        if Path(out).resolve() != nicer.resolve():
            try:
                nicer.write_bytes(Path(out).read_bytes())
                return str(nicer)
            except OSError:
                pass
        return out
    raise RuntimeError("La conversión RVC no produjo audio.")
