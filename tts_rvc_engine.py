"""Text → Edge TTS → RVC using the app's stable convert path.

Does NOT use tts-with-rvc's bundled RVC (segfaults on Mac Intel with RMVPE).
Edge needs internet once; RVC convert is local via rvc_engine.
"""
from __future__ import annotations

import asyncio
import os
import re
import unicodedata
from pathlib import Path

from library import find_index_for_model, list_rvc_voices
from rvc_engine import convert_voice

APP_ROOT = Path(__file__).resolve().parent
WORK = APP_ROOT / "library" / "tts_rvc_work"

# Gradio 6 Dropdown tuples: (label, value). Value MUST be the Edge ShortName.
EDGE_VOICES = [
    ("Elena — Argentina", "es-AR-ElenaNeural"),
    ("Tomás — Argentina", "es-AR-TomasNeural"),
    ("Dalia — México", "es-MX-DaliaNeural"),
    ("Jorge — México", "es-MX-JorgeNeural"),
    ("Elvira — España", "es-ES-ElviraNeural"),
    ("Álvaro — España", "es-ES-AlvaroNeural"),
    ("Paloma — EE.UU.", "es-US-PalomaNeural"),
    ("Alonso — EE.UU.", "es-US-AlonsoNeural"),
]

EDGE_VOICE_IDS = {value for _label, value in EDGE_VOICES}
_EDGE_ID_RE = re.compile(r"es-[A-Z]{2}-[A-Za-z]+Neural")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    for dash in ("\u2014", "\u2013", "\u2212"):
        text = text.replace(dash, "-")
    # fold accents: Tomás -> Tomas
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    return text.strip().lower()


def resolve_edge_voice(raw) -> str:
    """Map Gradio dropdown value/label → Edge ShortName."""
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[-1]  # prefer value if (label, value) leaked as tuple
    if isinstance(raw, dict):
        raw = raw.get("value") or raw.get("path") or raw.get("name") or ""
    voice = str(raw or "").strip()
    if not voice:
        return "es-AR-ElenaNeural"
    if voice in EDGE_VOICE_IDS:
        return voice
    # Exact / fuzzy label match
    n = _norm(voice)
    for label, vid in EDGE_VOICES:
        if voice == label or voice == vid:
            return vid
        if _norm(label) == n or _norm(vid) == n:
            return vid
        # "argentina tomas" / "tomas argentina"
        if "tomas" in n and "argent" in n and vid == "es-AR-TomasNeural":
            return vid
        if "elena" in n and "argent" in n and vid == "es-AR-ElenaNeural":
            return vid
    m = _EDGE_ID_RE.search(voice)
    if m and m.group(0) in EDGE_VOICE_IDS:
        return m.group(0)
    raise ValueError(
        f"Voz Edge inválida: {voice!r}. Elegí una de la lista "
        f"(ej. Tomás — Argentina → es-AR-TomasNeural)."
    )


def resolve_voice_model(voice_name_or_path: str | None) -> tuple[str, str | None]:
    voices = list_rvc_voices()
    if not voices:
        raise ValueError(
            "No hay modelos RVC en la biblioteca. Entrená uno o cargá un .pth."
        )
    path = voice_name_or_path
    if isinstance(path, (list, tuple)) and path:
        path = path[-1]
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

    if voice not in EDGE_VOICE_IDS:
        raise ValueError(f"Refuse Edge voice that is not a ShortName: {voice!r}")
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
) -> tuple[str, str]:
    """TTS (ElevenLabs or Edge) → wav → RVC. Returns (path, source_label)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Escribí un texto.")
    if isinstance(model_path, (list, tuple)) and model_path:
        model_path = model_path[-1]
    pth, index = resolve_voice_model(model_path)
    voice = resolve_edge_voice(edge_voice)

    WORK.mkdir(parents=True, exist_ok=True)
    edge_mp3 = WORK / "edge_tmp.mp3"
    edge_wav = WORK / "edge_tmp.wav"
    used_el = False
    try:
        import eleven_tts

        if eleven_tts.available():
            eleven_tts.speak_to_mp3(text, edge_mp3)
            used_el = True
    except Exception:
        used_el = False
    if not used_el:
        asyncio.run(_edge_tts_to_file(text, voice, edge_mp3))
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
    if out and os.path.isfile(out):
        nicer = Path(out).with_name("voz_tts_rvc.wav")
        if Path(out).resolve() != nicer.resolve():
            try:
                nicer.write_bytes(Path(out).read_bytes())
                return str(nicer), ("ElevenLabs" if used_el else f"Edge {voice}")
            except OSError:
                pass
        return out, ("ElevenLabs" if used_el else f"Edge {voice}")
    raise RuntimeError("La conversión RVC no produjo audio.")
