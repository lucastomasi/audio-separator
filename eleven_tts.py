"""Optional ElevenLabs TTS. Fallback is Edge in tts_rvc_engine."""
from __future__ import annotations

from pathlib import Path

import requests

from gpu_secrets import get

DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"  # multilingual, override with ELEVENLABS_VOICE_ID
API = "https://api.elevenlabs.io/v1"


def available() -> bool:
    return bool(get("ELEVENLABS_API_KEY"))


def speak_to_mp3(text: str, dest: Path, voice_id: str | None = None) -> Path:
    key = get("ELEVENLABS_API_KEY")
    if not key:
        raise ValueError("No hay API key de ElevenLabs.")
    text = (text or "").strip()
    if not text:
        raise ValueError("Escribí un texto.")
    voice = voice_id or get("ELEVENLABS_VOICE_ID") or DEFAULT_VOICE
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.post(
        f"{API}/text-to-speech/{voice}",
        headers={
            "xi-api-key": key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        },
        json={"text": text, "model_id": "eleven_multilingual_v2"},
        timeout=60,
    )
    if response.status_code == 401:
        raise ValueError("ElevenLabs: key inválida.")
    if response.status_code == 402 or response.status_code == 429:
        raise ValueError("Cupo ElevenLabs agotado.")
    if response.status_code >= 400:
        raise ValueError("ElevenLabs no pudo sintetizar.")
    dest.write_bytes(response.content)
    if dest.stat().st_size < 100:
        raise ValueError("ElevenLabs no produjo audio.")
    return dest
