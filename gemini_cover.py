"""Optional Gemini image for album covers. Pillow remains the default."""
from __future__ import annotations

import base64
from pathlib import Path

import requests

from gpu_secrets import get


def available() -> bool:
    return bool(get("GEMINI_API_KEY"))


def generate_png(title: str, artist: str, dest: Path) -> Path:
    key = get("GEMINI_API_KEY")
    if not key:
        raise ValueError("No hay API key de Gemini (AI Studio, no la app).")
    prompt = (
        f"Square album cover, no text, graphic, "
        f"title mood '{title or 'song'}' artist '{artist or ''}'"
    )
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.0-flash-preview-image-generation:generateContent"
    )
    response = requests.post(
        url,
        params={"key": key},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
        },
        timeout=60,
    )
    if response.status_code in (401, 403):
        raise ValueError("Gemini: key inválida o no es la API de AI Studio.")
    if response.status_code == 429:
        raise ValueError("Cupo Gemini agotado. Uso portada local.")
    if response.status_code >= 400:
        raise ValueError("Gemini no generó imagen.")
    payload = response.json()
    blob = _first_image_b64(payload)
    if not blob:
        raise ValueError("Gemini no devolvió imagen.")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(base64.b64decode(blob))
    return dest


def _first_image_b64(payload: dict) -> str | None:
    for cand in payload.get("candidates") or []:
        content = cand.get("content") or {}
        for part in content.get("parts") or []:
            inline = part.get("inlineData") or part.get("inline_data") or {}
            data = inline.get("data")
            if data:
                return data
    return None
