"""API keys for optional acceleration. Never commit this file's JSON."""
from __future__ import annotations

import json
import os
from pathlib import Path

from app_env import data_dir

KEYS = (
    "RUNPOD_API_KEY",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_VOICE_ID",
    "GEMINI_API_KEY",
    "HF_TOKEN",
    "RUNPOD_POD_ID",
)


def secrets_path() -> Path:
    return Path(data_dir()) / "gpu_secrets.json"


def load() -> dict:
    data = {}
    path = secrets_path()
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data.update({k: str(v) for k, v in loaded.items() if v})
        except (OSError, json.JSONDecodeError):
            pass
    for key in KEYS:
        env = os.environ.get(key)
        if env:
            data[key] = env
    return data


def save(updates: dict) -> None:
    path = secrets_path()
    current = {}
    if path.is_file():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            current = {}
    if not isinstance(current, dict):
        current = {}
    for key, value in updates.items():
        if key not in KEYS:
            continue
        if value:
            current[key] = str(value)
        else:
            current.pop(key, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")


def get(name: str) -> str | None:
    value = load().get(name)
    return value or None
