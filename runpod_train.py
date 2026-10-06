"""Optional RunPod hire for RVC train. Always stoppable; TTL default 45 min."""
from __future__ import annotations

import os
from typing import Any

import requests

from gpu_secrets import get, save

API = "https://rest.runpod.io/v1"
DEFAULT_TTL_MIN = 45


def _headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def available() -> bool:
    return bool(get("RUNPOD_API_KEY"))


def estimate_cpu_minutes(n_files: int, epochs: int) -> int:
    """Rough Intel CPU: ~1–2 min per file per epoch at batch 1. Floor 10."""
    n_files = max(int(n_files or 0), 1)
    epochs = max(int(epochs or 1), 1)
    return max(10, n_files * epochs)


def estimate_copy(n_files: int, epochs: int) -> str:
    cpu = estimate_cpu_minutes(n_files, epochs)
    return f"{n_files} audios × {epochs} epochs ≈ {cpu} min en este Mac."


def check_key(session=None) -> str:
    key = get("RUNPOD_API_KEY")
    if not key:
        raise ValueError("No hay API key de RunPod.")
    http = session or requests
    response = http.get(f"{API}/me", headers=_headers(key), timeout=20)
    if response.status_code in (401, 403):
        raise ValueError("RunPod: key inválida.")
    if response.status_code >= 400:
        raise ValueError("RunPod no respondió. Sigo en el Mac.")
    return key


def stop_pod(pod_id: str | None = None, session=None) -> None:
    key = get("RUNPOD_API_KEY")
    pod_id = pod_id or get("RUNPOD_POD_ID")
    if not key or not pod_id:
        return
    http = session or requests
    try:
        http.delete(
            f"{API}/pods/{pod_id}",
            headers=_headers(key),
            timeout=20,
        )
    except requests.RequestException:
        pass
    save({"RUNPOD_POD_ID": ""})


def start_train_pod(
    *,
    ttl_min: int = DEFAULT_TTL_MIN,
    image: str | None = None,
    session=None,
) -> str:
    """Create a CUDA pod. Requires RUNPOD_TRAIN_IMAGE; otherwise caller trains local."""
    key = check_key(session)
    image = image or os.environ.get("RUNPOD_TRAIN_IMAGE")
    if not image:
        raise ValueError(
            "RunPod no tiene imagen de train configurada. Entrenando en este Mac."
        )
    http = session or requests
    body: dict[str, Any] = {
        "name": "audio-separator-train",
        "imageName": image,
        "gpuTypeIds": ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000"],
        "interruptible": False,
        "ttl": max(5, int(ttl_min)) * 60,
    }
    response = http.post(
        f"{API}/pods",
        headers=_headers(key),
        json=body,
        timeout=30,
    )
    if response.status_code >= 400:
        raise ValueError("RunPod no pudo crear el pod. Entrenando en este Mac.")
    payload = response.json() if response.content else {}
    pod_id = str(payload.get("id") or payload.get("podId") or "")
    if not pod_id:
        raise ValueError("RunPod no devolvió id. Entrenando en este Mac.")
    save({"RUNPOD_POD_ID": pod_id})
    return pod_id
