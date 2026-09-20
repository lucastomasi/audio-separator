"""Single Intel-CPU occupancy lock. File + ps. Do not import torch."""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

HOLD_UVR = "uvr"
HOLD_TRAIN = "train"
HOLD_CONVERT = "convert"

MSG_TRAIN = "Hay un entrenamiento en curso. Convertir está bloqueado."
MSG_CONVERT = "Hay una conversión en curso. Esperá a que termine."
MSG_UVR = "Se está separando audio. Esperá a que termine."

_MESSAGES = {
    HOLD_TRAIN: MSG_TRAIN,
    HOLD_CONVERT: MSG_CONVERT,
    HOLD_UVR: MSG_UVR,
}


@dataclass(frozen=True)
class Occupancy:
    holder: str
    pid: int | None = None
    exp: str | None = None


def lock_path() -> Path:
    from app_env import data_dir

    path = Path(data_dir()) / "library" / "train_runs" / "occupancy.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def blocked_message(occ: Occupancy | None) -> str:
    if occ is None:
        return MSG_TRAIN
    return _MESSAGES.get(occ.holder, MSG_TRAIN)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _ps_commands() -> str:
    try:
        return subprocess.check_output(["ps", "ax", "-o", "command="], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return ""


def _train_from_ps(text: str) -> Occupancy | None:
    for line in text.splitlines():
        if "train.train" not in line:
            continue
        exp = None
        parts = line.split()
        for i, part in enumerate(parts):
            if part == "-e" and i + 1 < len(parts):
                exp = parts[i + 1]
                break
        return Occupancy(HOLD_TRAIN, exp=exp)
    return None


def _convert_from_ps(text: str) -> bool:
    return "infer_worker.py" in text


def _read_lock() -> Occupancy | None:
    path = lock_path()
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    holder = data.get("holder")
    if holder not in _MESSAGES:
        return None
    pid = data.get("pid")
    try:
        pid = int(pid) if pid is not None else None
    except (TypeError, ValueError):
        pid = None
    if pid is not None and not _pid_alive(pid):
        return None
    exp = data.get("exp")
    return Occupancy(holder, pid=pid, exp=str(exp) if exp else None)


def _clear_lock() -> None:
    path = lock_path()
    try:
        path.unlink()
    except OSError:
        pass


def snapshot() -> Occupancy | None:
    """Who holds Intel right now. ps wins over a stale lock file."""
    text = _ps_commands()
    train = _train_from_ps(text)
    if train is not None:
        return train
    locked = _read_lock()
    if locked is not None:
        return locked
    if _convert_from_ps(text):
        return Occupancy(HOLD_CONVERT)
    _clear_lock()
    return None


def acquire(holder: str, exp: str | None = None) -> Occupancy:
    if holder not in _MESSAGES:
        raise ValueError("Ocupación inválida.")
    current = snapshot()
    if current is not None:
        same = (
            current.holder == holder
            and current.pid is not None
            and current.pid == os.getpid()
        )
        if not same:
            raise ValueError(blocked_message(current))
    payload = {
        "holder": holder,
        "exp": exp,
        "pid": os.getpid(),
        "started": time.time(),
    }
    lock_path().write_text(json.dumps(payload), encoding="utf-8")
    return Occupancy(holder, pid=os.getpid(), exp=exp)


def release(holder: str) -> None:
    locked = _read_lock()
    if locked is None:
        _clear_lock()
        return
    if locked.holder != holder:
        return
    if locked.pid is not None and locked.pid != os.getpid() and _pid_alive(locked.pid):
        return
    _clear_lock()
