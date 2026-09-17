"""Isolated conversion worker client. Do not import torch here."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time

def _paths():
    from app_env import data_dir, home, vc_python as env_vc_python, vc_root as env_vc_root

    data = data_dir()
    return {
        "app": home(),
        "log": os.path.join(data, "library", "train_runs", "vc_infer.log"),
        "pid": os.path.join(data, "library", "train_runs", "vc_worker.pid"),
        "py": env_vc_python(),
        "root": env_vc_root(),
    }


def vc_root():
    return _paths()["root"]


def vc_python():
    return _paths()["py"]


def infer_log_path():
    return _paths()["log"]
_worker: subprocess.Popen | None = None
_lock = threading.Lock()


# paths via _paths() / app_env


def last_infer_tail(n: int = 12) -> str:
    log = infer_log_path()
    if not os.path.isfile(log):
        return ""
    try:
        with open(log, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        return "".join(lines[-n:]).rstrip()
    except OSError:
        return ""


def ensure_vc_engine():
    py = vc_python()
    if os.path.isfile(py) and os.access(py, os.X_OK):
        return py
    script = os.path.join(_paths()["app"], "scripts", "ensure_vc_venv.sh")
    if not os.path.isfile(script):
        raise ValueError(
            "Falta el motor de conversión. Pulsá Completar instalación."
        )
    result = subprocess.run(
        ["bash", script],
        cwd=_paths()["app"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not os.path.isfile(py):
        err = (result.stderr or result.stdout or "").strip()[-800:]
        raise ValueError(
            "Falta el motor de conversión. Pulsá Completar instalación. " + err
        )
    return py


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _stop_unlocked() -> None:
    global _worker
    proc = _worker
    _worker = None
    if proc and proc.poll() is None:
        try:
            proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            proc.stdin.flush()
            proc.wait(timeout=8)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
    pid_path = _paths()["pid"]
    if os.path.isfile(pid_path):
        try:
            pid = int(open(pid_path, encoding="utf-8").read().strip())
        except (OSError, ValueError):
            pid = None
        if pid and _pid_alive(pid):
            try:
                os.kill(pid, 15)
            except OSError:
                pass
        try:
            os.remove(pid_path)
        except OSError:
            pass


def stop_vc_worker() -> None:
    with _lock:
        _stop_unlocked()


def _spawn_worker() -> subprocess.Popen:
    py = ensure_vc_engine()
    worker = os.path.join(vc_root(), "infer_worker.py")
    if not os.path.isfile(worker):
        raise ValueError(
            "Falta el motor de conversión. Pulsá Completar instalación."
        )
    log_path = infer_log_path()
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = open(log_path, "a", encoding="utf-8")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [py, "-u", worker],
        cwd=vc_root(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
        bufsize=1,
        env=env,
    )
    with open(_paths()["pid"], "w", encoding="utf-8") as handle:
        handle.write(str(proc.pid))
    log.write(f"\n--- worker pid={proc.pid} ---\n")
    log.flush()
    return proc


def _get_worker() -> subprocess.Popen:
    global _worker
    if _worker is not None and _worker.poll() is None:
        return _worker
    _worker = _spawn_worker()
    return _worker


def _ask(proc: subprocess.Popen, payload: dict, timeout: float = 3600) -> dict:
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    proc.stdin.write(line)
    proc.stdin.flush()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise ValueError("El motor de conversión se cerró.")
        raw = proc.stdout.readline()
        if not raw:
            raise ValueError("El motor de conversión no respondió.")
        raw = raw.strip()
        if not raw:
            continue
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            continue
    raise ValueError("Timeout esperando al motor de conversión.")


def run_vc_infer(
    audio_path: str,
    model_path: str,
    index_path: str | None = None,
    *,
    pitch: int = 0,
    index_rate: float = 0.66,
    f0_method: str = "rmvpe",
    protect: float = 0.33,
) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    job = {
        "cmd": "infer",
        "input": os.path.abspath(audio_path),
        "output": tmp.name,
        "pth": os.path.abspath(model_path),
        "index": os.path.abspath(index_path) if index_path and os.path.isfile(str(index_path)) else "",
        "pitch": int(pitch),
        "index_rate": float(index_rate),
        "protect": float(protect),
        "f0_method": f0_method or "rmvpe",
        "split_audio": True,
        "embedder": "contentvec",
    }
    last_error = None
    for attempt in range(2):
        with _lock:
            try:
                proc = _get_worker()
                reply = _ask(proc, job)
            except Exception as exc:
                last_error = exc
                _stop_unlocked()
                continue
        if reply.get("ok") and reply.get("path") and os.path.isfile(reply["path"]):
            return os.path.abspath(reply["path"])
        last_error = ValueError(reply.get("error") or "sin audio")
        with _lock:
            _stop_unlocked()
    tail = last_infer_tail(20)
    raise ValueError(
        "La conversión no produjo audio. Log: "
        + infer_log_path()
        + (f" ({last_error})" if last_error else "")
        + (("\n" + tail) if tail else "")
    )
