"""Train run paths, detached supervisor, and inference-weight export.

Does not import torch at module load. Export runs in a subprocess.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


def voice_dir(exp_name: str) -> Path:
    from app_env import data_dir

    path = Path(data_dir()) / "Voces" / exp_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def library_root() -> Path:
    """Kept so old callers do not recreate Application Support/library."""
    from app_env import data_dir

    return Path(data_dir()) / "Voces"


def runs_dir(exp_name: str) -> Path:
    path = voice_dir(exp_name) / "trabajo"
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_data_dir(exp_name: str) -> Path:
    path = voice_dir(exp_name) / "entrada"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ckpt_dir(exp_name: str) -> Path:
    path = runs_dir(exp_name) / "ckpt"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_path(exp_name: str) -> Path:
    return runs_dir(exp_name) / "train.log"


def job_path(exp_name: str) -> Path:
    return runs_dir(exp_name) / "job.json"


def published_path(exp_name: str) -> Path:
    return runs_dir(exp_name) / "published.json"


def write_job(exp_name: str, files: list, epochs) -> Path:
    dest = job_path(exp_name)
    payload = {
        "exp": exp_name,
        "epochs": epochs,
        "files": [str(Path(item).resolve()) for item in files if item],
    }
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def write_published(
    exp_name: str,
    *,
    ok: bool,
    pth: str | None = None,
    index: str | None = None,
    error: str | None = None,
) -> Path:
    dest = published_path(exp_name)
    dest.write_text(
        json.dumps(
            {"ok": ok, "pth": pth, "index": index, "error": error},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return dest


def read_published(exp_name: str) -> dict | None:
    path = published_path(exp_name)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def spawn_supervisor(job: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["RVC_TRAIN_SUPERVISOR"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    root = Path(__file__).resolve().parent
    return subprocess.Popen(
        [sys.executable, "-m", "train_run", "supervise", str(job)],
        cwd=str(root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def wait_supervisor(proc: subprocess.Popen, exp_name: str, progress=None, total_epochs=None):
    log = log_path(exp_name)
    offset = 0
    while proc.poll() is None:
        offset = _feed_progress(log, offset, progress, total_epochs)
        published = read_published(exp_name)
        if published and published.get("ok"):
            break
        time.sleep(0.5)
    _feed_progress(log, offset, progress, total_epochs)
    code = proc.wait()
    published = read_published(exp_name)
    if published and published.get("ok") and published.get("pth"):
        return published["pth"], published.get("index")
    err = (published or {}).get("error") if published else None
    raise RuntimeError(err or f"Train terminó mal (exit {code}).")


def _feed_progress(log: Path, offset: int, progress, total_epochs) -> int:
    if not log.is_file():
        return offset
    try:
        with open(log, encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            chunk = handle.read()
            offset = handle.tell()
    except OSError:
        return offset
    if progress is None or not chunk:
        return offset
    last = ""
    for line in chunk.splitlines():
        last = line.strip()
        frac = None
        if total_epochs and last:
            match = re.search(r"Training epoch:\s*(\d+)", last)
            if match:
                epoch = int(match.group(1))
                frac = 0.45 + 0.5 * min(epoch / max(int(total_epochs), 1), 1.0)
        if last:
            try:
                progress(frac if frac is not None else 0, desc=last[:80])
            except Exception:
                pass
    return offset


def export_weight(exp_name: str, log: Path, epochs) -> str | None:
    """Run inference-weight export in a subprocess so the UI never imports torch."""
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "train_run",
        "export",
        exp_name,
        str(log),
        str(epochs if epochs is not None else ""),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "").strip().splitlines()
    path = out[-1] if out else ""
    if proc.returncode != 0 or not path:
        err = (proc.stderr or proc.stdout or "").strip()[-1500:]
        raise RuntimeError(err or "No se pudo exportar el .pth de inferencia.")
    return path


def supervise(job_file: str) -> int:
    import occupancy

    job = json.loads(Path(job_file).read_text(encoding="utf-8"))
    exp = job["exp"]
    occupancy.acquire(occupancy.HOLD_TRAIN, exp)
    try:
        import rvc_train

        pth, index = rvc_train.execute_train(
            exp, job.get("files") or [], epochs=job.get("epochs")
        )
        write_published(exp, ok=True, pth=pth, index=index)
        return 0
    except Exception as exc:
        existing = read_published(exp)
        if not (existing and existing.get("ok") and existing.get("pth")):
            write_published(exp, ok=False, error=str(exc)[-1500:])
        return 1
    finally:
        occupancy.release(occupancy.HOLD_TRAIN)


def _export_main(exp_name: str, log: str, epochs_raw: str) -> int:
    from pathlib import Path as P

    import rvc_train

    epochs = int(epochs_raw) if str(epochs_raw).strip().isdigit() else None
    path = rvc_train._ensure_inference_weight(
        exp_name, log_path=P(log) if log else None, epochs=epochs
    )
    if not path:
        return 1
    print(path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) >= 2 and args[0] == "supervise":
        return supervise(args[1])
    if len(args) >= 2 and args[0] == "export":
        log = args[2] if len(args) > 2 else ""
        epochs = args[3] if len(args) > 3 else ""
        return _export_main(args[1], log, epochs)
    print("uso: python -m train_run supervise JOB.json", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
