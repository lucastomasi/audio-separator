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


def _cache_has_files(path: Path) -> bool:
    try:
        return any(child.name != ".DS_Store" for child in path.iterdir())
    except OSError:
        return False


def runs_dir(exp_name: str) -> Path:
    """RVC experiment dir. Bytes live in Cache; Voces/<name>/trabajo is a link.

    Post-publish garbage collection drops features, G_/D_, checkpoints and entrada;
    the canonical .pth/.index in Voces remain available for inference.
    """
    from app_env import data_dir

    cache = Path(data_dir()) / "Cache" / "voces" / exp_name
    cache.parent.mkdir(parents=True, exist_ok=True)
    link = voice_dir(exp_name) / "trabajo"
    if not link.is_symlink() and link.is_dir():
        if _cache_has_files(cache):
            return link
        if cache.exists():
            cache.rmdir()
        link.rename(cache)
    cache.mkdir(parents=True, exist_ok=True)
    if not link.is_symlink() and not link.exists():
        link.symlink_to(os.path.relpath(cache, start=link.parent))
    return link


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


def latest_job() -> dict | None:
    """Newest train job on disk: name, files, epochs."""
    from app_env import data_dir

    root = Path(data_dir()) / "Voces"
    if not root.is_dir():
        return None
    best = None
    best_mtime = -1.0
    for child in root.iterdir():
        path = child / "trabajo" / "job.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            mtime = path.stat().st_mtime
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict) or not data.get("exp") or not data.get("files"):
            continue
        if mtime >= best_mtime:
            best = data
            best_mtime = mtime
    return best


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
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("USE_LIBUV", "0")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    root = Path(__file__).resolve().parent
    exp = "run"
    try:
        payload = json.loads(Path(job).read_text(encoding="utf-8"))
        if payload.get("exp"):
            exp = str(payload["exp"])
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    supervisor_log = log_path(exp).with_name("supervisor.log")
    supervisor_log.parent.mkdir(parents=True, exist_ok=True)
    handle = open(supervisor_log, "ab")
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "train_run", "supervise", str(job)],
            cwd=str(root),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        handle.close()


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
            match = re.search(
                r"(?:Training epoch:\s*|epoch=)(\d+)", last
            )
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
    runner = sys.executable
    try:
        import rvc_train

        runner = rvc_train._engine_python()
        env["PYTHONPATH"] = str(rvc_train.RVC_ROOT) + os.pathsep + env.get(
            "PYTHONPATH", ""
        )
    except Exception:
        pass
    cmd = [
        runner,
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


def _last_log_line(exp: str | None) -> str:
    if not exp:
        return ""
    for name in ("train.log", "supervisor.log"):
        path = log_path(exp).with_name(name)
        if not path.is_file():
            continue
        try:
            lines = [
                line.strip()
                for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip()
            ]
        except OSError:
            continue
        if lines:
            return lines[-1][:180]
    return ""


def _newest_published() -> dict | None:
    from app_env import data_dir

    root = Path(data_dir()) / "Voces"
    if not root.is_dir():
        return None
    best = None
    best_mtime = -1.0
    for child in root.iterdir():
        path = child / "trabajo" / "published.json"
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            mtime = path.stat().st_mtime
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if not isinstance(data, dict) or not data.get("ok") or not data.get("pth"):
            continue
        if mtime >= best_mtime:
            best = data
            best_mtime = mtime
    return best


def boot_status() -> str | None:
    """What to show when the app opens. Does not start a train."""
    import occupancy

    occ = occupancy.snapshot()
    if occ is not None and occ.holder == occupancy.HOLD_TRAIN:
        line = _last_log_line(occ.exp)
        return line or "Entrenando… cerrar la ventana no lo corta. Separar y Convertir sí."
    published = _newest_published()
    if published:
        name = Path(str(published["pth"])).name
        return f"Modelo listo: {name}"
    return None


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
