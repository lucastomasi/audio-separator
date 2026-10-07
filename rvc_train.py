"""CPU training through the vendored Applio stack in third_party/vc."""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import train_run
from library import ensure_dirs, rvc_support_dir

APP_ROOT = Path(__file__).resolve().parent
RVC_ROOT = APP_ROOT / "third_party" / "vc"
SR = "40k"
SR_HZ = 40000
VERSION = "v2"
# Ten epochs is a deliberately small CPU-Mac default; checkpoints and infer weights
# are emitted every five epochs to avoid a large stream of multi-file dumps.
EPOCHS_DEFAULT = 10
SAVE_EVERY_EPOCH = 5
SAVE_EVERY_WEIGHTS = "1"
# Applio layout (not official RVC-WebUI 0_gt_wavs / 3_feature768).
_FEATURE_DIRS = (
    "sliced_audios",
    "sliced_audios_16k",
    "f0",
    "f0_voiced",
    "extracted",
)
BATCH = 1
WORKERS = 2
# Applio train.py uses os._exit(2333333) on a successful finish.
_TRAIN_OK_CODES = {0, 2333333}


def _epochs(override=None):
    raw = override if override is not None else os.environ.get(
        "RVC_TRAIN_EPOCHS", str(EPOCHS_DEFAULT)
    )
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = EPOCHS_DEFAULT
    return max(1, value)


def require_rvc_webui():
    """Train engine is the vendored Applio tree, not a git clone of RVC-WebUI."""
    if not (RVC_ROOT / "rvc" / "train" / "train.py").is_file():
        raise ValueError(
            "Falta third_party/vc (motor Applio de entrenamiento)."
        )


def _engine_python() -> str:
    """Prefer the isolated conversion venv; fall back to this interpreter."""
    try:
        from vc_runner import vc_python

        py = vc_python()
        if os.path.isfile(py) and os.access(py, os.X_OK):
            return py
    except Exception:
        pass
    return sys.executable


def _is_transformers_hubert(path: Path) -> bool:
    """Train needs Transformers layout; classic hubert_base.pt is for convert only."""
    if not path.is_dir():
        return False
    has_config = (path / "config.json").is_file()
    has_weights = any(
        (path / name).is_file()
        for name in (
            "pytorch_model.bin",
            "model.safetensors",
            "model.safetensors.index.json",
        )
    )
    return has_config and has_weights


def require_train_assets():
    ensure_dirs()
    root = Path(rvc_support_dir())
    hubert_dir = root / "hubert_base"
    rmvpe = root / "rmvpe.pt"
    g = root / "f0G40k.pth"
    d = root / "f0D40k.pth"
    missing = []
    hubert = None
    if _is_transformers_hubert(hubert_dir):
        hubert = hubert_dir
    else:
        missing.append(
            "hubert_base/ (Transformers: config.json + pytorch_model.bin)"
        )
    for label, path in (("rmvpe.pt", rmvpe), ("f0G40k.pth", g), ("f0D40k.pth", d)):
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(label)
    if missing:
        raise ValueError(
            "Faltan pesos para entrenar. Pulsá Completar instalación: "
            + ", ".join(missing)
        )
    return {"hubert": hubert, "rmvpe": rmvpe, "g": g, "d": d}


def _features_ready(exp_dir: Path) -> bool:
    feat = exp_dir / "extracted"
    if not feat.is_dir():
        return False
    try:
        return any(path.suffix == ".npy" for path in feat.iterdir())
    except OSError:
        return False


def _library_root() -> Path:
    package = Path(__file__).resolve().parent
    if APP_ROOT.resolve() != package:
        return APP_ROOT / "library"
    return train_run.library_root()


def _ckpt_dir(exp_name: str) -> Path:
    return train_run.ckpt_dir(exp_name)


def _needs_ckpt_copy(src: Path, dest: Path) -> bool:
    if not dest.is_file():
        return True
    try:
        return (
            src.stat().st_mtime > dest.stat().st_mtime
            or src.stat().st_size != dest.stat().st_size
        )
    except OSError:
        return True


def publish_infer_weight(exp_name, log_path: Path | None = None) -> Path | None:
    """Copy the ~50 MB infer .pth into Convertir. Never opens G_/D_ or torch."""
    small = _find_small_weight(exp_name)
    if small is None:
        return None
    try:
        path = train_run.voice_dir(exp_name) / f"{exp_name}.pth"
        if small.resolve() != path.resolve():
            shutil.copy2(small, path)
    except Exception:
        return None
    if log_path is not None:
        try:
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"[library] pth -> {path} ({path.stat().st_size} bytes)\n")
        except OSError:
            pass
    return path


def snapshot_checkpoints(exp_name: str, *, heavy: bool = False) -> list[Path]:
    """Copy infer .pth first. G_/D_ only if heavy=True. Never raises."""
    copied: list[Path] = []
    try:
        dest = _ckpt_dir(exp_name)
        logs = RVC_ROOT / "logs" / exp_name
        weights = RVC_ROOT / "assets" / "weights"
        infer_sources: list[Path] = []
        if weights.is_dir():
            infer_sources.extend(
                path
                for path in weights.glob(f"{exp_name}*.pth")
                if _is_infer_weight(path)
            )
        for src in infer_sources:
            try:
                target = dest / src.name
                if _needs_ckpt_copy(src, target):
                    shutil.copy2(src, target)
                    copied.append(target)
            except OSError:
                continue
        publish_infer_weight(exp_name)
        if not heavy:
            return copied
        heavy_sources: list[Path] = []
        if logs.is_dir():
            heavy_sources.extend(logs.glob("G_*.pth"))
            heavy_sources.extend(logs.glob("D_*.pth"))
        for src in heavy_sources:
            try:
                target = dest / src.name
                if _needs_ckpt_copy(src, target):
                    shutil.copy2(src, target)
                    copied.append(target)
            except OSError:
                continue
    except Exception:
        return copied
    return copied


def train_running(exp_name: str) -> str | None:
    """Return a command line if Applio train.py is alive for this experiment."""
    try:
        out = subprocess.check_output(["ps", "ax", "-o", "command="], text=True)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None
    needle = f"-e {exp_name} "
    for line in out.splitlines():
        parts = line.split()
        applio = "rvc/train/train.py" in line or "rvc.train.train" in line
        webui = "train.train" in line and needle in line
        if applio and exp_name in parts:
            return line.strip()
        if webui:
            return line.strip()
    return None


def _run(
    cmd,
    log_path: Path,
    progress=None,
    frac=None,
    desc=None,
    detach=False,
    exp_name=None,
    total_epochs=None,
):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(RVC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["RVC_AUDIO_FORCE_CPU"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(RVC_ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=detach,
        )
    last = ""
    try:
        offset = log_path.stat().st_size
    except OSError:
        offset = 0
    while proc.poll() is None:
        last, offset = _tail_train_log(
            log_path, offset, last, progress, frac, total_epochs, exp_name
        )
        time.sleep(0.25)
    last, offset = _tail_train_log(
        log_path, offset, last, progress, frac, total_epochs, exp_name
    )
    code = proc.wait()
    if code not in _TRAIN_OK_CODES:
        err = last or ""
        raise RuntimeError(err[-1500:] if err else f"Falló: {' '.join(cmd)}")
    if desc and progress is not None:
        try:
            progress(frac if frac is not None else 0, desc=desc)
        except Exception:
            pass


def _tail_train_log(log_path, offset, last, progress, frac, total_epochs, exp_name):
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            handle.seek(offset)
            chunk = handle.read()
            offset = handle.tell()
    except OSError:
        return last, offset
    for line in chunk.splitlines():
        last = (line or "").strip()
        frac_now = frac
        if total_epochs and last:
            match = re.search(
                r"(?:Training epoch:\s*|epoch=)(\d+)", last
            )
            if match:
                epoch = int(match.group(1))
                frac_now = 0.45 + 0.5 * min(epoch / max(int(total_epochs), 1), 1.0)
        if progress is not None and last:
            try:
                progress(frac_now if frac_now is not None else 0, desc=last[:80])
            except Exception:
                pass
    return last, offset


def _replace_with_link(src: Path, dest: Path):
    """Point dest at src (symlink). Copy only if the OS refuses links."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()
    if dest.exists() or dest.is_symlink():
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        else:
            dest.unlink()
    try:
        os.symlink(src, dest)
        return
    except OSError:
        pass
    if src.is_dir():
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)


def bind_outside(repo_path: Path, outside: Path) -> Path:
    """Move a generated directory out of the repo and leave a symlink."""
    outside.mkdir(parents=True, exist_ok=True)
    if repo_path.is_symlink():
        return repo_path.resolve()
    if repo_path.is_dir():
        for child in list(repo_path.iterdir()):
            dest = outside / child.name
            if dest.exists() or dest.is_symlink():
                continue
            shutil.move(str(child), str(dest))
        leftover = list(repo_path.iterdir())
        if leftover:
            names = ", ".join(path.name for path in leftover[:5])
            raise RuntimeError(f"Siguen archivos en el repo: {names}")
        repo_path.rmdir()
    elif repo_path.exists():
        raise RuntimeError(f"{repo_path} no es una carpeta.")
    repo_path.symlink_to(outside.resolve(), target_is_directory=True)
    return outside.resolve()


def ensure_rvc_outputs_outside() -> None:
    from app_env import data_dir

    base = Path(data_dir()) / "RVC"
    bind_outside(RVC_ROOT / "logs", base / "logs")
    bind_outside(RVC_ROOT / "assets" / "weights", base / "weights")
    bind_outside(RVC_ROOT / "assets" / "indices", base / "indices")


def _sync_assets(assets):
    """Point Applio at library HuBERT / RMVPE / f0G/D without copying bytes."""
    pred = RVC_ROOT / "rvc" / "models" / "predictors"
    emb = RVC_ROOT / "rvc" / "models" / "embedders" / "contentvec"
    pred.mkdir(parents=True, exist_ok=True)
    emb.mkdir(parents=True, exist_ok=True)

    hubert = Path(assets["hubert"])
    for name in (
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "pytorch_model.bin",
    ):
        src = hubert / name
        if src.is_file():
            _replace_with_link(src, emb / name)
    if not (emb / "model.safetensors").is_file() and not (
        emb / "pytorch_model.bin"
    ).is_file():
        raise RuntimeError(
            "Tras sync no hay pesos HuBERT en embedders/contentvec "
            "(model.safetensors)."
        )

    _replace_with_link(Path(assets["rmvpe"]), pred / "rmvpe.pt")


_SAVEE_HELPER = '''
def inference_weights_dir():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "assets", "weights")
    os.makedirs(path, exist_ok=True)
    return path


def inference_weight_path(name):
    base = os.path.basename(str(name))
    if not base.endswith(".pth"):
        base = "%s.pth" % base
    return os.path.join(inference_weights_dir(), base)
'''


def _ensure_savee_absolute():
    """RVC-WebUI is gitignored; patch savee so weights are not cwd-relative."""
    path = RVC_ROOT / "train" / "process_ckpt.py"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if "def inference_weight_path" in text:
        return
    needle = "i18n = I18nAuto()\n"
    if needle in text:
        text = text.replace(needle, needle + "\n" + _SAVEE_HELPER.lstrip("\n"), 1)
    text = text.replace(
        'torch.save(opt, "assets/weights/%s.pth" % name)',
        "torch.save(opt, inference_weight_path(name))",
    )
    text = text.replace(
        'torch.save(ckpt, "assets/weights/%s" % name)',
        "torch.save(ckpt, inference_weight_path(name))",
    )
    path.write_text(text, encoding="utf-8")


def _src_path(value):
    if not value:
        return None
    if isinstance(value, dict):
        for key in ("path", "name", "orig_name"):
            item = value.get(key)
            if isinstance(item, str) and item:
                return item
        return None
    if isinstance(value, (list, tuple)) and value:
        return _src_path(value[0])
    if isinstance(value, (str, os.PathLike)):
        return str(value)
    return getattr(value, "name", None) or getattr(value, "path", None)


def _prepare_dataset(dataset_files, exp_name):
    dataset_dir = train_run.train_data_dir(exp_name)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in dataset_files or []:
        path = _src_path(src)
        if not path or not os.path.isfile(path):
            continue
        ext = os.path.splitext(path)[1].lower() or ".wav"
        from youtube_lib import VIDEO_FILE_EXTS, extract_audio_from_media

        if ext in VIDEO_FILE_EXTS:
            dest = dataset_dir / f"sample_{count:04d}.wav"
            extract_audio_from_media(path, str(dest), sample_rate=SR_HZ, mono=True)
        else:
            shutil.copy2(path, dataset_dir / f"sample_{count:04d}{ext}")
        count += 1
    if count == 0:
        raise ValueError("Subí al menos un audio de la voz a entrenar.")
    return dataset_dir


def _stem_set(folder: Path, pattern: str):
    names = set()
    for path in folder.glob(pattern):
        name = path.name
        if name.endswith(".wav.npy"):
            names.add(name[: -len(".wav.npy")])
        elif name.endswith(".npy"):
            names.add(path.stem)
        elif name.endswith(".wav"):
            names.add(path.stem)
    return names


def _write_filelist_and_config(exp_dir: Path):
    gt = exp_dir / "sliced_audios"
    feat = exp_dir / "extracted"
    f0 = exp_dir / "f0"
    f0nsf = exp_dir / "f0_voiced"
    for folder in (gt, feat, f0, f0nsf):
        if not folder.is_dir():
            raise RuntimeError(f"Falta carpeta tras preprocess: {folder.name}")

    names = (
        _stem_set(gt, "*.wav")
        & _stem_set(feat, "*.npy")
        & _stem_set(f0, "*")
        & _stem_set(f0nsf, "*")
    )
    if not names:
        raise RuntimeError(
            "No hay audios válidos tras preprocess/F0/HuBERT. Revisá el dataset."
        )

    lines = []
    for name in sorted(names):
        f0_path = f0 / f"{name}.wav.npy"
        if not f0_path.is_file():
            f0_path = f0 / f"{name}.npy"
        f0nsf_path = f0nsf / f"{name}.wav.npy"
        if not f0nsf_path.is_file():
            f0nsf_path = f0nsf / f"{name}.npy"
        line = (
            f"{(gt / (name + '.wav')).as_posix()}|"
            f"{(feat / (name + '.npy')).as_posix()}|"
            f"{f0_path.as_posix()}|{f0nsf_path.as_posix()}|0"
        )
        lines.append(line)
    (exp_dir / "filelist.txt").write_text("\n".join(lines), encoding="utf-8")

    template = RVC_ROOT / "rvc" / "configs" / f"{SR_HZ}.json"
    if not template.is_file():
        raise RuntimeError(f"No está el config {template.relative_to(RVC_ROOT)}")
    config = json.loads(template.read_text(encoding="utf-8"))
    config.pop("speaker_info", None)
    (exp_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def _is_infer_weight(path: Path) -> bool:
    """True for extracted inference .pth (~50–80 MB). Never G_/D_ train dumps."""
    name = path.name.lower()
    if name.startswith("g_") or name.startswith("d_"):
        return False
    try:
        mb = path.stat().st_size / (1024 * 1024)
    except OSError:
        return False
    return 20 <= mb <= 150


def _find_small_weight(exp_name):
    """Prefer assets/weights (~50–80 MB). Never return G_*.pth / D_*.pth."""
    search = []
    weights = RVC_ROOT / "assets" / "weights"
    logs = RVC_ROOT / "logs" / exp_name
    if weights.is_dir():
        search.extend(weights.glob("*.pth"))
    if logs.is_dir():
        search.extend(logs.glob("*.pth"))
    if not search:
        return None
    exp = exp_name.lower()
    named = [p for p in search if _is_infer_weight(p) and exp in p.name.lower()]
    pool = named or [p for p in search if _is_infer_weight(p)]
    if not pool:
        return None
    return max(pool, key=lambda p: p.stat().st_mtime)


def _find_index(exp_name):
    search = []
    logs = RVC_ROOT / "logs" / exp_name
    indices = RVC_ROOT / "assets" / "indices"
    if logs.is_dir():
        search.extend(logs.glob("*.index"))
    if indices.is_dir():
        search.extend(indices.glob("*.index"))
    real = [p for p in search if p.is_file()]
    if not real:
        return None
    exp = exp_name.lower()
    added = [
        p for p in real if "added" in p.name.lower() and exp in p.name.lower()
    ]
    if added:
        return max(added, key=lambda p: p.stat().st_size)
    named = [p for p in real if exp in p.name.lower()]
    if named:
        return max(named, key=lambda p: p.stat().st_size)
    return None


def _ensure_inference_weight(
    exp_name, log_path: Path | None = None, epochs=None
):
    """Prefer assets/weights/<exp>.pth; else extract from G_*.pth via savee."""
    def _log(msg: str):
        if log_path is None:
            return
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(msg.rstrip() + "\n")

    expected = RVC_ROOT / "assets" / "weights" / f"{exp_name}.pth"
    existing = _find_small_weight(exp_name)
    if existing is not None:
        _log(f"[weight] usando {existing} ({existing.stat().st_size} bytes)")
        return existing
    _log(f"[weight] no está {expected}; fallback desde G_*.pth")
    logs = RVC_ROOT / "logs" / exp_name
    g_candidates = sorted(
        logs.glob("G_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True
    ) if logs.is_dir() else []
    if not g_candidates:
        _log("[weight] no hay G_*.pth para fallback")
        return None
    config_path = logs / "config.json"
    if not config_path.is_file():
        _log("[weight] falta config.json para fallback")
        return None
    weights_dir = RVC_ROOT / "assets" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    import torch
    from rvc_engine import _scan_model
    from train.process_ckpt import savee
    from train.utils import HParams

    config = json.loads(config_path.read_text(encoding="utf-8"))
    hps = HParams(**config)
    try:
        _scan_model(str(g_candidates[0]))
    except ValueError as exc:
        raise ValueError(
            "El checkpoint de train no pasó la revisión. No se abre."
        ) from exc
    try:
        ckpt = torch.load(
            str(g_candidates[0]), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise ValueError(
            "No pude leer el checkpoint de train de forma segura."
        ) from exc
    weight = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    prev = os.getcwd()
    try:
        os.chdir(str(RVC_ROOT))
        _log(f"[weight] fallback cwd={os.getcwd()} from {g_candidates[0].name}")
        msg = savee(weight, SR, 1, exp_name, epochs or _epochs(), VERSION, hps)
        _log(f"[weight] savee: {msg}")
    finally:
        os.chdir(prev)
    out = weights_dir / f"{exp_name}.pth"
    if not out.is_file():
        raise RuntimeError(f"No se pudo exportar weight de inferencia: {msg}")
    _log(f"[weight] escrito {out} ({out.stat().st_size} bytes)")
    return out


def _publish_voice_to_library(exp_name, pth: Path, index: Path | None, log_path: Path):
    """Copy inference pth + index into Voces/<exp>/. One folder, no second copy."""
    folder = train_run.voice_dir(exp_name)
    canonical = folder / f"{exp_name}.pth"
    if pth.resolve() != canonical.resolve():
        shutil.copy2(pth, canonical)
    index_dest = None
    if index is not None and index.is_file():
        index_dest = folder / f"{exp_name}.index"
        if index.resolve() != index_dest.resolve():
            shutil.copy2(index, index_dest)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(
            f"[library] pth -> {canonical} ({canonical.stat().st_size} bytes)\n"
        )
        if index_dest:
            log.write(f"[library] index -> {index_dest}\n")
        else:
            log.write("[library] sin index (convert igual puede usar el .pth)\n")
    return str(canonical), (str(index_dest) if index_dest else None)


def finish_train_publish(exp_name, log_path, total, progress=None, py=None):
    """Publish infer .pth before index / G_/D_ copies. Writes published.json."""
    pth = publish_infer_weight(exp_name, log_path)
    if pth is None:
        exported = train_run.export_weight(exp_name, log_path, total)
        if not exported:
            raise RuntimeError(
                "Train terminó sin .pth de inferencia. "
                f"{RVC_ROOT / 'assets' / 'weights' / (exp_name + '.pth')}"
            )
        pth = Path(exported)
    dest_pth, dest_index = _publish_voice_to_library(
        exp_name, Path(pth), _find_index(exp_name), log_path
    )
    train_run.write_published(
        exp_name, ok=True, pth=dest_pth, index=dest_index
    )
    if progress is not None:
        try:
            progress(1.0, desc="Modelo listo")
        except Exception:
            pass
    indices_dir = RVC_ROOT / "assets" / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)
    runner = py or _engine_python()
    exp_dir = RVC_ROOT / "logs" / exp_name
    try:
        _run(
            [
                runner,
                str(RVC_ROOT / "rvc" / "train" / "process" / "extract_index.py"),
                str(exp_dir),
                "Auto",
            ],
            log_path,
            progress=progress,
            frac=0.9,
            desc="Índice…",
        )
        dest_pth, dest_index = _publish_voice_to_library(
            exp_name, Path(dest_pth), _find_index(exp_name), log_path
        )
        train_run.write_published(
            exp_name, ok=True, pth=dest_pth, index=dest_index
        )
    except Exception as exc:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n[index] omitido: {exc}\n")
    discard_run_scratch(exp_name)
    return dest_pth, dest_index


def garbage_collect_train_artifacts(exp_name: str) -> dict[str, int]:
    """Remove published-run scratch without touching the canonical voice files.

    The final ``Voces/<exp>/<exp>.pth`` is the guard: until it exists, no cleanup
    runs.  This removes all periodic inference exports, large G_/D_ checkpoints,
    extracted feature logs, copied input audio, and local checkpoint snapshots.
    The optional ``.index`` beside the canonical model is never targeted.
    """
    voice = train_run.voice_dir(exp_name)
    canonical = voice / f"{exp_name}.pth"
    removed = {"weights": 0, "logs": 0, "entrada": 0, "snapshots": 0}
    if not canonical.is_file():
        return removed

    weights = RVC_ROOT / "assets" / "weights"
    if weights.is_dir():
        for path in weights.glob(f"{exp_name}_e*_s*.pth"):
            if path.is_file() and not path.is_symlink():
                try:
                    path.unlink()
                    removed["weights"] += 1
                except OSError:
                    pass

    # RVC logs contain only this experiment's G_/D_ files, features and config.
    # Remove the whole experiment directory after publication, but never follow
    # an unexpected symlink supplied by a malformed experiment directory.
    logs = RVC_ROOT / "logs" / exp_name
    if logs.is_dir() and not logs.is_symlink():
        try:
            shutil.rmtree(logs)
            removed["logs"] = 1
        except OSError:
            pass

    entrada = voice / "entrada"
    if entrada.is_dir() and not entrada.is_symlink():
        try:
            shutil.rmtree(entrada)
            removed["entrada"] = 1
        except OSError:
            pass

    # snapshot_checkpoints() stores optional heavy/infer copies under trabajo.
    snapshots = voice / "trabajo" / "ckpt"
    if snapshots.is_dir() and not snapshots.is_symlink():
        try:
            shutil.rmtree(snapshots)
            removed["snapshots"] = 1
        except OSError:
            pass
    return removed


def discard_run_scratch(exp_name: str) -> dict[str, int]:
    """Compatibility wrapper for the post-publish train-artifact GC."""
    return garbage_collect_train_artifacts(exp_name)


def _clear_log_scratch(logs: Path) -> None:
    """Legacy helper retained for callers that want non-directory cleanup."""
    if logs.is_dir() and not logs.is_symlink():
        for path in logs.iterdir():
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                elif path.is_file() or path.is_symlink():
                    path.unlink()
            except OSError:
                pass


def train_voice(exp_name, dataset_files, epochs=None, progress=None):
    if os.environ.get("RVC_TRAIN_SUPERVISOR") == "1":
        return execute_train(
            exp_name, dataset_files, epochs=epochs, progress=progress
        )
    require_rvc_webui()
    require_train_assets()
    exp_name = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in (exp_name or "").strip()
    )
    if not exp_name:
        raise ValueError("Poné un nombre para la voz.")
    total = _epochs(epochs)
    running = train_running(exp_name)
    if running:
        raise ValueError(
            f"{exp_name} sigue entrenando. No lo relances. "
            "Cerrar la ventana no lo corta."
        )
    import occupancy

    snap = occupancy.snapshot()
    if snap is not None:
        raise ValueError(occupancy.blocked_message(snap))
    files = []
    for item in dataset_files or []:
        path = _src_path(item)
        if path:
            files.append(path)
    if not files:
        raise ValueError("Subí al menos un audio de la voz a entrenar.")
    job = train_run.write_job(exp_name, files, total)
    published = train_run.published_path(exp_name)
    if published.is_file():
        published.unlink()
    proc = train_run.spawn_supervisor(job)
    return train_run.wait_supervisor(proc, exp_name, progress, total)


def _log_progress(log_path: Path, progress):
    def report(frac, desc=None):
        line = (desc or "").strip()
        if line:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if progress is None:
            return
        try:
            progress(frac, desc=line)
        except Exception:
            pass

    return report


def execute_train(exp_name, dataset_files, epochs=None, progress=None):
    require_rvc_webui()
    assets = require_train_assets()
    exp_name = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in (exp_name or "").strip()
    )
    if not exp_name:
        raise ValueError("Poné un nombre para la voz.")
    total = _epochs(epochs)

    run_dir = train_run.runs_dir(exp_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    ensure_rvc_outputs_outside()
    _sync_assets(assets)
    _ensure_savee_absolute()
    (RVC_ROOT / "assets" / "weights").mkdir(parents=True, exist_ok=True)
    (RVC_ROOT / "assets" / "indices").mkdir(parents=True, exist_ok=True)
    exp_dir = RVC_ROOT / "logs" / exp_name
    resume = _features_ready(exp_dir)
    if not resume:
        log_path.write_text("", encoding="utf-8")
        from train_prep import prepare_for_train

        dataset_files = prepare_for_train(
            dataset_files, progress=_log_progress(log_path, progress)
        )
    dataset_dir = _prepare_dataset(dataset_files, exp_name)
    from train_prep import discard_prep_copies

    discard_prep_copies()
    py = _engine_python()
    if resume:
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n[resume] features listas en {exp_dir}; no se borra.\n")
    else:
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        _run(
            [
                py,
                str(RVC_ROOT / "rvc" / "train" / "preprocess" / "preprocess.py"),
                str(exp_dir),
                str(dataset_dir),
                str(SR_HZ),
                str(WORKERS),
                "Automatic",
                "True",
                "False",
                "0.5",
                "3.0",
                "0.3",
                "none",
            ],
            log_path,
            progress=progress,
            frac=0.1,
            desc="Cortando audios…",
        )
        _run(
            [
                py,
                str(RVC_ROOT / "rvc" / "train" / "extract" / "extract.py"),
                str(exp_dir),
                "rmvpe",
                str(WORKERS),
                "-",
                str(SR_HZ),
                "custom",
                str(assets.get("hubert") or ""),
                "0",
            ],
            log_path,
            progress=progress,
            frac=0.3,
            desc="F0 y HuBERT…",
        )
    if not (exp_dir / "filelist.txt").is_file():
        _write_filelist_and_config(exp_dir)
    _run(
        [
            py,
            str(RVC_ROOT / "rvc" / "train" / "train.py"),
            exp_name,
            str(SAVE_EVERY_EPOCH),
            str(total),
            str(assets.get("g") or ""),
            str(assets.get("d") or ""),
            "-",
            str(BATCH),
            str(SR_HZ),
            "True",
            "True",
            "False",
            "False",
            "HiFiGAN",
            "False",
        ],
        log_path,
        progress=progress,
        frac=0.7,
        desc=f"Entrenando {total} epochs (CPU)…",
        detach=False,
        exp_name=exp_name,
        total_epochs=total,
    )
    return finish_train_publish(
        exp_name, log_path, total, progress=progress, py=py
    )
