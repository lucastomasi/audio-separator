"""Official RVC-WebUI training on CPU. Minimal + robust."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from library import ensure_dirs, register, rvc_support_dir

APP_ROOT = Path(__file__).resolve().parent
RVC_ROOT = APP_ROOT / "third_party" / "RVC-WebUI"
SR = "40k"
SR_HZ = 40000
VERSION = "v2"
EPOCHS_DEFAULT = 50
BATCH = 1
WORKERS = 2


def _epochs():
    raw = os.environ.get("RVC_TRAIN_EPOCHS", str(EPOCHS_DEFAULT))
    try:
        value = int(raw)
    except ValueError:
        value = EPOCHS_DEFAULT
    return max(1, value)


def require_rvc_webui():
    if not (RVC_ROOT / "train" / "train.py").is_file():
        raise ValueError(
            "Falta third_party/RVC-WebUI (clone oficial de entrenamiento)."
        )


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
            "Faltan pesos para entrenar. Cargalos en el paso Entrenar: "
            + ", ".join(missing)
        )
    return {"hubert": hubert, "rmvpe": rmvpe, "g": g, "d": d}


def _run(cmd, log_path: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(RVC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["RVC_AUDIO_FORCE_CPU"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        result = subprocess.run(
            cmd,
            cwd=str(RVC_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            log.write(result.stdout)
        if result.stderr:
            log.write(result.stderr)
        log.flush()
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(err[-1500:] if err else f"Falló: {' '.join(cmd)}")


def _sync_assets(assets):
    hubert_dst = RVC_ROOT / "assets" / "hubert_base"
    rmvpe_dst = RVC_ROOT / "assets" / "rmvpe"
    pre_dst = RVC_ROOT / "assets" / "pretrained_v2"
    for folder in (hubert_dst, rmvpe_dst, pre_dst):
        folder.mkdir(parents=True, exist_ok=True)

    hubert = Path(assets["hubert"])
    # Replace any stale classic .pt so Transformers load sees a clean dir.
    # Prefer safetensors on torch<2.6 (Intel Mac): drop .bin if .safetensors present.
    if hubert_dst.exists():
        for stale in hubert_dst.iterdir():
            if stale.name in ("hubert_base.pt", "pytorch_model.bin.bak"):
                stale.unlink()
    has_sft = (hubert / "model.safetensors").is_file()
    for item in hubert.iterdir():
        if item.name.endswith(".bak"):
            continue
        if has_sft and item.name == "pytorch_model.bin":
            continue
        target = hubert_dst / item.name
        if item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
    # Intel Mac / torch 2.2: transformers refuses torch.load on .bin.
    for junk in ("pytorch_model.bin", "pytorch_model.bin.bak", "hubert_base.pt"):
        left = hubert_dst / junk
        if left.is_file():
            left.unlink()
    if not (hubert_dst / "model.safetensors").is_file() and not (
        hubert_dst / "pytorch_model.bin"
    ).is_file():
        raise RuntimeError(
            "Tras sync no hay pesos HuBERT en assets/hubert_base "
            "(model.safetensors)."
        )

    shutil.copy2(assets["rmvpe"], rmvpe_dst / "rmvpe.pt")
    shutil.copy2(assets["g"], pre_dst / "f0G40k.pth")
    shutil.copy2(assets["d"], pre_dst / "f0D40k.pth")


def _prepare_dataset(dataset_files, exp_name):
    dataset_dir = APP_ROOT / "library" / "train_data" / exp_name
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in dataset_files or []:
        if not src:
            continue
        path = src if isinstance(src, str) else getattr(src, "name", None)
        if not path or not os.path.isfile(path):
            continue
        ext = os.path.splitext(path)[1].lower() or ".wav"
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
    gt = exp_dir / "0_gt_wavs"
    feat = exp_dir / "3_feature768"
    f0 = exp_dir / "2a_f0"
    f0nsf = exp_dir / "2b-f0nsf"
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

    # Match official WebUI: 40k always uses v1/{sr}.json template (no v2/40k.json).
    if VERSION == "v1" or SR == "40k":
        template = RVC_ROOT / "configs" / "v1" / f"{SR}.json"
    else:
        template = RVC_ROOT / "configs" / "v2" / f"{SR}.json"
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
    if not search:
        return None
    exp = exp_name.lower()
    # Prefer added_* (has vectors) over trained_* (empty IVF shell).
    added = [
        p for p in search if "added" in p.name.lower() and exp in p.name.lower()
    ]
    if added:
        return max(added, key=lambda p: p.stat().st_size)
    named = [p for p in search if exp in p.name.lower()]
    pool = named or list(search)
    return max(pool, key=lambda p: p.stat().st_size)


def _ensure_inference_weight(exp_name, log_path: Path | None = None):
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
    from train.process_ckpt import savee
    from train.utils import HParams

    config = json.loads(config_path.read_text(encoding="utf-8"))
    hps = HParams(**config)
    ckpt = torch.load(str(g_candidates[0]), map_location="cpu")
    weight = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    prev = os.getcwd()
    try:
        # savee writes relative "assets/weights/<name>.pth"
        os.chdir(str(RVC_ROOT))
        _log(f"[weight] fallback cwd={os.getcwd()} from {g_candidates[0].name}")
        msg = savee(weight, SR, 1, exp_name, _epochs(), VERSION, hps)
        _log(f"[weight] savee: {msg}")
    finally:
        os.chdir(prev)
    out = weights_dir / f"{exp_name}.pth"
    if not out.is_file():
        raise RuntimeError(f"No se pudo exportar weight de inferencia: {msg}")
    _log(f"[weight] escrito {out} ({out.stat().st_size} bytes)")
    return out


def _publish_voice_to_library(exp_name, pth: Path, index: Path | None, log_path: Path):
    """Always copy inference pth + added index into library/models/rvc_voices/.

    Convert / dropdown only read that folder so runtime assets/weights cannot
    drift from what the UI lists.
    """
    dest_pth = register("rvc_voices", str(pth), f"{exp_name}.pth")
    dest_index = None
    if index is not None and index.is_file():
        dest_index = register("rvc_voices", str(index), f"{exp_name}.index")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(
            f"[library] pth -> {dest_pth['path']} "
            f"({Path(dest_pth['path']).stat().st_size} bytes)\n"
        )
        if dest_index:
            log.write(f"[library] index -> {dest_index['path']}\n")
        else:
            log.write("[library] sin index (convert igual puede usar el .pth)\n")
    return dest_pth["path"], (dest_index["path"] if dest_index else None)


def train_voice(exp_name, dataset_files):
    require_rvc_webui()
    assets = require_train_assets()
    exp_name = "".join(
        c if c.isalnum() or c in "-_" else "_" for c in (exp_name or "").strip()
    )
    if not exp_name:
        raise ValueError("Poné un nombre para la voz.")

    run_dir = APP_ROOT / "library" / "train_runs" / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"

    _sync_assets(assets)
    # Small inference weights are written here by process_ckpt.savee
    (RVC_ROOT / "assets" / "weights").mkdir(parents=True, exist_ok=True)
    (RVC_ROOT / "assets" / "indices").mkdir(parents=True, exist_ok=True)
    dataset_dir = _prepare_dataset(dataset_files, exp_name)
    exp_dir = RVC_ROOT / "logs" / exp_name
    if exp_dir.exists():
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    # Always -m from RVC_ROOT (cwd + PYTHONPATH). Script paths put
    # train/ on sys.path[0] and break `import train` with circular imports.
    if log_path.is_file():
        log_path.write_text("", encoding="utf-8")
    _run(
        [
            py,
            "-m",
            "train.preprocess",
            str(dataset_dir),
            str(SR_HZ),
            str(WORKERS),
            str(exp_dir),
            "True",
            "3.7",
        ],
        log_path,
    )
    _run(
        [
            py,
            "-m",
            "train.dataset.extract_f0",
            "cpu",
            str(exp_dir),
            str(WORKERS),
            "rmvpe",
        ],
        log_path,
    )
    _run(
        [
            py,
            "-m",
            "train.dataset.extract_hubert_feature",
            "cpu",
            "1",
            "0",
            str(exp_dir),
            VERSION,
            "false",
        ],
        log_path,
    )
    _write_filelist_and_config(exp_dir)
    _run(
        [
            py,
            "-m",
            "train.train",
            "-e",
            exp_name,
            "-sr",
            SR,
            "-f0",
            "1",
            "-bs",
            str(BATCH),
            "-te",
            str(_epochs()),
            "-se",
            str(min(5, _epochs())),
            "-pg",
            "assets/pretrained_v2/f0G40k.pth",
            "-pd",
            "assets/pretrained_v2/f0D40k.pth",
            "-l",
            "1",
            "-c",
            "0",
            "-sw",
            "1",
            "-v",
            VERSION,
        ],
        log_path,
    )
    indices_dir = RVC_ROOT / "assets" / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run(
            [
                py,
                "-m",
                "train.train_index",
                exp_name,
                VERSION,
                str(indices_dir),
                str(WORKERS),
                "single",
            ],
            log_path,
        )
    except RuntimeError as exc:
        # index is optional for convert; keep pth if train succeeded
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"\n[index] omitido: {exc}\n")

    # savee writes relative assets/weights/<name>.pth — spawn cwd is RVC_ROOT.
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"[weight] spawn cwd esperado={RVC_ROOT}\n")
    prev_path = sys.path[:]
    if str(RVC_ROOT) not in sys.path:
        sys.path.insert(0, str(RVC_ROOT))
    try:
        pth = _ensure_inference_weight(exp_name, log_path=log_path)
    finally:
        sys.path[:] = prev_path
    if not pth:
        raise RuntimeError(
            "Train terminó sin .pth de inferencia. Orden a revisar: "
            f"{RVC_ROOT / 'assets' / 'weights' / (exp_name + '.pth')}, "
            "fallback _ensure_inference_weight desde G_*.pth, "
            f"cwd del spawn={RVC_ROOT}. Log: library/train_runs/{exp_name}/train.log"
        )
    index = _find_index(exp_name)
    return _publish_voice_to_library(exp_name, Path(pth), index, log_path)
