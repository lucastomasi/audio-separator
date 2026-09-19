"""Download public RVC support weights into library/models/rvc/ (lite install).

Sources: Hugging Face lj1995/VoiceConversionWebUI (public).
Does not upload anything; only downloads.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from library import ensure_dirs, rvc_support_dir

REPO = "lj1995/VoiceConversionWebUI"
WEBUI_GIT = (
    "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"
)

# Relative paths inside the HF repo → local names under library/models/rvc/
# Each local name → candidate paths inside the HF repo (first that exists wins).
HF_FILE_CANDIDATES = {
    "hubert_base.pt": ("hubert_base.pt",),
    "rmvpe.pt": ("rmvpe.pt", "rmvpe/rmvpe.pt", "assets/rmvpe/rmvpe.pt"),
    "f0G40k.pth": (
        "pretrained_v2/f0G40k.pth",
        "pretrained/f0G40k.pth",
        "assets/pretrained_v2/f0G40k.pth",
    ),
    "f0D40k.pth": (
        "pretrained_v2/f0D40k.pth",
        "pretrained/f0D40k.pth",
        "assets/pretrained_v2/f0D40k.pth",
    ),
}

HF_HUBERT_DIR = {
    "hubert_base/config.json": "config.json",
    "hubert_base/preprocessor_config.json": "preprocessor_config.json",
    "hubert_base/pytorch_model.bin": "pytorch_model.bin",
}


def _root() -> Path:
    ensure_dirs()
    return Path(rvc_support_dir())


def missing_rvc_assets() -> list[str]:
    """Human-readable list of missing install pieces."""
    root = _root()
    missing = []
    for local in HF_FILE_CANDIDATES:
        path = root / local
        if local == "hubert_base.pt":
            if not _hubert_pt_is_fairseq(path):
                missing.append(local)
            continue
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(local)
    hubert_dir = root / "hubert_base"
    need = ("config.json", "preprocessor_config.json")
    for name in need:
        if not (hubert_dir / name).is_file():
            missing.append(f"hubert_base/{name}")
    has_weights = (hubert_dir / "model.safetensors").is_file() or (
        hubert_dir / "pytorch_model.bin"
    ).is_file()
    if not has_weights:
        missing.append("hubert_base/model.safetensors")
    return missing


def rvc_assets_ready() -> bool:
    return not missing_rvc_assets()


def rvc_webui_root() -> Path:
    return Path(__file__).resolve().parent / "third_party" / "RVC-WebUI"


def ensure_rvc_webui(log=None) -> str | None:
    """Clone official WebUI once (free git). Train needs it; app stays up if git fails."""
    dest = rvc_webui_root()
    train_py = dest / "train" / "train.py"
    if train_py.is_file():
        if log:
            log("OK RVC-WebUI")
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    if log:
        log("Clonando RVC-WebUI (gratis, una vez)…")
    import subprocess

    result = subprocess.run(
        ["git", "clone", "--depth", "1", WEBUI_GIT, str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not train_py.is_file():
        err = (result.stderr or result.stdout or "").strip()[:200]
        raise RuntimeError(
            "Sin red no pude clonar RVC-WebUI. Entrenar queda apagado "
            "hasta que haya cupo o red. " + err
        )
    if log:
        log("Listo RVC-WebUI")
    return "RVC-WebUI"


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        try:
            os.symlink(src, dest)
        except OSError:
            shutil.copy2(src, dest)


def _download(repo_file: str, dest: Path, cache_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    if REPO.startswith("/"):
        raise RuntimeError(f"repo_id inválido: {REPO}")
    token = None
    try:
        from gpu_secrets import get as _secret

        token = _secret("HF_TOKEN")
    except Exception:
        token = None
    cached = hf_hub_download(
        repo_id=REPO,
        filename=repo_file,
        cache_dir=str(cache_dir) if cache_dir else None,
        resume_download=True,
        token=token or None,
    )
    _link_or_copy(Path(cached), dest)


def _torch_load(path: Path, *, allow_unsafe: bool = False):
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception:
        if not allow_unsafe:
            raise
        return torch.load(str(path), map_location="cpu", weights_only=False)


def _hubert_pt_is_fairseq(path: Path) -> bool:
    """True if hubert_base.pt is ContentVec/fairseq, not a misnamed RVC .pth."""
    if not path.is_file() or path.stat().st_size < 80_000_000:
        return False
    try:
        state = _torch_load(path, allow_unsafe=True)
    except Exception:
        return False
    if not isinstance(state, dict):
        return False
    # RVC infer weights look like this — reject them.
    if "weight" in state and "sr" in state and "f0" in state:
        return False
    return "model" in state or "args" in state or "best_loss" in state


def _ensure_safetensors(hubert_dir: Path) -> None:
    """Torch 2.2 + transformers recent: prefer safetensors over .bin."""
    sft = hubert_dir / "model.safetensors"
    bin_path = hubert_dir / "pytorch_model.bin"
    if sft.is_file() and sft.stat().st_size > 0:
        if bin_path.is_file():
            bin_path.unlink()
        return
    if not bin_path.is_file():
        raise FileNotFoundError(bin_path)
    from safetensors.torch import save_file

    state = _torch_load(bin_path, allow_unsafe=True)
    if isinstance(state, dict) and "state_dict" in state and len(state) < 5:
        state = state["state_dict"]
    tensors = {
        k: v.detach().contiguous()
        for k, v in state.items()
        if hasattr(v, "detach")
    }
    save_file(tensors, str(sft))
    bin_path.unlink()


def install_rvc_assets(log=None) -> list[str]:
    """Download missing weights. Returns list of files written/updated."""
    def _log(msg: str):
        if log:
            log(msg)

    root = _root()
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    cache.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def _fetch_named(local_name: str, candidates: tuple) -> str | None:
        dest = root / local_name
        if local_name == "hubert_base.pt" and _hubert_pt_is_fairseq(dest):
            _log(f"OK {local_name} (fairseq)")
            return None
        if (
            local_name != "hubert_base.pt"
            and dest.is_file()
            and dest.stat().st_size > 0
        ):
            _log(f"OK {local_name}")
            return None
        if local_name == "hubert_base.pt" and dest.is_file():
            bad = dest.with_suffix(dest.suffix + ".invalid")
            _log(f"hubert_base.pt no es fairseq; renombro a {bad.name}")
            dest.replace(bad)
        last_err = None
        for repo_file in candidates:
            try:
                _log(f"Descargando {local_name} ← {repo_file}…")
                _download(repo_file, dest, cache)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                _log(f"  no en {repo_file}: {exc}")
        if last_err is not None or not dest.is_file():
            raise RuntimeError(f"No se pudo bajar {local_name}: {last_err}")
        _log(f"Listo {local_name} ({dest.stat().st_size} bytes)")
        return local_name

    from concurrent.futures import ThreadPoolExecutor, as_completed

    jobs = list(HF_FILE_CANDIDATES.items())
    with ThreadPoolExecutor(max_workers=min(4, len(jobs) or 1)) as pool:
        futs = [pool.submit(_fetch_named, name, cands) for name, cands in jobs]
        for fut in as_completed(futs):
            got = fut.result()
            if got:
                written.append(got)

    hubert_dir = root / "hubert_base"
    hubert_dir.mkdir(parents=True, exist_ok=True)
    for repo_file, local_name in HF_HUBERT_DIR.items():
        dest = hubert_dir / local_name
        # Skip bin download if safetensors already present
        if local_name == "pytorch_model.bin" and (
            hubert_dir / "model.safetensors"
        ).is_file():
            continue
        if dest.is_file() and dest.stat().st_size > 0:
            _log(f"OK hubert_base/{local_name}")
            continue
        if local_name == "pytorch_model.bin" and (
            hubert_dir / "model.safetensors"
        ).is_file():
            continue
        _log(f"Descargando hubert_base/{local_name}…")
        _download(repo_file, dest, cache)
        written.append(f"hubert_base/{local_name}")

    _log("Convirtiendo HuBERT a safetensors (Mac Intel / torch 2.2)…")
    _ensure_safetensors(hubert_dir)
    if (hubert_dir / "model.safetensors").is_file():
        written.append("hubert_base/model.safetensors")

    # mute samples for train (small)
    try:
        from pathlib import Path as P

        rvc_root = P(__file__).resolve().parent / "third_party" / "RVC-WebUI"
        mute_dir = rvc_root / "logs" / "mute"
        if rvc_root.is_dir() and not mute_dir.is_dir():
            _log("Descargando mute.zip…")
            import zipfile

            zpath = cache / "mute.zip"
            if not zpath.is_file():
                _download("mute.zip", zpath, cache)
            mute_dir.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(mute_dir.parent)
            written.append("logs/mute")
    except Exception as exc:
        _log(f"mute opcional omitido: {exc}")

    try:
        got_webui = ensure_rvc_webui(log=_log)
        if got_webui:
            written.append(got_webui)
    except Exception as exc:
        _log(str(exc))

    left = missing_rvc_assets()
    if left:
        raise RuntimeError("Siguen faltando: " + ", ".join(left))
    _log("Instalación de pesos RVC completa.")
    return written


if __name__ == "__main__":
    def _print(msg):
        print(msg, flush=True)

    if rvc_assets_ready():
        print("Ya están todos los pesos RVC.")
    else:
        print("Faltan:", ", ".join(missing_rvc_assets()))
        install_rvc_assets(log=_print)
