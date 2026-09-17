"""Voice conversion: validate paths, scan .pth, run isolated engine."""
import os

from picklescan.scanner import scan_file_path

from exports import copy_to_downloads
from library import rvc_support_dir
from vc_runner import ensure_vc_engine, run_vc_infer, vc_python, vc_root

RVC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rvc_models")


def _support_roots():
    roots = [rvc_support_dir(), RVC_DIR]
    return [root for root in roots if root]


def _is_transformers_hubert_dir(path):
    if not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    return any(
        os.path.isfile(os.path.join(path, name))
        for name in ("model.safetensors", "pytorch_model.bin")
    )


def local_hubert_path():
    for root in _support_roots():
        path = os.path.join(root, "hubert_base")
        if _is_transformers_hubert_dir(path):
            return path
    return None


def local_rmvpe_path():
    for root in _support_roots():
        path = os.path.join(root, "rmvpe.pt")
        if os.path.isfile(path):
            return path
    return None


def require_support_models():
    hubert = local_hubert_path()
    rmvpe = local_rmvpe_path()
    missing = []
    if not hubert:
        missing.append(
            "library/models/rvc/hubert_base/ (config.json + model.safetensors)"
        )
    if not rmvpe:
        missing.append("library/models/rvc/rmvpe.pt")
    if missing:
        raise ValueError(
            "Faltan los modelos locales de RVC: " + ", ".join(missing)
        )
    return hubert, rmvpe


def _scan_model(model_path):
    try:
        result = scan_file_path(model_path)
    except Exception as exc:
        raise ValueError(f"No se pudo revisar el modelo: {exc}") from exc
    infected = getattr(result, "infected_files", 0) or 0
    issues_count = getattr(result, "issues_count", 0) or 0
    suspicious = getattr(result, "suspicious_count", 0) or 0
    if infected or issues_count or suspicious:
        raise ValueError("El archivo del modelo no pasó la revisión de seguridad.")


def convert_voice(
    audio_path,
    model_path,
    index_path=None,
    *,
    pitch=0,
    index_rate=0.66,
    f0_method="rmvpe",
    protect=0.33,
    filter_radius=3,
    rms_mix_rate=0.25,
    copy_downloads=True,
):
    """Convert with a library .pth (+ optional .index) via isolated engine."""
    if isinstance(audio_path, dict):
        audio_path = audio_path.get("path") or audio_path.get("name")
    if isinstance(model_path, dict):
        model_path = model_path.get("path") or model_path.get("name")
    if not audio_path or not os.path.isfile(str(audio_path)):
        raise ValueError(
            "Falta la pista de voz a convertir (archivo inexistente)."
        )
    if not model_path or not os.path.isfile(str(model_path)):
        raise ValueError("Falta el modelo RVC (.pth) en disco.")
    _scan_model(str(model_path))
    produced = run_vc_infer(
        str(audio_path),
        str(model_path),
        index_path,
        pitch=pitch,
        index_rate=index_rate,
        f0_method=f0_method,
        protect=protect,
    )
    if not copy_downloads:
        return produced
    _, copied = copy_to_downloads([produced], ["voz_rvc"])
    return copied[0] if copied else produced
