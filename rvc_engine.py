"""Local RVC conversion. No Hugging Face download, no extra UI options."""
import os
import sys
import types

from picklescan.scanner import scan_file_path

from exports import copy_to_downloads

RVC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rvc_models")
HUBERT_CANDIDATES = (
    os.path.join(RVC_DIR, "hubert_base"),
    os.path.join(RVC_DIR, "hubert_base.pt"),
)
RMVPE_CANDIDATES = (os.path.join(RVC_DIR, "rmvpe.pt"),)

_converter = None


def _stub_pyworld():
    if "pyworld" in sys.modules:
        return

    def _missing(*_args, **_kwargs):
        raise RuntimeError("Este pitch no está disponible. Se usa rmvpe.")

    stub = types.ModuleType("pyworld")
    stub.harvest = _missing
    stub.stonemask = _missing
    sys.modules["pyworld"] = stub


def local_hubert_path():
    for path in HUBERT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def local_rmvpe_path():
    for path in RMVPE_CANDIDATES:
        if os.path.isfile(path):
            return path
    return None


def require_support_models():
    hubert = local_hubert_path()
    rmvpe = local_rmvpe_path()
    missing = []
    if not hubert:
        missing.append("rvc_models/hubert_base")
    if not rmvpe:
        missing.append("rvc_models/rmvpe.pt")
    if missing:
        raise ValueError(
            "Faltan los modelos locales de RVC: " + ", ".join(missing)
        )
    return hubert, rmvpe


def convert_voice(audio_path, model_path, index_path=None):
    if not audio_path or not os.path.isfile(audio_path):
        raise ValueError("Falta la voz a convertir.")
    if not model_path or not os.path.isfile(model_path):
        raise ValueError("Falta el modelo RVC (.pth) en disco.")
    return audio_path
