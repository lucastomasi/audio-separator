"""Local RVC conversion. No Hugging Face download, no extra UI options."""
import os
import sys
import types

from picklescan.scanner import scan_file_path

from exports import copy_to_downloads

from library import rvc_support_dir

RVC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rvc_models")


def _support_roots():
    roots = [rvc_support_dir(), RVC_DIR]
    return [root for root in roots if root]


def local_hubert_path():
    for root in _support_roots():
        for name in ("hubert_base", "hubert_base.pt"):
            path = os.path.join(root, name)
            if os.path.exists(path):
                return path
    return None


def local_rmvpe_path():
    for root in _support_roots():
        path = os.path.join(root, "rmvpe.pt")
        if os.path.isfile(path):
            return path
    return None

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


def require_support_models():
    hubert = local_hubert_path()
    rmvpe = local_rmvpe_path()
    missing = []
    if not hubert:
        missing.append("library/models/rvc/hubert_base")
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
    issues = getattr(result, "issues", None) or getattr(result, "globals", None)
    if issues:
        raise ValueError("El archivo del modelo no pasó la revisión de seguridad.")


def get_converter():
    global _converter
    if _converter is not None:
        return _converter
    hubert, rmvpe = require_support_models()
    _stub_pyworld()
    from infer_rvc_python import BaseLoader

    _converter = BaseLoader(
        only_cpu=True,
        hubert_path=hubert,
        rmvpe_path=rmvpe,
        preload_models=False,
    )
    return _converter


def convert_voice(audio_path, model_path, index_path=None):
    if not audio_path or not os.path.isfile(audio_path):
        raise ValueError("Falta la voz a convertir.")
    if not model_path or not os.path.isfile(model_path):
        raise ValueError("Falta el modelo RVC (.pth) en disco.")
    _scan_model(model_path)
    converter = get_converter()
    tag = "local_voice"
    converter.apply_conf(
        tag=tag,
        file_model=model_path,
        pitch_algo="rmvpe",
        pitch_lvl=0,
        file_index=index_path or "",
        index_influence=0.66,
        respiration_median_filtering=3,
        envelope_ratio=0.25,
        consonant_breath_protection=0.33,
        resample_sr=0,
    )
    results = converter(
        [audio_path],
        tag,
        overwrite=False,
        parallel_workers=1,
        type_output="wav",
        show_progress=False,
    )
    if not results:
        raise ValueError("La conversión no produjo audio.")
    out_path = results[0] if isinstance(results, (list, tuple)) else results
    if not out_path or not os.path.isfile(out_path):
        raise ValueError("La conversión no produjo audio.")
    _, copied = copy_to_downloads([out_path], ["voz_rvc"])
    return copied[0] if copied else os.path.abspath(out_path)
