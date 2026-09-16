"""Local RVC conversion. No Hugging Face download, no extra UI options."""
import gc
import os
import sys
import types

# Before torch/OpenMP load: avoid thread+OpenMP deadlocks / segfaults on Intel Mac.
# KMP_DUPLICATE_LIB_OK: desktop.py + CLI both load libiomp → intermittent SIGSEGV.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    infected = getattr(result, "infected_files", 0) or 0
    issues_count = getattr(result, "issues_count", 0) or 0
    suspicious = getattr(result, "suspicious_count", 0) or 0
    if infected or issues_count or suspicious:
        raise ValueError("El archivo del modelo no pasó la revisión de seguridad.")


def _patch_inline_infer(loader):
    """infer_rvc_python always spawns Thread(s); on Mac Intel that deadlocks
    (main join + worker lock + libiomp). Run infer on the main thread instead.
    """

    def run_threads_inline(self, threads):
        for thread in threads:
            target = thread._target
            args = thread._args or ()
            kwargs = thread._kwargs or {}
            if target is not None:
                target(*args, **kwargs)
        gc.collect()

    loader.run_threads = types.MethodType(run_threads_inline, loader)
    return loader


def get_converter():
    global _converter
    if _converter is not None:
        return _converter
    hubert, rmvpe = require_support_models()
    _stub_pyworld()
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:
        pass
    from infer_rvc_python import BaseLoader

    _converter = _patch_inline_infer(
        BaseLoader(
            only_cpu=True,
            hubert_path=hubert,
            rmvpe_path=rmvpe,
            preload_models=False,
        )
    )
    return _converter


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
    """Convert with a library .pth (+ optional .index). Mac Intel: CPU, no half."""
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
        pitch_algo=f0_method or "rmvpe",
        pitch_lvl=int(pitch),
        file_index=index_path or "",
        index_influence=float(index_rate),
        respiration_median_filtering=int(filter_radius),
        envelope_ratio=float(rms_mix_rate),
        consonant_breath_protection=float(protect),
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
    if not copy_downloads:
        return os.path.abspath(out_path)
    _, copied = copy_to_downloads([out_path], ["voz_rvc"])
    return copied[0] if copied else os.path.abspath(out_path)
