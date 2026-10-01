"""Local RVC conversion. No Hugging Face download, no extra UI options."""
import os
import sys
import time
import types

import numpy as np
import soundfile as sf
from picklescan.scanner import scan_file_path

from app_paths import data_dir

RVC_DIR = os.path.join(data_dir(), "rvc_models")
OUTPUT_DIR = os.path.join(data_dir(), "rvc_output")

_converter = None
_converter_key = None


def _stub_pyworld():
    if "pyworld" in sys.modules:
        return

    def _missing(*_args, **_kwargs):
        raise RuntimeError("Este pitch no está disponible. Se usa rmvpe.")

    stub = types.ModuleType("pyworld")
    stub.harvest = _missing
    stub.stonemask = _missing
    sys.modules["pyworld"] = stub


def _ensure_torchcrepe():
    if "torchcrepe" in sys.modules:
        return
    try:
        import torchcrepe  # noqa: F401
    except Exception:
        def _missing(*_args, **_kwargs):
            raise RuntimeError("Crepe no está disponible. Se usa rmvpe.")

        stub = types.ModuleType("torchcrepe")
        stub.predict = _missing
        filt = types.ModuleType("torchcrepe.filter")
        filt.median = _missing
        filt.mean = _missing
        stub.filter = filt
        sys.modules["torchcrepe"] = stub
        sys.modules["torchcrepe.filter"] = filt


def local_hubert_path():
    folder = os.path.join(RVC_DIR, "hubert_base")
    weights = os.path.join(RVC_DIR, "hubert_base.pt")
    if os.path.isdir(folder):
        return folder
    if os.path.isfile(weights):
        return weights
    return None


def local_rmvpe_path():
    path = os.path.join(RVC_DIR, "rmvpe.pt")
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


def _validate_hubert(path):
    if os.path.isdir(path):
        if os.path.isfile(os.path.join(path, "config.json")):
            return path
        raise ValueError(
            "rvc_models/hubert_base no tiene config.json. "
            "Copiá la carpeta completa del modelo HuBERT, con config.json y los pesos."
        )
    raise ValueError(
        "El motor RVC necesita la carpeta rvc_models/hubert_base "
        "(config.json y los pesos). Un hubert_base.pt suelto no alcanza."
    )


def _reject_unsafe_checkpoint(path):
    try:
        result = scan_file_path(path)
    except Exception as exc:
        raise ValueError("No pude revisar el modelo RVC (.pth).") from exc
    if getattr(result, "issues_count", 0):
        raise ValueError(
            "Ese .pth no es seguro para cargarlo. Usá un modelo RVC de confianza."
        )
    if getattr(result, "scan_err", False):
        raise ValueError("No pude revisar el modelo RVC (.pth).")


def _output_path(audio_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(audio_path))[0] or "voz"
    stamp = time.strftime("%H%M%S")
    dest = os.path.join(OUTPUT_DIR, f"{stem}-rvc-{stamp}.wav")
    if os.path.exists(dest):
        dest = os.path.join(
            OUTPUT_DIR, f"{stem}-rvc-{stamp}-{time.time_ns() % 100000}.wav"
        )
    return dest


def _import_loader():
    _stub_pyworld()
    _ensure_torchcrepe()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    try:
        from infer_rvc_python import BaseLoader
    except Exception as exc:
        raise ValueError(
            "No pude cargar el motor RVC local. "
            "Instalá infer-rvc-python, torch, faiss y librosa. "
            f"Detalle: {exc}"
        ) from exc
    return BaseLoader


def _get_converter(hubert, rmvpe):
    global _converter, _converter_key
    key = (os.path.abspath(hubert), os.path.abspath(rmvpe))
    if _converter is not None and _converter_key == key:
        return _converter
    BaseLoader = _import_loader()
    _converter = BaseLoader(
        only_cpu=False, hubert_path=hubert, rmvpe_path=rmvpe
    )
    _converter_key = key
    return _converter


def _write_converted(audio, sample_rate, dest):
    data = np.asarray(audio)
    if data.ndim > 1:
        data = np.squeeze(data)
    if data.size == 0:
        raise ValueError("La conversión devolvió un audio vacío.")
    if np.issubdtype(data.dtype, np.integer):
        sf.write(dest, data, int(sample_rate), subtype="PCM_16")
    else:
        sf.write(dest, np.asarray(data, dtype=np.float32), int(sample_rate))
    return dest


def convert_voice(audio_path, model_path, index_path=None):
    if not audio_path or not os.path.isfile(audio_path):
        raise ValueError("Falta la voz a convertir.")
    if not model_path or not os.path.isfile(model_path):
        raise ValueError("Falta el modelo RVC (.pth) en disco.")
    hubert, rmvpe = require_support_models()
    hubert = _validate_hubert(hubert)
    if os.path.getsize(rmvpe) <= 0:
        raise ValueError("rvc_models/rmvpe.pt está vacío.")
    if index_path and not os.path.isfile(index_path):
        raise ValueError("No encontré el índice RVC (.index).")
    _reject_unsafe_checkpoint(model_path)
    converter = _get_converter(hubert, rmvpe)
    tag = os.path.abspath(model_path)
    index = index_path or ""
    try:
        converter.apply_conf(
            tag=tag,
            file_model=os.path.abspath(model_path),
            pitch_algo="rmvpe",
            pitch_lvl=0,
            file_index=index,
            index_influence=0.66 if index else 0.0,
            respiration_median_filtering=3,
            envelope_ratio=0.25,
            consonant_breath_protection=0.33,
        )
        converted = converter.generate_from_cache(
            audio_data=os.path.abspath(audio_path), tag=tag
        )
    except ValueError as exc:
        text = str(exc)
        if text[:1].isupper() and any(
            text.startswith(prefix)
            for prefix in (
                "Falta",
                "No pude",
                "No encontré",
                "Ese ",
                "El motor",
                "rvc_models",
                "La conversión",
            )
        ):
            raise
        raise ValueError(f"No pude convertir la voz: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"No pude convertir la voz: {exc}") from exc
    if not isinstance(converted, tuple) or len(converted) != 2:
        raise ValueError("La conversión no devolvió audio.")
    samples, sample_rate = converted
    dest = _output_path(audio_path)
    _write_converted(samples, sample_rate, dest)
    if os.path.abspath(dest) == os.path.abspath(audio_path):
        raise ValueError("La conversión no generó un archivo nuevo.")
    return dest
