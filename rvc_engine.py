"""Local RVC conversion. No Hugging Face download, no extra UI options."""
import hashlib
import os
import shutil
import sys
import tempfile
import time
import types

import numpy as np
import soundfile as sf
from picklescan.scanner import scan_file_path

from app_paths import data_dir

RVC_DIR = os.path.join(data_dir(), "rvc_models")
OUTPUT_DIR = os.path.join(data_dir(), "rvc_output")
MAX_INDEX_BYTES = 512 * 1024 * 1024
_PICKLE_SUFFIXES = (".bin", ".pt", ".pth")

_converter = None
_converter_key = None
_weight_tmpdir = None


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


def _staging_dir():
    global _weight_tmpdir
    if _weight_tmpdir is None:
        _weight_tmpdir = tempfile.TemporaryDirectory(prefix="as-rvc-")
        try:
            os.chmod(_weight_tmpdir.name, 0o700)
        except OSError:
            pass
    return _weight_tmpdir.name


def _pickle_files(directory):
    found = []
    for name in os.listdir(directory):
        if name.lower().endswith(_PICKLE_SUFFIXES):
            found.append(os.path.join(directory, name))
    return found


def _validate_hubert(path):
    if not os.path.isdir(path):
        raise ValueError(
            "El motor RVC necesita la carpeta rvc_models/hubert_base "
            "(config.json y los pesos). Un hubert_base.pt suelto no alcanza."
        )
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise ValueError(
            "rvc_models/hubert_base no tiene config.json. "
            "Copiá la carpeta completa del modelo HuBERT, con config.json y los pesos."
        )
    has_safe = os.path.isfile(os.path.join(path, "model.safetensors"))
    pickles = _pickle_files(path)
    if not has_safe and not pickles:
        raise ValueError("Faltan los pesos de HuBERT.")
    for pickle_path in pickles:
        _reject_unsafe_checkpoint(pickle_path)
    return os.path.realpath(path)


def _reject_unsafe_checkpoint(path):
    try:
        result = scan_file_path(path)
    except Exception as exc:
        raise ValueError("No pude revisar el modelo.") from exc
    if getattr(result, "issues_count", 0):
        raise ValueError(
            "Ese modelo no es seguro para cargarlo. Usá archivos de confianza."
        )
    if getattr(result, "scan_err", False):
        raise ValueError("No pude revisar el modelo.")


def _checked_copy(path):
    path = os.path.realpath(path)
    _reject_unsafe_checkpoint(path)
    digest = hashlib.sha256(path.encode("utf-8")).hexdigest()[:16]
    dest = os.path.join(_staging_dir(), f"{digest}-{os.path.basename(path)}")
    shutil.copy2(path, dest)
    try:
        os.chmod(dest, 0o600)
    except OSError:
        pass
    _reject_unsafe_checkpoint(dest)
    return dest


def _checked_hubert_dir(path):
    path = _validate_hubert(path)
    dest = os.path.join(_staging_dir(), "hubert_base")
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    shutil.copytree(path, dest)
    return _validate_hubert(dest)


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
    hubert = _checked_hubert_dir(hubert)
    if os.path.getsize(rmvpe) <= 0:
        raise ValueError("rvc_models/rmvpe.pt está vacío.")
    rmvpe = _checked_copy(rmvpe)
    if index_path:
        if not os.path.isfile(index_path):
            raise ValueError("No encontré el índice RVC (.index).")
        if os.path.getsize(index_path) > MAX_INDEX_BYTES:
            raise ValueError("Ese índice es demasiado grande.")
    model_path = _checked_copy(model_path)
    converter = _get_converter(hubert, rmvpe)
    tag = os.path.abspath(model_path)
    index = os.path.realpath(index_path) if index_path else ""
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
        raise ValueError("No pude convertir la voz.") from exc
    except Exception as exc:
        raise ValueError("No pude convertir la voz.") from exc
    if not isinstance(converted, tuple) or len(converted) != 2:
        raise ValueError("La conversión no devolvió audio.")
    samples, sample_rate = converted
    dest = _output_path(audio_path)
    _write_converted(samples, sample_rate, dest)
    if os.path.abspath(dest) == os.path.abspath(audio_path):
        raise ValueError("La conversión no generó un archivo nuevo.")
    return dest
