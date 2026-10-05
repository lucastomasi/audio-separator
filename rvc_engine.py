"""Local RVC conversion.

model_fetch downloads HuBERT and rmvpe before this runs. Conversion stays
offline so those files are not pulled again. The voice .pth is the user's.
A training generator (G_*.pth) is packed into an inference checkpoint when
the training config.json sits next to it.
"""
import json
import os
import re
import sys
import types

import numpy as np
import soundfile as sf
from picklescan.scanner import scan_file_path

from app_paths import data_dir

RVC_DIR = os.path.join(data_dir(), "rvc_models")
OUTPUT_DIR = os.path.join(data_dir(), "rvc_output")

_converter = None
_converter_key = None
_cpu_only = False
last_note = ""

_SONG_SUFFIX = re.compile(
    r"-(?:voz-rvc(?: \(\d+\))?|voz-\d{6}(?:-\d+)?|instrumental-\d{6}(?:-\d+)?|rvc-\d{6}(?:-\d+)?|remix)$"
)


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
            "La descarga quedó incompleta. Volvé a bajar los modelos."
        )
    raise ValueError(
        "El motor RVC necesita la carpeta rvc_models/hubert_base "
        "(config.json y los pesos). Un hubert_base.pt suelto no alcanza."
    )


def _is_dangerous_global(item):
    safety = getattr(item, "safety", None)
    label = getattr(safety, "value", None) or getattr(safety, "name", None) or safety
    return str(label).lower() == "dangerous"


def _reject_unsafe_checkpoint(path):
    try:
        result = scan_file_path(path)
    except Exception as exc:
        raise ValueError("No pude revisar el modelo RVC (.pth).") from exc
    if getattr(result, "scan_err", False):
        raise ValueError("No pude revisar el modelo RVC (.pth).")
    found = getattr(result, "globals", None)
    if found is None:
        if getattr(result, "issues_count", 0):
            raise ValueError(
                "Ese .pth no es seguro para cargarlo. Usá un modelo RVC de confianza."
            )
        return
    if any(_is_dangerous_global(item) for item in found):
        raise ValueError(
            "Ese .pth no es seguro para cargarlo. Usá un modelo RVC de confianza."
        )


def song_stem(path):
    name = os.path.splitext(os.path.basename(str(path)))[0]
    previous = None
    while name and name != previous:
        previous = name
        name = _SONG_SUFFIX.sub("", name)
    name = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("._")
    return (name or "audio")[:60]


def _output_path(audio_path):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = song_stem(audio_path)
    dest = os.path.join(OUTPUT_DIR, f"{stem}-voz-rvc.wav")
    index = 1
    while os.path.exists(dest):
        index += 1
        dest = os.path.join(OUTPUT_DIR, f"{stem}-voz-rvc ({index}).wav")
    return dest


def match_index(model_path, index_paths):
    """Pick the .index that belongs to this voice, or ''."""
    if not model_path:
        return ""
    stem = os.path.splitext(os.path.basename(str(model_path)))[0].lower()
    best_path = ""
    best_score = 0
    for path in index_paths:
        if not path:
            continue
        name = os.path.splitext(os.path.basename(path))[0].lower()
        score = 0
        if name == stem:
            score = 100
        elif name.endswith("_" + stem) or name.endswith(stem):
            score = 80
        elif len(stem) >= 4 and f"_{stem}_" in f"_{name}_":
            score = 60
        elif len(stem) >= 4 and stem in name:
            score = 40
        if score > best_score:
            best_score = score
            best_path = path
    return best_path


def _looks_like_weights(mapping):
    if not isinstance(mapping, dict) or not mapping:
        return False
    return any(isinstance(key, str) and key.startswith("enc_p.") for key in mapping)


def _extract_weight(loaded):
    if not isinstance(loaded, dict):
        return None
    if _looks_like_weights(loaded.get("weight")):
        return loaded["weight"]
    for key in ("model", "generator", "state_dict", "net_g"):
        if _looks_like_weights(loaded.get(key)):
            return loaded[key]
    if _looks_like_weights(loaded):
        return loaded
    return None


def _without_posterior(weight):
    return {key: value for key, value in weight.items() if "enc_q" not in key}


def _phone_dim(weight):
    tensor = weight.get("enc_p.emb_phone.weight")
    shape = getattr(tensor, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[1])


def _version_of(loaded, weight):
    dim = _phone_dim(weight)
    inferred = "v1" if dim == 256 else "v2" if dim == 768 else None
    stated = loaded.get("version") if isinstance(loaded, dict) else None
    if inferred:
        return inferred
    if stated in ("v1", "v2"):
        return stated
    raise ValueError("No pude ver si el modelo es v1 o v2.")


def _f0_flag(loaded, weight):
    if isinstance(loaded, dict) and loaded.get("f0") in (0, 1):
        return int(loaded["f0"])
    if any(isinstance(key, str) and key.startswith("enc_p.emb_pitch.") for key in weight):
        return 1
    return 0


def _sibling_train_config(path):
    candidate = os.path.join(os.path.dirname(os.path.abspath(path)), "config.json")
    if not os.path.isfile(candidate):
        return None
    try:
        with open(candidate, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if isinstance(loaded, dict) and loaded.get("model_type") == "hubert":
        return None
    return loaded


def _config_list(loaded, path):
    if isinstance(loaded, dict) and isinstance(loaded.get("config"), list):
        if len(loaded["config"]) >= 18:
            return list(loaded["config"][:18])
    raw = _sibling_train_config(path)
    if isinstance(raw, list) and len(raw) >= 18:
        return list(raw[:18])
    if not isinstance(raw, dict):
        return None
    model = raw.get("model") if isinstance(raw.get("model"), dict) else {}
    data = raw.get("data") if isinstance(raw.get("data"), dict) else {}
    sampling = data.get("sampling_rate", model.get("sr", raw.get("sr")))
    filt = data.get("filter_length")
    if filt:
        spec = int(filt) // 2 + 1
    else:
        spec = model.get("spec_channels")
    values = [
        spec,
        32,
        model.get("inter_channels"),
        model.get("hidden_channels"),
        model.get("filter_channels"),
        model.get("n_heads"),
        model.get("n_layers"),
        model.get("kernel_size"),
        model.get("p_dropout", 0.0),
        model.get("resblock"),
        model.get("resblock_kernel_sizes"),
        model.get("resblock_dilation_sizes"),
        model.get("upsample_rates"),
        model.get("upsample_initial_channel"),
        model.get("upsample_kernel_sizes"),
        model.get("spk_embed_dim"),
        model.get("gin_channels"),
        sampling,
    ]
    if any(item is None for item in values):
        return None
    return values


def build_inference_checkpoint(loaded, path):
    """Return (packed, note, already_on_disk).

    already_on_disk means `path` is already an inference checkpoint.
    """
    weight = _extract_weight(loaded)
    if weight is None:
        raise ValueError(
            f"{os.path.basename(path)} no es un modelo RVC. "
            "Usá el .pth de la carpeta weights, no el G_ del entrenamiento."
        )
    version = _version_of(loaded, weight)
    f0 = _f0_flag(loaded, weight)
    config = _config_list(loaded, path)
    if config is None:
        raise ValueError(
            f"{os.path.basename(path)} es un archivo del entrenamiento, sin el paquete de inferencia. "
            "Usá el .pth de la carpeta weights, o dejá el config.json de ese entrenamiento al lado."
        )
    if "emb_g.weight" not in weight:
        raise ValueError(
            f"{os.path.basename(path)} no trae el embedding del hablante. "
            "Usá el .pth de la carpeta weights."
        )
    packed = {
        "weight": _without_posterior(weight),
        "config": config,
        "f0": f0,
        "version": version,
    }
    complete = (
        isinstance(loaded, dict)
        and loaded.get("version") in ("v1", "v2")
        and loaded.get("f0") in (0, 1)
        and loaded.get("weight") is weight
        and loaded.get("config") == config
        and not any("enc_q" in key for key in weight)
        and loaded.get("version") == version
    )
    if isinstance(loaded, dict) and isinstance(loaded.get("config"), list):
        note = f"Modelo {version}."
    else:
        note = f"Armé el modelo {version} con el config.json del entrenamiento."
    return packed, note, complete


def _prepared_dir():
    path = os.path.join(os.path.dirname(os.path.abspath(RVC_DIR)), "rvc_prepared")
    os.makedirs(path, exist_ok=True)
    return path


def _load_checkpoint(path):
    import torch

    try:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")
    except Exception as exc:
        raise ValueError(
            f"{os.path.basename(path)} no es un modelo RVC. "
            "Usá el .pth de la carpeta weights, no el G_ del entrenamiento."
        ) from exc


def prepare_voice_model(path):
    """Return (inference_path, version, note)."""
    _reject_unsafe_checkpoint(path)
    loaded = _load_checkpoint(path)
    packed, note, complete = build_inference_checkpoint(loaded, path)
    if complete:
        return path, packed["version"], note
    import torch

    dest = os.path.join(_prepared_dir(), os.path.basename(path))
    torch.save(packed, dest)
    return dest, packed["version"], note


def _hubert_weight_file(folder):
    for name in ("model.safetensors", "pytorch_model.bin"):
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            return candidate
    return None


def _safetensor_keys(path):
    try:
        with open(path, "rb") as handle:
            raw = handle.read(8)
            if len(raw) < 8:
                return None
            size = int.from_bytes(raw, "little")
            if size <= 0 or size > 8_000_000:
                return None
            header = json.loads(handle.read(size))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(header, dict):
        return None
    return [key for key in header if key != "__metadata__"]


def _require_hubert_weights(folder, version):
    path = _hubert_weight_file(folder)
    if path is None:
        raise ValueError(
            "HuBERT no tiene los pesos (model.safetensors). Volvé a bajar los modelos."
        )
    if version != "v1" or not path.endswith(".safetensors"):
        return path
    keys = _safetensor_keys(path)
    if not keys or not any(key.startswith("final_proj.") for key in keys):
        raise ValueError(
            "Este modelo es v1 y HuBERT no tiene la proyección final_proj. "
            "Volvé a bajar los modelos."
        )
    return path


def _index_for(path, version):
    if not path:
        return None, "Sin índice."
    try:
        import faiss
    except Exception:
        return path, ""
    try:
        index = faiss.read_index(path)
    except Exception:
        return None, "No pude abrir el índice. Convertí sin índice."
    expected = 256 if version == "v1" else 768
    if int(getattr(index, "d", expected)) != expected:
        return None, f"El índice no coincide con el modelo {version}. Convertí sin índice."
    return path, ""


def _clamp_pitch(value):
    try:
        value = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return min(12, max(-12, value))


def _clamp_influence(value, has_index):
    if not has_index:
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.66
    return min(1.0, max(0.0, value))


def accelerator_available():
    try:
        import torch
    except Exception:
        return False
    mps = getattr(torch.backends, "mps", None)
    return bool(torch.cuda.is_available() or (mps is not None and mps.is_available()))


def _reset_converter():
    global _converter, _converter_key
    _converter = None
    _converter_key = None


def run_with_device_fallback(callback):
    """Call callback(only_cpu). If the chip fails, retry once on CPU.

    Returns (value, used_cpu).
    """
    global _cpu_only
    only_cpu = _cpu_only or not accelerator_available()
    try:
        return callback(only_cpu), only_cpu
    except ValueError:
        raise
    except Exception:
        if only_cpu:
            raise
        _cpu_only = True
        _reset_converter()
        return callback(True), True


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


def _get_converter(hubert, rmvpe, only_cpu=False):
    global _converter, _converter_key
    key = (os.path.abspath(hubert), os.path.abspath(rmvpe), bool(only_cpu))
    if _converter is not None and _converter_key == key:
        return _converter
    BaseLoader = _import_loader()
    _converter = BaseLoader(
        only_cpu=bool(only_cpu), hubert_path=hubert, rmvpe_path=rmvpe
    )
    _full_precision_on_mps(_converter)
    _converter_key = key
    return _converter


def _full_precision_on_mps(converter):
    """The engine leaves half precision on for MPS. That crashes the chip."""
    config = converter.config
    if not str(getattr(config, "device", "")).startswith("mps"):
        return
    config.is_half = False
    config.x_pad = 1
    config.x_query = 6
    config.x_center = 38
    config.x_max = 41


def _load_hubert_module(folder):
    from infer_rvc_python.main import HubertModelWithFinalProj
    from transformers import HubertConfig

    try:
        return HubertModelWithFinalProj.from_pretrained(folder)
    except Exception:
        config = HubertConfig.from_pretrained(folder)
        if not getattr(config, "classifier_proj_size", None):
            config.classifier_proj_size = 256
        model = HubertModelWithFinalProj(config)
        weights = _hubert_weight_file(folder)
        if weights.endswith(".safetensors"):
            from safetensors.torch import load_file

            state = load_file(weights)
        else:
            import torch

            state = torch.load(weights, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        return model


def _attach_hubert(converter, folder):
    import torch
    from torch import nn
    from infer_rvc_python.main import FairseqHubertWrapper

    class LocalHubert(FairseqHubertWrapper):
        def __init__(self, model):
            nn.Module.__init__(self)
            self.model = model

    module = LocalHubert(_load_hubert_module(folder))
    device = converter.config.device
    module = module.to(device)
    if converter.config.is_half and torch.device(device).type != "cpu":
        module = module.half()
    else:
        module = module.float()
    module.eval()
    converter.hu_bert_model = module


def _passthrough_error(exc):
    text = str(exc)
    return text[:1].isupper() and any(
        text.startswith(prefix)
        for prefix in (
            "Falta",
            "No pude",
            "No encontré",
            "Ese ",
            "El motor",
            "rvc_models",
            "La conversión",
            "Este modelo",
            "HuBERT",
            "Armé",
        )
    )


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


def _execute(audio_path, model_path, hubert, rmvpe, index_path, pitch, influence, only_cpu):
    converter = _get_converter(hubert, rmvpe, only_cpu=only_cpu)
    _attach_hubert(converter, hubert)
    tag = os.path.abspath(model_path)
    try:
        converter.apply_conf(
            tag=tag,
            file_model=os.path.abspath(model_path),
            pitch_algo="rmvpe",
            pitch_lvl=pitch,
            file_index=index_path or "",
            index_influence=influence,
            respiration_median_filtering=3,
            envelope_ratio=0.25,
            consonant_breath_protection=0.33,
        )
        converted = converter.generate_from_cache(
            audio_data=os.path.abspath(audio_path), tag=tag
        )
    except ValueError as exc:
        if _passthrough_error(exc):
            raise
        raise ValueError(f"No pude convertir la voz: {exc}") from exc
    if not isinstance(converted, tuple) or len(converted) != 2:
        raise ValueError("La conversión no devolvió audio.")
    samples, sample_rate = converted
    dest = _output_path(audio_path)
    _write_converted(samples, sample_rate, dest)
    if os.path.abspath(dest) == os.path.abspath(audio_path):
        raise ValueError("La conversión no generó un archivo nuevo.")
    return dest


def convert_voice(audio_path, model_path, index_path=None, pitch=0, index_influence=0.66):
    global last_note
    last_note = ""
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
    model_path, version, prep_note = prepare_voice_model(model_path)
    _require_hubert_weights(hubert, version)
    index_path, index_note = _index_for(index_path, version)
    pitch = _clamp_pitch(pitch)
    influence = _clamp_influence(index_influence, bool(index_path))
    started_on_cpu = _cpu_only or not accelerator_available()

    def once(only_cpu):
        return _execute(
            audio_path, model_path, hubert, rmvpe, index_path, pitch, influence, only_cpu
        )

    try:
        dest, used_cpu = run_with_device_fallback(once)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"No pude convertir la voz: {exc}") from exc
    if used_cpu and not started_on_cpu:
        device_note = "El chip no pudo convertir. Seguí en CPU."
    elif used_cpu:
        device_note = "Corrí en CPU."
    else:
        device_note = "Corrí en el chip."
    notes = [prep_note]
    if index_note:
        notes.append(index_note)
    notes.append(device_note)
    last_note = " ".join(note for note in notes if note)
    return dest
