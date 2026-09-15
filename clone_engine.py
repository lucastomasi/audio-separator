"""Local Coqui XTTS-v2 clone. Full model, CPU/CUDA, no Hugging Face download."""
import os

from exports import copy_to_downloads

from library import xtts_dir

XTTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xtts_models")


def _xtts_roots():
    return [xtts_dir(), XTTS_DIR]
REQUIRED = ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth")

_tts = None


def xtts_dir():
    return XTTS_DIR


def _root_complete(root):
    return all(
        os.path.isfile(os.path.join(root, name)) and os.path.getsize(os.path.join(root, name)) > 0
        for name in REQUIRED
    )


def missing_xtts_files():
    if any(_root_complete(root) for root in _xtts_roots()):
        return []
    return [f"xtts_models/{name}" for name in REQUIRED]


def active_xtts_dir():
    for root in _xtts_roots():
        if _root_complete(root):
            return root
    raise ValueError(
        "Faltan los modelos locales de XTTS: " + ", ".join(missing_xtts_files())
    )


def require_xtts_models():
    missing = missing_xtts_files()
    if missing:
        raise ValueError("Faltan los modelos locales de XTTS: " + ", ".join(missing))


def _patch_torch_load():
    import torch

    original = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = _load


def get_tts():
    global _tts
    if _tts is not None:
        return _tts
    model_dir = active_xtts_dir()
    os.environ["COQUI_TOS_AGREED"] = "1"
    _patch_torch_load()
    import torch
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    config = XttsConfig()
    config.load_json(os.path.join(model_dir, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    _tts = (model, config, device)
    return _tts


def clone_voice(text, speaker_wav, language="es"):
    text = (text or "").strip()
    if not text:
        raise ValueError("Escribí el texto a clonar.")
    if not speaker_wav or not os.path.isfile(speaker_wav):
        raise ValueError("Falta el audio de referencia.")
    model, config, _device = get_tts()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clone_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "voz_clon.wav")
    result = model.synthesize(
        text,
        config,
        speaker_wav=speaker_wav,
        language=language,
    )
    wav = result["wav"] if isinstance(result, dict) else result
    import soundfile as sf
    import numpy as np

    audio = np.asarray(wav)
    if audio.ndim == 1:
        pass
    sf.write(out_path, audio, 24000)
    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise ValueError("La clonación no produjo audio.")
    _, copied = copy_to_downloads([out_path], ["voz_clon"])
    return copied[0] if copied else os.path.abspath(out_path)
