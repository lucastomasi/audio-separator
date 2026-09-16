"""Voice clone = tonyassi/voice-clone procedure, CPU local, no HF download."""
import os
import sys
import types

from exports import copy_to_downloads
from library import xtts_dir

LEGACY_XTTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xtts_models")
REQUIRED = ("config.json", "model.pth", "vocab.json", "speakers_xtts.pth")

_tts = None


def _xtts_roots():
    return [xtts_dir(), LEGACY_XTTS]


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


def _stub_vits_deps():
    """TTS.api imports VITS; we only run XTTS. Stub the C extension."""
    if "monotonic_alignment_search" in sys.modules:
        return

    def _missing(*_a, **_k):
        raise RuntimeError("VITS no se usa; solo XTTS-v2.")

    stub = types.ModuleType("monotonic_alignment_search")
    stub.maximum_path = _missing
    sys.modules["monotonic_alignment_search"] = stub
    sys.modules["monotonic_alignment"] = stub


def _patch_torch_load():
    import torch

    original = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)

    torch.load = _load


def get_tts():
    """Same as tonyassi/voice-clone: TTS(...).to(device), but CPU + local files."""
    global _tts
    if _tts is not None:
        return _tts
    model_dir = active_xtts_dir()
    os.environ["COQUI_TOS_AGREED"] = "1"
    _stub_vits_deps()
    _patch_torch_load()
    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(
        model_path=model_dir,
        config_path=os.path.join(model_dir, "config.json"),
        gpu=(device == "cuda"),
        progress_bar=False,
    )
    tts.to(device)
    _tts = tts
    return _tts


def clone_voice(text, speaker_wav, language="es"):
    """Same call as the Space: tts_to_file(text=, speaker_wav=, language=, file_path=)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("Escribí el texto a clonar.")
    if not speaker_wav or not os.path.isfile(speaker_wav):
        raise ValueError("Falta el audio de referencia.")
    tts = get_tts()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clone_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "voz_clon.wav")
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        file_path=out_path,
    )
    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise ValueError("La clonación no produjo audio.")
    _, copied = copy_to_downloads([out_path], ["voz_clon"])
    return copied[0] if copied else os.path.abspath(out_path)
