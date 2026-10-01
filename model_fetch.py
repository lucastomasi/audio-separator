"""Download the models the app can fetch on its own.

The vocal ONNX, HuBERT and rmvpe.pt are public files. The user's voice
(.pth) is not: nothing here invents or downloads that checkpoint.
"""
import json
import os
import threading
import urllib.error
import urllib.request

# UVR-MDX-NET-Voc_FT predicts vocals.
# Input is [1, 4, 3072, 256]; n_fft is 6144 and compensate is 1.021.
# Leaving n_fft unset makes _mdx_config pick 7680 when dim_f is 3072.
VOCAL_ONNX_NAME = "UVR-MDX-NET-Voc_FT.onnx"
VOCAL_ONNX_URL = (
    "https://github.com/TRvlvr/model_repo/releases/download/"
    "all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx"
)
VOCAL_ONNX_MIN_BYTES = 60_000_000
VOCAL_SIDECAR = {
    "n_fft": 6144,
    "dim_f": 3072,
    "dim_t": 256,
    "hop": 1024,
    "overlap": 0.5,
    "compensate": 1.021,
}

HUBERT_DIR_NAME = "hubert_base"
HUBERT_CONFIG_URL = "https://huggingface.co/r3gm/hubert_base/resolve/main/config.json"
HUBERT_WEIGHTS_NAME = "model.safetensors"
HUBERT_WEIGHTS_URL = (
    "https://huggingface.co/r3gm/hubert_base/resolve/main/model.safetensors"
)
HUBERT_WEIGHTS_MIN_BYTES = 300_000_000
HUBERT_CONFIG_MIN_BYTES = 128

RMVPE_NAME = "rmvpe.pt"
RMVPE_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
RMVPE_MIN_BYTES = 150_000_000

_LOCK = threading.RLock()
_CHUNK = 256 * 1024


class DownloadError(ValueError):
    """A model download failed. The message is safe to show in the UI."""


def _report(on_progress, frac, message):
    if on_progress is None:
        return
    on_progress(max(0.0, min(1.0, float(frac))), message)


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        return


def _big_enough(path, min_bytes):
    try:
        return os.path.isfile(path) and os.path.getsize(path) >= min_bytes
    except OSError:
        return False


def _looks_like_html(chunk):
    sample = chunk.lstrip().lower()
    return (
        sample.startswith(b"<!doctype")
        or sample.startswith(b"<html")
        or sample.startswith(b"<head")
    )


def _read_json_object(path):
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if isinstance(loaded, dict):
        return loaded
    return None


def download_file(url, dest, min_bytes, on_progress=None, label="el archivo"):
    """Stream url to dest. Return True when this call wrote the file."""
    if _big_enough(dest, min_bytes):
        return False
    directory = os.path.dirname(os.path.abspath(dest))
    os.makedirs(directory, exist_ok=True)
    part = dest + ".part"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AudioSeparator/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "text/html" in content_type:
                raise DownloadError(
                    f"No pude bajar {label}. La respuesta no es el archivo."
                )
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header and total_header.isdigit() else None
            if total is not None and total < min_bytes:
                raise DownloadError(f"No pude bajar {label}. Quedó incompleto.")
            written = 0
            first = True
            with open(part, "wb") as handle:
                while True:
                    chunk = response.read(_CHUNK)
                    if not chunk:
                        break
                    if first:
                        first = False
                        if _looks_like_html(chunk):
                            raise DownloadError(
                                f"No pude bajar {label}. La respuesta no es el archivo."
                            )
                    handle.write(chunk)
                    written += len(chunk)
                    if total:
                        _report(on_progress, written / total, f"Bajando {label}…")
                    else:
                        _report(on_progress, 0.0, f"Bajando {label}…")
            if written < min_bytes:
                raise DownloadError(f"No pude bajar {label}. Quedó incompleto.")
            os.replace(part, dest)
    except DownloadError:
        _remove(part)
        raise
    except urllib.error.HTTPError as exc:
        _remove(part)
        raise DownloadError(f"No pude bajar {label}.") from exc
    except Exception as exc:
        _remove(part)
        raise DownloadError(
            f"No pude bajar {label}. Revisá la conexión y volvé a intentar."
        ) from exc
    _report(on_progress, 1.0, f"Bajando {label}…")
    return True


def write_vocal_sidecar(onnx_path):
    dest = os.path.splitext(onnx_path)[0] + ".json"
    loaded = _read_json_object(dest)
    if loaded and {"n_fft", "dim_f", "dim_t"} <= set(loaded):
        return dest
    with open(dest, "w", encoding="utf-8") as handle:
        json.dump(VOCAL_SIDECAR, handle)
    return dest


def _hubert_config_ready(path):
    loaded = _read_json_object(path)
    return bool(loaded) and loaded.get("model_type") == "hubert"


def _hubert_weights_ready(folder):
    for name in (HUBERT_WEIGHTS_NAME, "pytorch_model.bin"):
        if _big_enough(os.path.join(folder, name), HUBERT_WEIGHTS_MIN_BYTES):
            return True
    return False


def ensure_vocal_model(mdx_dir, on_progress=None):
    """Download the vocal ONNX and its sidecar. Return (path, downloaded)."""
    with _LOCK:
        os.makedirs(mdx_dir, exist_ok=True)
        dest = os.path.join(mdx_dir, VOCAL_ONNX_NAME)
        downloaded = download_file(
            VOCAL_ONNX_URL,
            dest,
            VOCAL_ONNX_MIN_BYTES,
            on_progress=on_progress,
            label="el modelo de separación",
        )
        write_vocal_sidecar(dest)
        return dest, downloaded


def ensure_rvc_support(rvc_dir, on_progress=None):
    """Download HuBERT and rmvpe.pt. Return True if something was written."""
    with _LOCK:
        folder = os.path.join(rvc_dir, HUBERT_DIR_NAME)
        os.makedirs(folder, exist_ok=True)
        config_path = os.path.join(folder, "config.json")
        weights_path = os.path.join(folder, HUBERT_WEIGHTS_NAME)
        rmvpe_path = os.path.join(rvc_dir, RMVPE_NAME)
        downloaded = False
        if not _hubert_config_ready(config_path):
            _remove(config_path)
            download_file(
                HUBERT_CONFIG_URL,
                config_path,
                HUBERT_CONFIG_MIN_BYTES,
                on_progress=on_progress,
                label="HuBERT",
            )
            if not _hubert_config_ready(config_path):
                _remove(config_path)
                raise DownloadError("No pude bajar HuBERT. La respuesta no es el archivo.")
            downloaded = True
        if not _hubert_weights_ready(folder):
            download_file(
                HUBERT_WEIGHTS_URL,
                weights_path,
                HUBERT_WEIGHTS_MIN_BYTES,
                on_progress=on_progress,
                label="HuBERT",
            )
            downloaded = True
        if not _big_enough(rmvpe_path, RMVPE_MIN_BYTES):
            download_file(
                RMVPE_URL,
                rmvpe_path,
                RMVPE_MIN_BYTES,
                on_progress=on_progress,
                label="el estimador de pitch",
            )
            downloaded = True
        return downloaded


def ensure_standard_models(mdx_dir, rvc_dir, on_progress=None):
    """Download separation and RVC support files. Return True if any were new."""
    _path, vocal_downloaded = ensure_vocal_model(mdx_dir, on_progress=on_progress)
    rvc_downloaded = ensure_rvc_support(rvc_dir, on_progress=on_progress)
    return vocal_downloaded or rvc_downloaded


def fetch_status(downloaded):
    if downloaded:
        return "Listo. Bajé los modelos. Tu voz sigue siendo un archivo .pth tuyo."
    return "Los modelos ya estaban. Tu voz sigue siendo un archivo .pth tuyo."
