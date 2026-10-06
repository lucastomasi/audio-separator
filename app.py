"""Gradio UI for the local Mac app. Launch with desktop.py."""
import json
import os
import re
import secrets
import shutil
import threading
import time
from urllib.parse import parse_qs, urlparse

import gradio as gr
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter
from scipy.signal import istft, stft

from app_paths import data_dir, is_inside, source_dir
from audio_io import get_duration, load
from audio_text import STEM_AMBAS, STEM_SOLO_INST, STEM_SOLO_VOZ, stem_choice_to_list
from exports import copy_to_downloads, exports_dir, is_exportable, open_exports_dir
from remix import REMIX_DIR, SAMPLE_RATE, mix_stems
from rvc_engine import RVC_DIR, convert_voice
from youtube_lib import download_audio

ROOT = source_dir()
CLEAN_DIR = os.path.join(data_dir(), "clean_song_output")
MDX_DIR = os.path.join(data_dir(), "mdx_models")
HOST = "127.0.0.1"
PORT = 7860
CSS_PATH = os.path.join(ROOT, "ui.css")
TOKEN_ENV = "AUDIO_SEPARATOR_TOKEN"
TOKEN_COOKIE = "as_token"
TOKEN_QUERY = "access_token"
TOKEN_HEADER = "x-audio-separator-token"
MAX_UPLOAD = "200mb"
MAX_N_FFT = 16384
MAX_DIM_F = 8192
MAX_DIM_T = 2048
MAX_HOP = 4096
_JOB_LOCK = threading.Lock()
_HEAD_TOKEN_JS = """
<script>
(function () {
  var params = new URLSearchParams(window.location.search);
  var token = params.get("access_token");
  if (token) {
    document.cookie = "as_token=" + encodeURIComponent(token) + "; path=/; SameSite=Strict";
  }
})();
</script>
"""

STEM_CHOICES = [
    ("Solo la voz", STEM_SOLO_VOZ),
    ("Solo el instrumental", STEM_SOLO_INST),
    ("Voz e instrumental", STEM_AMBAS),
]
LOCAL_SEPARATION = "Separación local"


def _token_ok(provided, token):
    if not provided or not token or len(provided) != len(token):
        return False
    return secrets.compare_digest(provided, token)


def _loopback_host(request):
    host = request.headers.get("host") or ""
    if host.startswith("["):
        name = host.rsplit("]", 1)[0].lstrip("[")
    else:
        name = host.split(":")[0]
    return name.lower() in {"127.0.0.1", "localhost", "::1"}


def _token_from_referer(request, token):
    referer = request.headers.get("referer") or ""
    parsed = urlparse(referer)
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return None
    values = parse_qs(parsed.query).get(TOKEN_QUERY) or []
    if not values:
        return None
    return values[0] if _token_ok(values[0], token) else None


def auth_dependency(request):
    token = os.environ.get(TOKEN_ENV, "")
    if not token or not _loopback_host(request):
        return None
    provided = (
        request.headers.get(TOKEN_HEADER)
        or request.query_params.get(TOKEN_QUERY)
        or request.cookies.get(TOKEN_COOKIE)
        or _token_from_referer(request, token)
    )
    if _token_ok(provided, token):
        return "local"
    return None


def launch_kwargs(**overrides):
    """Arguments for demo.launch(). desktop.py passes prevent_thread_lock and inbrowser."""
    try:
        with open(CSS_PATH, encoding="utf-8") as handle:
            css = handle.read()
    except OSError:
        css = ""
    kwargs = {
        "server_name": HOST,
        "server_port": PORT,
        "inbrowser": False,
        "share": False,
        "show_error": False,
        "ssr_mode": False,
        "css": css,
        "head": _HEAD_TOKEN_JS,
        "theme": gr.themes.Soft(),
        "allowed_paths": _allowed_paths(),
        "blocked_paths": _blocked_paths(),
        "footer_links": [],
        "max_file_size": MAX_UPLOAD,
        "strict_cors": True,
        "enable_monitoring": False,
        "run_history": False,
        "mcp_server": False,
        "auth_dependency": auth_dependency,
        "app_kwargs": {
            "docs_url": None,
            "redoc_url": None,
            "openapi_url": None,
        },
    }
    kwargs.update(overrides)
    import inspect

    supported = set(inspect.signature(gr.Blocks.launch).parameters)
    return {key: value for key, value in kwargs.items() if key in supported}


def _output_dirs():
    return [
        CLEAN_DIR,
        REMIX_DIR,
        os.path.join(data_dir(), "rvc_output"),
        os.path.join(data_dir(), "downloads"),
        exports_dir(),
    ]


def _allowed_paths():
    paths = _output_dirs()
    for path in paths:
        os.makedirs(path, exist_ok=True)
    return paths


def _blocked_paths():
    blocked = [
        os.path.expanduser("~/.ssh"),
        os.path.expanduser("~/.gnupg"),
        os.path.expanduser("~/.aws"),
        "/etc",
    ]
    if os.path.realpath(data_dir()) != os.path.realpath(source_dir()):
        blocked.append(source_dir())
    else:
        blocked.extend(
            [
                os.path.join(source_dir(), ".git"),
                os.path.join(source_dir(), ".cursor"),
                os.path.join(source_dir(), ".env"),
            ]
        )
    return [path for path in blocked if path]


def _inside(path, root):
    return is_inside(path, root)


def _run_job(work):
    if not _JOB_LOCK.acquire(blocking=False):
        raise gr.Error("Esperá a que termine lo que está corriendo.")
    try:
        return work()
    finally:
        _JOB_LOCK.release()


def _list_files(directory, extension):
    if not os.path.isdir(directory):
        return []
    found = []
    for name in sorted(os.listdir(directory)):
        if name.startswith("."):
            continue
        if name.lower().endswith(extension):
            found.append(os.path.join(directory, name))
    return found


def _audio_path(value):
    if value is None:
        return None
    if isinstance(value, (str, os.PathLike)):
        path = str(value)
        return path if path and os.path.isfile(path) else None
    if isinstance(value, dict):
        path = value.get("path") or value.get("name")
        return path if path and os.path.isfile(path) else None
    return None


def _safe_stem(path):
    name = os.path.splitext(os.path.basename(str(path)))[0]
    name = re.sub(r"[^\w.-]+", "_", name, flags=re.UNICODE).strip("._")
    return (name or "audio")[:60]


def _duration_note(path):
    try:
        seconds = float(get_duration(path))
    except Exception:
        return ""
    return f" ({seconds:.1f} s)"


def _as_stereo(wave):
    wave = np.asarray(wave, dtype=np.float32)
    if wave.size == 0 or wave.shape[-1] == 0:
        raise ValueError("El audio está vacío.")
    if wave.ndim == 1:
        return np.stack([wave, wave], axis=0)
    if wave.shape[0] == 1:
        return np.concatenate([wave, wave], axis=0)
    return np.ascontiguousarray(wave[:2], dtype=np.float32)


def _copied_stem(src, stem, label):
    os.makedirs(CLEAN_DIR, exist_ok=True)
    stamp = time.strftime("%H%M%S")
    ext = os.path.splitext(src)[1] or ".wav"
    dest = os.path.join(CLEAN_DIR, f"{stem}-{label}-{stamp}{ext}")
    if os.path.exists(dest):
        dest = os.path.join(
            CLEAN_DIR, f"{stem}-{label}-{stamp}-{time.time_ns() % 100000}{ext}"
        )
    shutil.copy2(src, dest)
    return dest


def _write_audio(directory, stem, label, wave, sample_rate):
    os.makedirs(directory, exist_ok=True)
    stamp = time.strftime("%H%M%S")
    dest = os.path.join(directory, f"{stem}-{label}-{stamp}.wav")
    if os.path.exists(dest):
        dest = os.path.join(
            directory, f"{stem}-{label}-{stamp}-{time.time_ns() % 100000}.wav"
        )
    data = np.nan_to_num(np.asarray(wave, dtype=np.float32))
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak > 0.99:
        data = data * np.float32(0.99 / peak)
    if data.ndim == 1:
        sf.write(dest, data, sample_rate, subtype="PCM_16")
    else:
        sf.write(dest, np.ascontiguousarray(data.T), sample_rate, subtype="PCM_16")
    return dest


def _vocal_weight(freqs):
    freqs = np.asarray(freqs, dtype=np.float64)
    low = np.clip((freqs - 80.0) / (250.0 - 80.0), 0.0, 1.0)
    high = np.clip((7000.0 - freqs) / (7000.0 - 4000.0), 0.0, 1.0)
    return (low * high).astype(np.float32)


def separate_center(stereo, sample_rate):
    """Split a stereo mix into a centered vocal and the leftover instrumental."""
    length = stereo.shape[1]
    n_fft = 4096
    if length < n_fft:
        n_fft = 2048 if length >= 2048 else 1024 if length >= 1024 else 512
    hop = max(128, n_fft // 4)
    noverlap = n_fft - hop
    mid = ((stereo[0] + stereo[1]) * 0.5).astype(np.float32)
    side = ((stereo[0] - stereo[1]) * 0.5).astype(np.float32)
    freqs, _, mid_spec = stft(
        mid,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )
    _, _, side_spec = stft(
        side,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )
    mag_mid = np.abs(mid_spec)
    mag_side = np.abs(side_spec)
    weight = _vocal_weight(freqs)
    side_ratio = float(np.mean(mag_side) / (np.mean(mag_mid) + 1e-8))
    if side_ratio < 0.05:
        mask = weight[:, None]
    else:
        center = mag_mid / (mag_mid + mag_side + 1e-8)
        mask = np.clip((center - 0.35) / 0.45, 0.0, 1.0) * weight[:, None]
    if mask.shape[0] > 1 and mask.shape[1] > 1:
        mask = uniform_filter(mask, size=(9, 5), mode="nearest")
    vocal_spec = mid_spec * mask.astype(np.float32)
    _, vocal = istft(
        vocal_spec,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=noverlap,
        input_onesided=True,
        boundary=True,
    )
    vocal = np.nan_to_num(vocal.astype(np.float32))
    if vocal.shape[0] < length:
        vocal = np.pad(vocal, (0, length - vocal.shape[0]))
    vocal = vocal[:length]
    vocals = np.stack([vocal, vocal], axis=0)
    instrumental = np.nan_to_num(stereo - vocals)
    return vocals, instrumental


def _positive_dim(shape, index):
    if not isinstance(shape, (list, tuple)) or len(shape) <= index:
        return None
    value = shape[index]
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return int(value)
    return None


def _mdx_config(model_path, session):
    config = {"hop": 1024, "overlap": 0.5, "compensate": 1.0}
    sidecar = os.path.splitext(model_path)[0] + ".json"
    if os.path.isfile(sidecar):
        with open(sidecar, encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("El .json del modelo tiene que ser un objeto.")
        for key in ("hop", "overlap", "compensate", "dim_f", "dim_t", "n_fft"):
            if key in loaded:
                config[key] = loaded[key]
    shape = session.get_inputs()[0].shape
    if "dim_f" not in config:
        dim_f = _positive_dim(shape, 2)
        if not dim_f:
            raise ValueError(
                "No pude leer dim_f del ONNX. Dejá un .json al lado con n_fft, dim_f y dim_t."
            )
        config["dim_f"] = dim_f
    if "dim_t" not in config:
        dim_t = _positive_dim(shape, 3)
        if not dim_t:
            raise ValueError(
                "No pude leer dim_t del ONNX. Dejá un .json al lado con n_fft, dim_f y dim_t."
            )
        config["dim_t"] = dim_t
    if "n_fft" not in config:
        dim_f = int(config["dim_f"])
        config["n_fft"] = 7680 if dim_f == 3072 else dim_f * 2
    if int(config["n_fft"]) // 2 + 1 < int(config["dim_f"]):
        config["n_fft"] = int(config["dim_f"]) * 2
    n_fft = int(config["n_fft"])
    hop = int(config["hop"])
    dim_f = int(config["dim_f"])
    dim_t = int(config["dim_t"])
    if (
        n_fft > MAX_N_FFT
        or hop > MAX_HOP
        or dim_f > MAX_DIM_F
        or dim_t > MAX_DIM_T
        or hop < 1
        or dim_t < 2
        or dim_f < 1
        or n_fft < 2
    ):
        raise ValueError("La configuración del modelo ONNX no es válida.")
    return config


def _fit_freq(spec, dim_f):
    spec = np.asarray(spec)
    if spec.shape[0] > dim_f:
        return spec[:dim_f]
    if spec.shape[0] < dim_f:
        return np.pad(spec, ((0, dim_f - spec.shape[0]), (0, 0)))
    return spec


def _pack_channels(left, right, dim_f, dim_t):
    # Channel order: left real, left imag, right real, right imag.
    left = _fit_freq(left, dim_f)
    right = _fit_freq(right, dim_f)
    packed = np.stack(
        [
            np.asarray(left.real, dtype=np.float32),
            np.asarray(left.imag, dtype=np.float32),
            np.asarray(right.real, dtype=np.float32),
            np.asarray(right.imag, dtype=np.float32),
        ],
        axis=0,
    )
    frames = packed.shape[-1]
    if frames < dim_t:
        packed = np.pad(packed, ((0, 0), (0, 0), (0, dim_t - frames)))
    elif frames > dim_t:
        packed = packed[..., :dim_t]
    return packed


def _stft_channel(wave, n_fft, hop, sample_rate):
    _, _, spec = stft(
        np.asarray(wave, dtype=np.float32),
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        boundary="zeros",
        padded=True,
    )
    return spec


def _wave_from_packed(packed, n_fft, hop, sample_rate, length):
    freq_bins = n_fft // 2 + 1
    dim_f = packed.shape[1]
    frames = packed.shape[2]
    channels = []
    for real_index, imag_index in ((0, 1), (2, 3)):
        spec = np.zeros((freq_bins, frames), dtype=np.complex64)
        usable = min(dim_f, freq_bins)
        spec[:usable] = packed[real_index, :usable] + 1j * packed[imag_index, :usable]
        _, wave = istft(
            spec,
            fs=sample_rate,
            window="hann",
            nperseg=n_fft,
            noverlap=n_fft - hop,
            input_onesided=True,
            boundary=True,
        )
        wave = np.nan_to_num(wave.astype(np.float32))
        if wave.shape[0] < length:
            wave = np.pad(wave, (0, length - wave.shape[0]))
        channels.append(wave[:length])
    return np.stack(channels, axis=0)


def _overlap_window(start, valid, length, fade):
    window = np.ones(valid, dtype=np.float32)
    if fade <= 1:
        return window
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    if start > 0:
        fade_in = min(fade, valid)
        window[:fade_in] *= ramp[:fade_in]
    if start + valid < length:
        fade_out = min(fade, valid)
        window[-fade_out:] *= ramp[:fade_out][::-1]
    return window


def separate_with_onnx(stereo, sample_rate, model_path):
    import onnxruntime as ort

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    config = _mdx_config(model_path, session)
    n_fft = int(config["n_fft"])
    hop = int(config["hop"])
    dim_f = int(config["dim_f"])
    dim_t = int(config["dim_t"])
    overlap = float(config.get("overlap", 0.5))
    compensate = float(config.get("compensate", 1.0))
    if hop < 1 or dim_t < 2 or dim_f < 1 or n_fft < 2:
        raise ValueError("La configuración del modelo ONNX no es válida.")
    chunk = hop * (dim_t - 1)
    overlap = min(max(overlap, 0.0), 0.9)
    step = max(1, int(round(chunk * (1.0 - overlap))))
    length = stereo.shape[1]
    accumulator = np.zeros((2, length), dtype=np.float32)
    weight = np.zeros(length, dtype=np.float32)
    fade = int(round(chunk * overlap))
    input_name = session.get_inputs()[0].name
    starts = list(range(0, max(length, 1), step)) or [0]
    if starts[-1] + chunk < length:
        starts.append(max(0, length - chunk))
    seen = set()
    for start in starts:
        if start in seen or start >= length:
            continue
        seen.add(start)
        valid = min(chunk, length - start)
        piece = np.zeros((2, chunk), dtype=np.float32)
        piece[:, :valid] = stereo[:, start : start + valid]
        packed = _pack_channels(
            _stft_channel(piece[0], n_fft, hop, sample_rate),
            _stft_channel(piece[1], n_fft, hop, sample_rate),
            dim_f,
            dim_t,
        )
        predicted = session.run(None, {input_name: packed[None, ...]})[0][0]
        vocal = _wave_from_packed(predicted, n_fft, hop, sample_rate, chunk)
        window = _overlap_window(start, valid, length, fade)
        accumulator[:, start : start + valid] += vocal[:, :valid] * window
        weight[start : start + valid] += window
    vocals = accumulator / np.maximum(weight, 1e-6)
    instrumental = stereo - vocals * np.float32(compensate)
    return np.nan_to_num(vocals.astype(np.float32)), np.nan_to_num(
        instrumental.astype(np.float32)
    )


def separation_choices():
    choices = [(LOCAL_SEPARATION, "")]
    choices.extend(
        (os.path.basename(path), path) for path in _list_files(MDX_DIR, ".onnx")
    )
    return choices


def rvc_model_choices():
    return [(os.path.basename(path), path) for path in _list_files(RVC_DIR, ".pth")]


def rvc_index_choices():
    choices = [("Sin índice", "")]
    choices.extend(
        (os.path.basename(path), path) for path in _list_files(RVC_DIR, ".index")
    )
    return choices


def _choice_value(choices, prefer_first_real=False):
    if prefer_first_real:
        for _label, value in choices:
            if value:
                return value
    for _label, value in choices:
        return value
    return None


def model_updates():
    separation = separation_choices()
    models = rvc_model_choices()
    indexes = rvc_index_choices()
    return (
        gr.update(choices=models, value=_choice_value(models)),
        gr.update(choices=indexes, value=""),
        gr.update(
            choices=separation,
            value=_choice_value(separation, prefer_first_real=True),
        ),
    )


def _resolve_mdx(choice):
    if not choice or choice == LOCAL_SEPARATION:
        return None
    if not _inside(choice, MDX_DIR) or not str(choice).lower().endswith(".onnx"):
        raise ValueError("Ese modelo de separación no está en mdx_models.")
    if not os.path.isfile(choice):
        raise ValueError("No encontré ese modelo .onnx.")
    return os.path.abspath(choice)


def _resolve_rvc_model(choice):
    if not choice or not _inside(choice, RVC_DIR) or not str(choice).lower().endswith(".pth"):
        raise ValueError("Elegí un modelo .pth que esté en rvc_models.")
    if not os.path.isfile(choice):
        raise ValueError("No encontré ese modelo RVC.")
    return os.path.abspath(choice)


def _resolve_index(choice):
    if not choice:
        return None
    if not _inside(choice, RVC_DIR) or not str(choice).lower().endswith(".index"):
        raise ValueError("Ese índice no está en rvc_models.")
    if not os.path.isfile(choice):
        raise ValueError("No encontré ese índice.")
    return os.path.abspath(choice)


def separate_to_files(src_path, mdx_choice=""):
    wave, sample_rate = load(src_path, mono=False, sr=SAMPLE_RATE)
    stereo = _as_stereo(wave)
    model_path = _resolve_mdx(mdx_choice)
    note = "Usé la separación local (canal central)."
    vocals = instrumental = None
    if model_path:
        try:
            vocals, instrumental = separate_with_onnx(stereo, sample_rate, model_path)
            note = f"Usé el modelo {os.path.basename(model_path)}."
        except Exception as exc:
            vocals = instrumental = None
            note = (
                f"No pude usar {os.path.basename(model_path)} ({exc}). "
                "Seguí con la separación local."
            )
    if vocals is None:
        vocals, instrumental = separate_center(stereo, sample_rate)
        if model_path is None:
            note = (
                "Usé la separación local (canal central). "
                "Si tenés un .onnx, dejalo en mdx_models."
            )
    stem = _safe_stem(src_path)
    vocal_path = _write_audio(CLEAN_DIR, stem, "voz", vocals, sample_rate)
    instrumental_path = _write_audio(
        CLEAN_DIR, stem, "instrumental", instrumental, sample_rate
    )
    return vocal_path, instrumental_path, note


def _public_message(text):
    text = str(text)
    if re.search(r"(?i)(/users/|/home/|/library/|/tmp/|/var/|:\\|audio_separator_home)", text):
        return "Algo salió mal."
    return text


def _ui_error(exc):
    if isinstance(exc, gr.Error):
        raise exc
    if isinstance(exc, ValueError):
        raise gr.Error(_public_message(exc)) from exc
    raise gr.Error("Algo salió mal.") from exc


def on_download(url):
    url = (url or "").strip()
    if not url:
        raise gr.Error("Pegá un enlace de YouTube.")
    def work():
        return download_audio(url)
    try:
        path, cached, _ignored = _run_job(work)
    except Exception as exc:
        _ui_error(exc)
    if cached:
        message = f"Ese audio ya estaba descargado{_duration_note(path)}."
    else:
        message = f"Listo. Bajé el audio de YouTube{_duration_note(path)}."
    return path, None, None, None, None, message


def on_separate(audio, url, _stem, mdx_choice):
    source = _audio_path(audio)
    try:
        def work():
            chosen = source
            if chosen is None and (url or "").strip():
                chosen, _cached, _ignored = download_audio(url)
            if chosen is None:
                raise ValueError("Subí un audio o pegá un enlace de YouTube.")
            vocal_path, instrumental_path, note = separate_to_files(chosen, mdx_choice)
            return chosen, vocal_path, instrumental_path, note
        source, vocal_path, instrumental_path, note = _run_job(work)
    except Exception as exc:
        _ui_error(exc)
    message = f"Listo. Separé la voz y el instrumental{_duration_note(source)}. {note}"
    return source, vocal_path, instrumental_path, None, None, message


def on_convert(vocal, model, index):
    vocal_path = _audio_path(vocal)
    try:
        if vocal_path is None:
            raise ValueError("Primero separá el audio para tener la voz.")
        def work():
            model_path = _resolve_rvc_model(model)
            index_path = _resolve_index(index)
            converted = convert_voice(vocal_path, model_path, index_path)
            if not converted or not os.path.isfile(converted):
                raise ValueError("La conversión no devolvió un audio.")
            return _copied_stem(converted, _safe_stem(vocal_path), "voz-rvc")
        dest = _run_job(work)
    except Exception as exc:
        _ui_error(exc)
    return dest, None, "Listo. Convertí la voz con el modelo local."


def on_refresh():
    models = rvc_model_choices()
    note = (
        "Actualicé la lista de modelos."
        if models
        else "No hay modelos .pth en rvc_models. Copiá el modelo ahí y actualizá de nuevo."
    )
    return (*model_updates(), note)


def _vocal_for_remix(vocal, converted, use_converted):
    converted_path = _audio_path(converted)
    vocal_path = _audio_path(vocal)
    if use_converted and converted_path:
        return converted_path, "la voz convertida"
    if vocal_path:
        return vocal_path, "la voz separada"
    if converted_path:
        return converted_path, "la voz convertida"
    return None, ""


def on_remix(vocal, instrumental, converted, use_converted, vocal_db, instrumental_db):
    vocal_path, which = _vocal_for_remix(vocal, converted, use_converted)
    instrumental_path = _audio_path(instrumental)
    try:
        if vocal_path is None or instrumental_path is None:
            raise ValueError("Primero separá el audio. Hacen falta la voz y el instrumental.")
        def work():
            return mix_stems(
                vocal_path,
                instrumental_path,
                vocal_db=float(vocal_db),
                instrumental_db=float(instrumental_db),
                output_path=os.path.join(
                    REMIX_DIR,
                    f"{_safe_stem(vocal_path)}-remix-{time.strftime('%H%M%S')}.wav",
                ),
            )
        dest = _run_job(work)
    except Exception as exc:
        _ui_error(exc)
    return (
        dest,
        f"Listo. Uní {which} con el instrumental{_duration_note(dest)}.",
    )


def on_export(stem, vocal, instrumental, converted, remix, use_converted):
    wanted = set(stem_choice_to_list(stem))
    files = []
    labels = []
    vocal_path = _audio_path(vocal)
    instrumental_path = _audio_path(instrumental)
    converted_path = _audio_path(converted)
    remix_path = _audio_path(remix)
    if "vocal" in wanted and vocal_path and is_exportable(vocal_path):
        files.append(vocal_path)
        labels.append("voz")
    if "background" in wanted and instrumental_path and is_exportable(instrumental_path):
        files.append(instrumental_path)
        labels.append("instrumental")
    if "vocal" in wanted and use_converted and converted_path and is_exportable(converted_path):
        files.append(converted_path)
        labels.append("voz-convertida")
    if remix_path and is_exportable(remix_path):
        files.append(remix_path)
        labels.append("remix")
    if not files:
        raise gr.Error("Todavía no hay pistas para exportar. Separá un audio primero.")
    try:
        _directory, copied = copy_to_downloads(files, labels)
    except Exception as exc:
        _ui_error(exc)
    if not copied:
        raise gr.Error("No pude copiar los archivos a Descargas.")
    noun = "archivo" if len(copied) == 1 else "archivos"
    return f"Exporté {len(copied)} {noun} en Descargas."


def on_open_folder():
    try:
        open_exports_dir()
    except OSError:
        return "No pude abrirla desde acá. Buscá Descargas / Audio Separator."
    return "Abrí la carpeta de Descargas."


def build_server():
    separation = separation_choices()
    models = rvc_model_choices()
    indexes = rvc_index_choices()
    with gr.Blocks(title="Audio Separator", analytics_enabled=False) as demo:
        gr.Markdown(
            """
# Audio Separator
Separá la voz del instrumental, convertí la voz con un modelo RVC de `rvc_models` y volvé a unir las pistas.

Corre en tu Mac. No usa el Space de Hugging Face.
            """.strip()
        )
        status = gr.Markdown("Subí un audio o pegá un enlace de YouTube.")
        with gr.Row():
            audio_in = gr.Audio(
                label="Archivo",
                type="filepath",
                sources=["upload"],
            )
            with gr.Column():
                url = gr.Textbox(
                    label="Enlace de YouTube",
                    placeholder="Pegá el enlace acá",
                )
                download_btn = gr.Button("Bajá el audio")
        with gr.Row():
            stem = gr.Radio(
                choices=STEM_CHOICES,
                value=STEM_AMBAS,
                label="Qué querés exportar",
                info="La separación y la unión usan las dos pistas. Esto elige qué se copia a Descargas.",
            )
            mdx = gr.Dropdown(
                choices=separation,
                value=_choice_value(separation, prefer_first_real=True),
                label="Modelo de separación",
                info="Si no hay un .onnx en mdx_models, queda la separación local.",
            )
        separate_btn = gr.Button("Separá", variant="primary")
        with gr.Row():
            vocal = gr.Audio(label="Voz", type="filepath")
            instrumental = gr.Audio(label="Instrumental", type="filepath")
        gr.Markdown("### Convertir la voz")
        gr.Markdown(
            "Elegí un `.pth` de `rvc_models`. Para convertir también hacen falta la carpeta `hubert_base` (con config.json) y `rmvpe.pt`."
        )
        with gr.Row():
            model = gr.Dropdown(
                choices=models,
                value=_choice_value(models),
                label="Modelo RVC",
            )
            index = gr.Dropdown(
                choices=indexes,
                value="",
                label="Índice",
                info="Opcional. Si no tenés un .index, dejá Sin índice.",
            )
        with gr.Row():
            refresh_btn = gr.Button("Actualizá los modelos")
            convert_btn = gr.Button("Convertí la voz", variant="primary")
        converted = gr.Audio(label="Voz convertida", type="filepath")
        gr.Markdown("### Unir")
        use_converted = gr.Checkbox(label="Usar la voz convertida", value=True)
        with gr.Row():
            vocal_db = gr.Slider(
                minimum=-24,
                maximum=12,
                value=0,
                step=0.5,
                label="Nivel de la voz (dB)",
            )
            instrumental_db = gr.Slider(
                minimum=-24,
                maximum=12,
                value=0,
                step=0.5,
                label="Nivel del instrumental (dB)",
            )
        remix_btn = gr.Button("Uní", variant="primary")
        remix = gr.Audio(label="Remix", type="filepath")
        with gr.Row():
            export_btn = gr.Button("Exportá a Descargas", variant="primary")
            open_btn = gr.Button("Abrí la carpeta")

        download_btn.click(
            on_download,
            inputs=[url],
            outputs=[audio_in, vocal, instrumental, converted, remix, status],
            api_name="bajar",
        )
        separate_btn.click(
            on_separate,
            inputs=[audio_in, url, stem, mdx],
            outputs=[audio_in, vocal, instrumental, converted, remix, status],
            api_name="separar",
        )
        convert_btn.click(
            on_convert,
            inputs=[vocal, model, index],
            outputs=[converted, remix, status],
            api_name="convertir",
        )
        refresh_btn.click(
            on_refresh,
            outputs=[model, index, mdx, status],
            api_name="modelos",
        )
        remix_btn.click(
            on_remix,
            inputs=[vocal, instrumental, converted, use_converted, vocal_db, instrumental_db],
            outputs=[remix, status],
            api_name="unir",
        )
        export_btn.click(
            on_export,
            inputs=[stem, vocal, instrumental, converted, remix, use_converted],
            outputs=[status],
            api_name="exportar",
        )
        open_btn.click(
            on_open_folder,
            outputs=[status],
            api_name=False,
        )
        demo.load(
            model_updates,
            outputs=[model, index, mdx],
            api_visibility="private",
        )
    return demo


if __name__ == "__main__":
    if not os.environ.get(TOKEN_ENV):
        os.environ[TOKEN_ENV] = secrets.token_urlsafe(32)
    token = os.environ[TOKEN_ENV]
    print(f"http://127.0.0.1:{PORT}/?{TOKEN_QUERY}={token}")
    build_server().launch(**launch_kwargs())
