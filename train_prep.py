"""Voice-only prep before RVC train. No Gradio."""
import hashlib
import os
import shutil
import subprocess
import tempfile
import uuid

import audio_io
import diarize
import library
from rvc_train import _src_path
from youtube_lib import ffmpeg_binary

# 21 min with vocal + dereverb was 5.4 h on Intel. Vocal-only is one of those two passes.
_REF_SECONDS = 21 * 60
_REF_HOURS = 5.4 / 2
LONG_SEC = 180.0
NO_SPEECH = "No hay habla para entrenar. Paro."
NO_VOCAL = "No salió voz separada. Paro."


def eta_hours(duration_sec):
    seconds = max(0.0, float(duration_sec or 0))
    return seconds / _REF_SECONDS * _REF_HOURS


def names_match(given, channel):
    def norm(value):
        return "".join(c for c in (value or "").casefold() if c.isalnum())

    left, right = norm(given), norm(channel)
    if not left or not right:
        return False
    return left in right or right in left


def silence_seconds(duration, starts, ends):
    total = 0.0
    pending = list(ends)
    span = float(duration)
    for start in starts:
        end = pending.pop(0) if pending else span
        if end < start:
            continue
        total += min(end, span) - max(float(start), 0.0)
    return min(total, span)


def speech_regions_for_train(duration, starts, ends):
    span = float(duration)
    if span <= 0:
        raise ValueError(NO_SPEECH)
    if silence_seconds(span, starts, ends) >= span * 0.95:
        raise ValueError(NO_SPEECH)
    regions = diarize.speech_regions(span, starts, ends)
    spoken = sum(end - start for start, end in regions)
    if spoken < 0.4:
        raise ValueError(NO_SPEECH)
    return regions


def _silence_events(path):
    ffmpeg = ffmpeg_binary()
    if not ffmpeg:
        raise ValueError("No encuentro ffmpeg.")
    cmd = [
        ffmpeg,
        "-i",
        path,
        "-af",
        "silencedetect=noise=-35dB:d=0.35",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = (result.stderr or "") + (result.stdout or "")
    return diarize.parse_silence_times(stderr)


def _write_speech(src, regions):
    ffmpeg = ffmpeg_binary()
    if not ffmpeg:
        raise ValueError("No encuentro ffmpeg.")
    parts = []
    labels = []
    for index, (start, end) in enumerate(regions):
        parts.append(
            f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[s{index}]"
        )
        labels.append(f"[s{index}]")
    filt = ";".join(parts) + f";{''.join(labels)}concat=n={len(regions)}:v=0:a=1[out]"
    dest_dir = tempfile.mkdtemp(prefix="train_prep_")
    dest = os.path.join(dest_dir, "habla.wav")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        src,
        "-filter_complex",
        filt,
        "-map",
        "[out]",
        dest,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not os.path.isfile(dest) or os.path.getsize(dest) == 0:
        raise ValueError(NO_SPEECH)
    return dest


def trim_if_long(path, duration):
    if duration <= LONG_SEC:
        return path, duration
    starts, ends = _silence_events(path)
    regions = speech_regions_for_train(duration, starts, ends)
    spoken = sum(end - start for start, end in regions)
    if spoken >= duration * 0.98:
        return path, duration
    dest = _write_speech(path, regions)
    return dest, audio_io.get_duration(filename=dest)


def _prep_dir():
    from app_env import data_dir

    path = os.path.join(data_dir(), "RVC", "prep")
    os.makedirs(path, exist_ok=True)
    return path


def _keep_one(src):
    dest = os.path.join(_prep_dir(), f"{uuid.uuid4().hex}.wav")
    shutil.copy2(src, dest)
    return dest


def _drop_trim_temp(path):
    parent = os.path.dirname(os.path.abspath(path))
    if os.path.basename(parent).startswith("train_prep_"):
        shutil.rmtree(parent, ignore_errors=True)


def discard_prep_copies():
    from app_env import data_dir

    folder = os.path.join(data_dir(), "RVC", "prep")
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)


def separate_voice_only(path, progress=None):
    from uvr_runtime import output_dir, process_uvr_task

    song_id = hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:12]
    result = process_uvr_task(
        orig_song_path=path,
        main_vocals=False,
        dereverb=False,
        song_id=song_id,
        only_voiceless=False,
        progress=progress,
    )
    vocal = None
    if isinstance(result, (list, tuple)):
        if len(result) >= 5 and result[4]:
            vocal = result[4]
        elif result and result[0]:
            vocal = result[0]
    elif isinstance(result, str):
        vocal = result
    if not vocal or not os.path.isfile(vocal) or os.path.getsize(vocal) == 0:
        raise ValueError(NO_VOCAL)
    kept = _keep_one(vocal)
    song_dir = os.path.join(output_dir, song_id)
    if os.path.isdir(song_dir):
        shutil.rmtree(song_dir, ignore_errors=True)
    _drop_trim_temp(path)
    return os.path.abspath(kept)


def _report(progress, eta, already):
    if progress is None:
        return
    if already:
        text = "ETA estimada ~0.0 h. La voz ya está separada."
    else:
        text = f"ETA estimada ~{eta:.1f} h. Separando solo voz, sin dereverb."
    try:
        progress(0.02, desc=text)
    except TypeError:
        progress(0.02)


def _paths(files):
    found = []
    for item in files or []:
        path = _src_path(item)
        if path and os.path.isfile(path):
            found.append(os.path.abspath(path))
    return found


def assert_channel_matches(exp_name, files):
    name = (exp_name or "").strip()
    if not name:
        raise ValueError("Poné un nombre para la voz.")
    meta = library.get_session_meta()
    channel = meta.get("last_youtube_channel")
    if not channel:
        return
    linked = set()
    for key in ("last_audio_path", "last_video_path"):
        value = meta.get(key)
        if value:
            linked.add(os.path.abspath(value))
    if not linked:
        return
    used = False
    for path in _paths(files):
        if path in linked:
            used = True
            break
    if not used:
        return
    if not names_match(name, channel):
        raise ValueError(f"El canal es «{channel}», no «{name}». Paro.")


def _is_last_vocal(path):
    meta = library.get_session_meta()
    saved = meta.get("last_vocal_path")
    if not saved:
        return False
    return os.path.abspath(path) == os.path.abspath(saved)


def prepare_for_train(files, progress=None):
    paths = _paths(files)
    if not paths:
        raise ValueError("Subí al menos un audio de la voz a entrenar.")
    ready = []
    for path in paths:
        duration = float(audio_io.get_duration(filename=path))
        work, duration = trim_if_long(path, duration)
        already = _is_last_vocal(path)
        _report(progress, eta_hours(duration), already)
        if already:
            if os.path.abspath(work) != os.path.abspath(path):
                kept = _keep_one(work)
                _drop_trim_temp(work)
                ready.append(kept)
            else:
                ready.append(work)
            continue
        ready.append(separate_voice_only(work, progress=progress))
    return ready
