"""Local audio helpers so we don't need librosa (and its numba/llvmlite stack)."""
import json
import math
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from youtube_lib import ffmpeg_binary, ffprobe_binary


def _tool_env():
    return {"PATH": "/usr/bin:/bin", "LANG": "C"}


def _media_path(path):
    if not path:
        raise ValueError("Falta el archivo de audio.")
    real = os.path.realpath(path)
    if not os.path.isfile(real):
        raise ValueError("No encontré el audio.")
    return real


def get_duration(filename=None, path=None):
    audio_path = _media_path(filename or path)
    try:
        with sf.SoundFile(audio_path) as handle:
            return len(handle) / float(handle.samplerate)
    except Exception:
        probe = ffprobe_binary() or "ffprobe"
        output = subprocess.check_output(
            [
                probe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                "-i",
                audio_path,
            ],
            stderr=subprocess.STDOUT,
            env=_tool_env(),
            cwd=tempfile.gettempdir(),
        )
        return float(json.loads(output)["format"]["duration"])


def resample(y, orig_sr, target_sr):
    if int(orig_sr) == int(target_sr):
        return y
    gcd = math.gcd(int(orig_sr), int(target_sr))
    up = int(target_sr) // gcd
    down = int(orig_sr) // gcd
    if y.ndim == 1:
        return resample_poly(y, up, down).astype(np.float32)
    return np.stack(
        [resample_poly(channel, up, down) for channel in y]
    ).astype(np.float32)


def load(path, mono=False, sr=44100):
    audio_path = _media_path(path)
    try:
        wave, file_sr = sf.read(audio_path, always_2d=True)
    except Exception:
        ffmpeg = ffmpeg_binary() or "ffmpeg"
        handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        stereo_path = handle.name
        handle.close()
        try:
            subprocess.check_call(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    audio_path,
                    "-ac",
                    "2",
                    "-ar",
                    str(sr or 44100),
                    stereo_path,
                ],
                env=_tool_env(),
                cwd=tempfile.gettempdir(),
            )
            wave, file_sr = sf.read(stereo_path, always_2d=True)
        finally:
            try:
                os.remove(stereo_path)
            except OSError:
                pass
    wave = wave.T.astype(np.float32)
    if sr and int(file_sr) != int(sr):
        wave = resample(wave, orig_sr=file_sr, target_sr=sr)
        file_sr = sr
    if mono:
        return np.mean(wave, axis=0), file_sr
    if wave.shape[0] == 1:
        return wave[0], file_sr
    return wave, file_sr
