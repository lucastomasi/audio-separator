"""Local audio helpers so we don't need librosa (and its numba/llvmlite stack)."""
import json
import math
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def get_duration(filename=None, path=None):
    audio_path = filename or path
    try:
        with sf.SoundFile(audio_path) as handle:
            return len(handle) / float(handle.samplerate)
    except Exception:
        probe = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                audio_path,
            ],
            stderr=subprocess.STDOUT,
        )
        return float(json.loads(probe)["format"]["duration"])


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


def _as_path(path):
    if isinstance(path, dict):
        path = path.get("path") or path.get("name") or path.get("orig_name")
    return path


def load(path, mono=False, sr=44100):
    path = _as_path(path)
    try:
        wave, file_sr = sf.read(path, always_2d=True)
    except Exception:
        stereo_path = f"{path}.decoded.wav"
        subprocess.check_call(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                path,
                "-ac",
                "2",
                "-ar",
                str(sr or 44100),
                stereo_path,
            ]
        )
        wave, file_sr = sf.read(stereo_path, always_2d=True)
    wave = wave.T.astype(np.float32)
    if sr and int(file_sr) != int(sr):
        wave = resample(wave, orig_sr=file_sr, target_sr=sr)
        file_sr = sr
    if mono:
        return np.mean(wave, axis=0), file_sr
    if wave.shape[0] == 1:
        return wave[0], file_sr
    return wave, file_sr
