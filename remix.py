"""Mix vocal and instrumental stems and write a wav. No Gradio."""
import os

import numpy as np
import soundfile as sf
from pedalboard import Gain, Pedalboard

from app_paths import data_dir
from audio_io import load

SAMPLE_RATE = 44100
REMIX_DIR = os.path.join(data_dir(), "remix_output")


def as_stereo(wave):
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim == 1:
        return np.stack([wave, wave], axis=0)
    if wave.shape[0] == 1:
        return np.concatenate([wave, wave], axis=0)
    return np.ascontiguousarray(wave[:2], dtype=np.float32)


def apply_gain(wave, gain_db, sample_rate):
    wave = np.ascontiguousarray(wave, dtype=np.float32)
    if abs(float(gain_db)) < 1e-6:
        return wave
    board = Pedalboard([Gain(gain_db=float(gain_db))])
    processed = np.asarray(board(wave, int(sample_rate)), dtype=np.float32)
    if processed.ndim == 1:
        processed = processed.reshape(1, -1)
    return np.ascontiguousarray(processed, dtype=np.float32)


def match_length(left, right):
    length = max(left.shape[-1], right.shape[-1])

    def pad(wave):
        if wave.shape[-1] == length:
            return wave
        out = np.zeros((wave.shape[0], length), dtype=np.float32)
        out[:, : wave.shape[-1]] = wave
        return out

    return pad(left), pad(right)


def limit_peak(wave, ceiling=0.99):
    if wave.size == 0:
        return wave
    peak = float(np.max(np.abs(wave)))
    if peak > ceiling:
        return wave * np.float32(ceiling / peak)
    return wave


def mix_arrays(vocal, instrumental, vocal_db=0.0, instrumental_db=0.0, sample_rate=SAMPLE_RATE):
    vocal = apply_gain(as_stereo(vocal), vocal_db, sample_rate)
    instrumental = apply_gain(as_stereo(instrumental), instrumental_db, sample_rate)
    vocal, instrumental = match_length(vocal, instrumental)
    return limit_peak(vocal + instrumental)


def write_wav(path, wave, sample_rate):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    data = np.asarray(wave, dtype=np.float32)
    if data.ndim == 1:
        sf.write(path, data, sample_rate, subtype="PCM_16")
    else:
        sf.write(path, np.ascontiguousarray(data.T), sample_rate, subtype="PCM_16")
    return path


def mix_stems(
    vocal_path,
    instrumental_path,
    vocal_db=0.0,
    instrumental_db=0.0,
    output_path=None,
):
    if not vocal_path or not os.path.isfile(vocal_path):
        raise ValueError("Falta la pista de voz para unir.")
    if not instrumental_path or not os.path.isfile(instrumental_path):
        raise ValueError("Falta la pista instrumental para unir.")
    vocal, sample_rate = load(vocal_path, mono=False, sr=SAMPLE_RATE)
    instrumental, _sample_rate = load(instrumental_path, mono=False, sr=SAMPLE_RATE)
    if np.asarray(vocal).size == 0 or np.asarray(instrumental).size == 0:
        raise ValueError("Una de las pistas está vacía.")
    mixed = mix_arrays(
        vocal,
        instrumental,
        vocal_db=vocal_db,
        instrumental_db=instrumental_db,
        sample_rate=sample_rate,
    )
    if output_path is None:
        os.makedirs(REMIX_DIR, exist_ok=True)
        stem = os.path.splitext(os.path.basename(vocal_path))[0]
        output_path = os.path.join(REMIX_DIR, f"{stem}-remix.wav")
    return write_wav(output_path, mixed, sample_rate)
