"""Mix a new vocal onto an instrumental. No Gradio/torch."""
import os

import numpy as np
import pedalboard
import soundfile as sf

import audio_io

SR = 44100


def to_stereo(wave):
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim == 1:
        return np.stack([wave, wave])
    if wave.shape[0] == 1:
        return np.repeat(wave, 2, axis=0)
    return wave[:2]


def delay_ms(wave, samplerate, milliseconds):
    samples = int(round(float(milliseconds) / 1000.0 * samplerate))
    if samples == 0:
        return wave
    if samples > 0:
        pad = np.zeros((wave.shape[0], samples), dtype=np.float32)
        return np.concatenate([pad, wave], axis=1)
    trim = min(-samples, wave.shape[1])
    return wave[:, trim:]


def stretch_to_length(wave, samplerate, target_samples, high_quality=False):
    current = wave.shape[1]
    if current <= 0 or target_samples <= 0 or current == target_samples:
        return wave
    factor = current / float(target_samples)
    stretched = pedalboard.time_stretch(
        wave,
        samplerate=float(samplerate),
        stretch_factor=float(factor),
        high_quality=high_quality,
    )
    stretched = np.asarray(stretched, dtype=np.float32)
    if stretched.ndim == 1:
        stretched = np.stack([stretched, stretched])
    if stretched.shape[1] < target_samples:
        pad = np.zeros(
            (stretched.shape[0], target_samples - stretched.shape[1]),
            dtype=np.float32,
        )
        return np.concatenate([stretched, pad], axis=1)
    return stretched[:, :target_samples]


def db_to_gain(db):
    return float(10 ** (float(db) / 20.0))


def mix_tracks(voice, instrumental, voice_db=0.0, instrumental_db=0.0):
    voice = to_stereo(voice)
    instrumental = to_stereo(instrumental)
    length = max(voice.shape[1], instrumental.shape[1])

    def pad(wave):
        if wave.shape[1] < length:
            extra = np.zeros((wave.shape[0], length - wave.shape[1]), dtype=np.float32)
            return np.concatenate([wave, extra], axis=1)
        return wave[:, :length]

    mixed = pad(voice) * db_to_gain(voice_db) + pad(instrumental) * db_to_gain(
        instrumental_db
    )
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def remix_to_wav(
    voice_path,
    instrumental_path,
    out_path,
    delay_milliseconds=0.0,
    match_duration=True,
    voice_db=0.0,
    instrumental_db=0.0,
):
    if not voice_path:
        raise ValueError("Falta la voz nueva.")
    if not instrumental_path:
        raise ValueError("Falta el instrumental.")

    voice, _ = audio_io.load(voice_path, mono=False, sr=SR)
    instrumental, _ = audio_io.load(instrumental_path, mono=False, sr=SR)
    voice = to_stereo(voice)
    instrumental = to_stereo(instrumental)
    voice = delay_ms(voice, SR, delay_milliseconds)
    if match_duration:
        voice = stretch_to_length(voice, SR, instrumental.shape[1], high_quality=False)
    mixed = mix_tracks(voice, instrumental, voice_db, instrumental_db)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    sf.write(out_path, mixed.T, SR)
    return os.path.abspath(out_path)
