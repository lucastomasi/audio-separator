"""Split one audio into per-speaker clips. Local VAD + MFCC clustering."""
import os
import re
import subprocess

import numpy as np
import soundfile as sf
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

from exports import copy_to_downloads
from youtube_lib import ffmpeg_binary

import audio_io

SILENCE_RE = re.compile(
    r"silence_(?:start|end):\s*([0-9.]+)",
)
MIN_TURN = 0.4
MAX_REF_SEC = 25.0
SR = 16000


def parse_silence_times(ffmpeg_stderr):
    starts, ends = [], []
    for line in (ffmpeg_stderr or "").splitlines():
        if "silence_start:" in line:
            match = re.search(r"silence_start:\s*([0-9.]+)", line)
            if match:
                starts.append(float(match.group(1)))
        elif "silence_end:" in line:
            match = re.search(r"silence_end:\s*([0-9.]+)", line)
            if match:
                ends.append(float(match.group(1)))
    return starts, ends


def speech_regions(duration, starts, ends, min_turn=MIN_TURN):
    events = []
    for t in starts:
        events.append((t, "start"))
    for t in ends:
        events.append((t, "end"))
    events.sort()
    regions = []
    speaking = True
    t0 = 0.0
    for t, kind in events:
        if kind == "start" and speaking:
            if t - t0 >= min_turn:
                regions.append((t0, t))
            speaking = False
        elif kind == "end" and not speaking:
            t0 = t
            speaking = True
    if speaking and duration - t0 >= min_turn:
        regions.append((t0, duration))
    if not regions and duration >= min_turn:
        regions = [(0.0, duration)]
    return regions


def _silencedetect(path, duration):
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
    starts, ends = parse_silence_times(stderr)
    return speech_regions(duration, starts, ends)


def _mfcc_embed(wave, sr):
    if wave.ndim > 1:
        wave = np.mean(wave, axis=0)
    n = max(1, int(sr * 0.025))
    hop = max(1, int(sr * 0.010))
    if wave.size < n:
        wave = np.pad(wave, (0, n - wave.size))
    window = np.hanning(n)
    frames = []
    for start in range(0, max(1, wave.size - n), hop):
        frame = wave[start : start + n] * window
        spec = np.abs(np.fft.rfft(frame, n=512))
        logmel = np.log(spec + 1e-6)
        frames.append(logmel)
    mat = np.stack(frames)
    feat = np.concatenate([mat.mean(axis=0), mat.std(axis=0)])
    return feat.astype(np.float32)


def _cluster(embeddings):
    if len(embeddings) == 1:
        return np.array([0])
    x = normalize(np.stack(embeddings))
    n = len(x)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.55,
        metric="cosine",
        linkage="average",
    )
    labels = model.fit_predict(x)
    if len(set(labels)) > min(8, n):
        model = AgglomerativeClustering(
            n_clusters=min(8, n),
            metric="cosine",
            linkage="average",
        )
        labels = model.fit_predict(x)
    return labels


def _concat_cap(waves, sr, cap_sec=MAX_REF_SEC):
    cap = int(cap_sec * sr)
    chunks = []
    total = 0
    for wave in waves:
        if wave.ndim > 1:
            wave = np.mean(wave, axis=0)
        remain = cap - total
        if remain <= 0:
            break
        take = wave[:remain]
        chunks.append(take)
        total += take.shape[0]
    if not chunks:
        return np.zeros(sr, dtype=np.float32)
    return np.concatenate(chunks)


def detect_speakers(audio_path, out_dir=None):
    if not audio_path or not os.path.isfile(audio_path):
        raise ValueError("Falta el audio para detectar voces.")
    duration = audio_io.get_duration(filename=audio_path)
    regions = _silencedetect(audio_path, duration)
    if not regions:
        raise ValueError("No encontré turnos de habla.")

    wave, sr = audio_io.load(audio_path, mono=True, sr=SR)
    embeddings = []
    snippets = []
    for start, end in regions:
        a = int(start * sr)
        b = int(end * sr)
        snippet = wave[a:b]
        if snippet.size < int(MIN_TURN * sr):
            continue
        embeddings.append(_mfcc_embed(snippet, sr))
        snippets.append(snippet)
    if not embeddings:
        raise ValueError("No encontré turnos de habla.")

    labels = _cluster(embeddings)
    grouped = {}
    for label, snippet in zip(labels, snippets):
        grouped.setdefault(int(label), []).append(snippet)

    if out_dir is None:
        from app_env import data_dir

        out_dir = os.path.join(data_dir(), "Trabajos", "Diarizar")
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for index, label in enumerate(sorted(grouped.keys()), start=1):
        merged = _concat_cap(grouped[label], sr)
        dest = os.path.join(out_dir, f"voz_{index}.wav")
        sf.write(dest, merged, sr)
        paths.append(dest)
    _, copied = copy_to_downloads(
        paths, [f"voz_{i}" for i in range(1, len(paths) + 1)]
    )
    return copied or paths
