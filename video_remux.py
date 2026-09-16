"""Remux new audio onto original video. Video stream copied (no re-encode)."""
from __future__ import annotations

import os
import subprocess

from youtube_lib import ffmpeg_binary


def remux_audio_onto_video(video_path, audio_path, output_path, shortest=True):
    if not video_path or not os.path.isfile(video_path):
        raise ValueError("Falta el video original (descargalo en YouTube o subilo).")
    if not audio_path or not os.path.isfile(audio_path):
        raise ValueError("Falta el audio nuevo para pegar al video.")
    ffmpeg = ffmpeg_binary()
    if not ffmpeg:
        raise ValueError("No encuentro ffmpeg para remux.")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
    ]
    if shortest:
        cmd.append("-shortest")
    cmd.append(output_path)
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not os.path.isfile(output_path):
        err = proc.stderr.decode("utf-8", "ignore")[:300]
        raise ValueError(f"No se pudo unir audio al video: {err}")
    return os.path.abspath(output_path)
