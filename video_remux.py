"""Put new audio on a video (stream copy) or a still image (one picture, song length)."""
from __future__ import annotations

import os
import subprocess

from youtube_lib import ffmpeg_binary

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def is_still_image(path):
    return os.path.splitext(path or "")[1].lower() in IMAGE_EXTS


def _require_inputs(media_path, audio_path, missing_media):
    if not media_path or not os.path.isfile(media_path):
        raise ValueError(missing_media)
    if not audio_path or not os.path.isfile(audio_path):
        raise ValueError("Falta el audio nuevo para pegar.")
    ffmpeg = ffmpeg_binary()
    if not ffmpeg:
        raise ValueError("No encuentro ffmpeg para remux.")
    return ffmpeg


def remux_audio(media_path, audio_path, output_path, shortest=True):
    if is_still_image(media_path):
        return still_image_with_audio(media_path, audio_path, output_path)
    return remux_audio_onto_video(media_path, audio_path, output_path, shortest=shortest)


def still_image_with_audio(image_path, audio_path, output_path):
    ffmpeg = _require_inputs(
        image_path, audio_path, "Falta la imagen (jpg, png, webp)."
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-i",
        image_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        "-ar",
        "48000",
        "-shortest",
        "-movflags",
        "+faststart",
        output_path,
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not os.path.isfile(output_path):
        err = proc.stderr.decode("utf-8", "ignore")[:300]
        raise ValueError(f"No se pudo unir audio a la imagen: {err}")
    return os.path.abspath(output_path)


def remux_audio_onto_video(video_path, audio_path, output_path, shortest=True):
    ffmpeg = _require_inputs(
        video_path,
        audio_path,
        "Falta el video original (descargalo en YouTube o subilo).",
    )
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
