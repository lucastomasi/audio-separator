"""Download YouTube as audio-only mp3. No Gradio/torch."""
import os
import shutil

from app_paths import data_dir
from audio_text import extract_youtube_id, youtube_watch_url

AUDIO_EXTS = ("mp3", "m4a", "wav", "webm", "opus", "ogg")


def downloads_dir():
    path = os.path.join(data_dir(), "downloads")
    os.makedirs(path, exist_ok=True)
    return path


def existing_audio(video_id, directory=None):
    if not video_id:
        return None
    directory = directory or downloads_dir()
    for ext in AUDIO_EXTS:
        candidate = os.path.join(directory, f"{video_id}.{ext}")
        if os.path.isfile(candidate) and os.path.getsize(candidate) > 0:
            return os.path.abspath(candidate)
    return None


def ffmpeg_binary():
    here = os.path.dirname(os.path.abspath(__file__))
    bundled = os.path.abspath(os.path.join(here, "..", "bin", "ffmpeg"))
    if os.path.isfile(bundled):
        return bundled
    found = shutil.which("ffmpeg")
    if found:
        return found
    home = os.path.expanduser("~/.local/bin/ffmpeg")
    if os.path.isfile(home):
        return home
    return None


def ffprobe_binary():
    here = os.path.dirname(os.path.abspath(__file__))
    bundled = os.path.abspath(os.path.join(here, "..", "bin", "ffprobe"))
    if os.path.isfile(bundled):
        return bundled
    found = shutil.which("ffprobe")
    return found


def ydl_options(directory):
    opts = {
        "format": "bestaudio[vcodec=none]/bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "overwrites": True,
        "noplaylist": True,
        "no_warnings": True,
        "quiet": True,
        "socket_timeout": 15,
        "outtmpl": os.path.join(directory, "%(id)s.%(ext)s"),
        "restrictfilenames": True,
    }
    ffmpeg_path = ffmpeg_binary()
    if ffmpeg_path:
        opts["ffmpeg_location"] = ffmpeg_path
    return opts


def download_audio(url, directory=None):
    watch_url, video_id = youtube_watch_url(url)
    directory = directory or downloads_dir()
    os.makedirs(directory, exist_ok=True)
    cached = existing_audio(video_id, directory)
    if cached:
        return cached, True, None
    import yt_dlp
    try:
        with yt_dlp.YoutubeDL(ydl_options(directory)) as ydl:
            info = ydl.extract_info(watch_url, download=True)
    except Exception as exc:
        raise ValueError(
            "No se pudo descargar el audio de YouTube. Revisa el enlace."
        ) from exc
    if info and info.get("_type") == "playlist":
        raise ValueError("Pega un enlace de YouTube.")
    got_id = (info or {}).get("id")
    if got_id != video_id:
        raise ValueError("No pude identificar el video de YouTube.")
    path = existing_audio(video_id, directory)
    if not path:
        raise ValueError("La descarga terminó, pero no encontré el archivo de audio.")
    return path, False, None
