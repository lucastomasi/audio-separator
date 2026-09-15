"""Download YouTube as audio-only mp3. No Gradio/torch."""
import os
import re
import shutil
import subprocess

from audio_text import normalize_media_url

AUDIO_EXTS = ("mp3", "m4a", "wav", "webm", "opus", "ogg")
YOUTUBE_ID_RE = re.compile(
    r"(?:v=|/youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})"
)


def downloads_dir():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")
    os.makedirs(path, exist_ok=True)
    return path


def extract_youtube_id(url):
    url = normalize_media_url(url)
    if not url:
        return None
    match = YOUTUBE_ID_RE.search(url)
    return match.group(1) if match else None


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


def node_binary():
    found = shutil.which("node")
    if found:
        return found
    fallback = "/usr/local/bin/node"
    return fallback if os.path.isfile(fallback) else None


def ydl_options(directory):
    opts = {
        "format": "bestaudio[vcodec=none]/bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "overwrites": True,
        "noplaylist": True,
        "no_warnings": True,
        "quiet": True,
        "outtmpl": os.path.join(directory, "%(id)s.%(ext)s"),
        "restrictfilenames": True,
        "remote_components": ["ejs:github"],
    }
    node_path = node_binary()
    if node_path:
        opts["js_runtimes"] = {"node": {"path": node_path}}
    ffmpeg_path = ffmpeg_binary()
    if ffmpeg_path:
        opts["ffmpeg_location"] = ffmpeg_path
    return opts


def _playlist_first(info):
    if not info:
        raise ValueError("YouTube no devolvió información.")
    note = None
    if info.get("_type") == "playlist":
        entries = [entry for entry in (info.get("entries") or []) if entry]
        if not entries:
            raise ValueError("Esa lista no tiene videos.")
        info = entries[0]
        note = "Tomé el primer tema de la lista (solo audio)."
    return info, note


def parse_seconds(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return float(text)


def clip_audio(source_path, start=None, end=None, directory=None):
    if not source_path or not os.path.isfile(source_path):
        raise ValueError("Falta el audio para recortar.")
    start_s = parse_seconds(start)
    end_s = parse_seconds(end)
    if start_s is None and end_s is None:
        raise ValueError("Indicá inicio y/o fin para recortar.")
    if start_s is not None and start_s < 0:
        raise ValueError("El inicio no puede ser negativo.")
    if end_s is not None and start_s is not None and end_s <= start_s:
        raise ValueError("El fin tiene que ser después del inicio.")

    directory = directory or downloads_dir()
    os.makedirs(directory, exist_ok=True)
    base = os.path.splitext(os.path.basename(source_path))[0]
    start_tag = int(start_s) if start_s is not None else 0
    end_tag = int(end_s) if end_s is not None else "fin"
    dest = os.path.join(directory, f"{base}_{start_tag}-{end_tag}.wav")
    ffmpeg = ffmpeg_binary()
    if not ffmpeg:
        raise ValueError("No encuentro ffmpeg para recortar.")
    cmd = [ffmpeg, "-y", "-loglevel", "error"]
    if start_s is not None:
        cmd.extend(["-ss", str(start_s)])
    cmd.extend(["-i", source_path])
    if end_s is not None:
        if start_s is not None:
            cmd.extend(["-t", str(end_s - start_s)])
        else:
            cmd.extend(["-to", str(end_s)])
    cmd.extend(["-ac", "1", "-ar", "24000", dest])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not os.path.isfile(dest) or os.path.getsize(dest) == 0:
        raise ValueError("No se pudo recortar el audio.")
    return os.path.abspath(dest)


def download_audio(url, directory=None):
    url = normalize_media_url(url)
    if not url:
        raise ValueError("Pega un enlace de YouTube.")

    directory = directory or downloads_dir()
    os.makedirs(directory, exist_ok=True)

    cached_id = extract_youtube_id(url)
    cached = existing_audio(cached_id, directory)
    if cached:
        return cached, True, None

    import yt_dlp

    try:
        with yt_dlp.YoutubeDL(ydl_options(directory)) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:
        raise ValueError(
            "No se pudo descargar el audio de YouTube. "
            "Revisa el enlace e inténtalo de nuevo."
        ) from exc

    info, playlist_note = _playlist_first(info)
    video_id = info.get("id")
    if not video_id:
        raise ValueError("No pude identificar el video de YouTube.")

    path = existing_audio(video_id, directory)
    if not path:
        raise ValueError("La descarga terminó, pero no encontré el archivo de audio.")
    return path, False, playlist_note
