"""Download YouTube as audio-only mp3. No Gradio/torch."""
import os
import re
import shutil
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
