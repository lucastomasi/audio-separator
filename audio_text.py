"""Pure helpers for YouTube URLs and stem choices. No Gradio/torch."""
import re
from urllib.parse import urlparse

STEM_SOLO_VOZ = "solo_voz"
STEM_SOLO_INST = "solo_instrumental"
STEM_AMBAS = "ambas"

YOUTUBE_ID_RE = re.compile(
    r"(?:v=|/youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})"
)
YOUTUBE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "music.youtube.com",
        "youtu.be",
    }
)


def stem_choice_to_list(stem):
    if isinstance(stem, (list, tuple, set)):
        return [item for item in stem if item]
    if stem == STEM_SOLO_INST:
        return ["background"]
    if stem == STEM_AMBAS:
        return ["vocal", "background"]
    return ["vocal"]


def normalize_media_url(url_media):
    url_media = (url_media or "").strip().strip('"').strip("'")
    if not url_media:
        return ""
    found = re.search(r"https?://[^\s]+", url_media)
    if found:
        url_media = found.group(0).rstrip(".,);]")
    elif url_media.startswith("www.") or re.match(
        r"^(youtu\.be|youtube\.com|music\.youtube\.com)/", url_media, re.I
    ):
        url_media = "https://" + url_media
    return url_media


def extract_youtube_id(url):
    url = normalize_media_url(url)
    if not url:
        return None
    match = YOUTUBE_ID_RE.search(url)
    return match.group(1) if match else None


def youtube_watch_url(url):
    """Return a reconstructed youtube.com watch URL or raise ValueError."""
    url = normalize_media_url(url)
    if not url:
        raise ValueError("Pega un enlace de YouTube.")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Pega un enlace de YouTube.")
    if parsed.username or parsed.password:
        raise ValueError("Pega un enlace de YouTube.")
    host = (parsed.hostname or "").lower().rstrip(".")
    if host not in YOUTUBE_HOSTS:
        raise ValueError("Pega un enlace de YouTube.")
    video_id = extract_youtube_id(url)
    if not video_id:
        raise ValueError("Pega un enlace de YouTube.")
    return f"https://www.youtube.com/watch?v={video_id}", video_id
