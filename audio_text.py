"""Pure helpers for YouTube URLs and stem choices. No Gradio/torch."""
import re

STEM_SOLO_VOZ = "solo_voz"
STEM_SOLO_INST = "solo_instrumental"
STEM_AMBAS = "ambas"


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
