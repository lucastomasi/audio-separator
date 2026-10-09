"""Copy finished stems to a folder the user can open. No Gradio/torch."""
import os
import re
import shutil
import subprocess

from app_env import data_dir, package_dir

_ROLE_SUFFIXES = frozenset({"voz", "instrumental", "unir", "rvc"})
_UNSAFE_NAME = re.compile(r"[^\w\s.-]+", flags=re.UNICODE)
_SPACE_RUN = re.compile(r"[\s_]+")


def is_inside(path, root):
    try:
        path = os.path.realpath(path)
        root = os.path.realpath(root)
        return os.path.commonpath([path, root]) == root
    except (ValueError, OSError):
        return False


def exports_dir():
    path = os.path.join(os.path.expanduser("~"), "Downloads", "Audio Separator")
    os.makedirs(path, exist_ok=True)
    return os.path.realpath(path)


def unique_path(directory, filename):
    filename = os.path.basename(str(filename or "").replace("\\", "/"))
    if not filename or filename in {".", ".."}:
        filename = "audio"
    base, ext = os.path.splitext(filename)
    dest = os.path.join(directory, filename)
    index = 1
    while os.path.exists(dest):
        index += 1
        dest = os.path.join(directory, f"{base} ({index}){ext}")
    return dest


def export_stem(path_or_name):
    """Single path segment from a file path, Gradio dict, or label."""
    raw = path_or_name
    if isinstance(raw, dict):
        raw = raw.get("orig_name") or raw.get("name") or raw.get("path") or ""
    name = os.path.splitext(os.path.basename(str(raw or "").replace("\\", "/")))[0]
    name = _SPACE_RUN.sub("-", _UNSAFE_NAME.sub("", name)).strip("-.")
    return name or "audio"


def song_stem(path_or_name):
    """Song id without trailing -voz / -instrumental / -unir."""
    parts = [part for part in export_stem(path_or_name).split("-") if part]
    while len(parts) > 1 and parts[-1].lower() in _ROLE_SUFFIXES:
        parts.pop()
    return "-".join(parts) or "audio"


def export_label(*parts):
    """Download name without extension. Always one path segment."""
    chunks = []
    for part in parts:
        text = export_stem(part)
        if text:
            chunks.append(text)
    return os.path.basename("-".join(chunks) or "audio")


def _export_roots():
    home = data_dir()
    pkg = package_dir()
    return [
        os.path.join(home, "Trabajos"),
        os.path.join(home, "clean_song_output"),
        os.path.join(home, "remix_output"),
        os.path.join(home, "rvc_output"),
        os.path.join(home, "downloads"),
        os.path.join(home, "Voces"),
        os.path.join(pkg, "remix_output"),
        exports_dir(),
    ]


def is_exportable(path):
    if not path or not os.path.isfile(path):
        return False
    return any(is_inside(path, root) for root in _export_roots())


def copy_to_downloads(file_paths, labels):
    directory = exports_dir()
    os.makedirs(directory, exist_ok=True)
    copied = []
    for path, label in zip(file_paths, labels):
        if not is_exportable(path):
            continue
        ext = os.path.splitext(path)[1] or ".wav"
        dest = unique_path(directory, f"{label}{ext}")
        shutil.copy2(os.path.realpath(path), dest)
        copied.append(dest)
    return directory, copied


def open_exports_dir():
    directory = exports_dir()
    subprocess.Popen(["open", directory])
    return directory
