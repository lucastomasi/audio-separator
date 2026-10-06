"""Copy finished stems to a folder the user can open. No Gradio/torch."""
import os
import shutil
import subprocess

from app_paths import data_dir, is_inside


def exports_dir():
    path = os.path.join(os.path.expanduser("~"), "Downloads", "Audio Separator")
    os.makedirs(path, exist_ok=True)
    return os.path.realpath(path)


def unique_path(directory, filename):
    base, ext = os.path.splitext(filename)
    dest = os.path.join(directory, filename)
    index = 1
    while os.path.exists(dest):
        index += 1
        dest = os.path.join(directory, f"{base} ({index}){ext}")
    return dest


def _export_roots():
    home = data_dir()
    return [
        os.path.join(home, "clean_song_output"),
        os.path.join(home, "remix_output"),
        os.path.join(home, "rvc_output"),
        os.path.join(home, "downloads"),
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
