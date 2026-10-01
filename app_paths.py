"""Source files vs writable data.

The Mac app sets AUDIO_SEPARATOR_HOME so models and outputs stay in
~/Library/Application Support/Audio Separator, outside the .app bundle.
A checkout without that variable keeps everything next to the source.
"""
import os


def source_dir():
    return os.path.dirname(os.path.abspath(__file__))


def data_dir():
    override = os.environ.get("AUDIO_SEPARATOR_HOME", "").strip()
    if override:
        path = os.path.abspath(os.path.expanduser(override))
    else:
        path = source_dir()
    os.makedirs(path, exist_ok=True)
    return path
