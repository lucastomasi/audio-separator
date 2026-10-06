"""Source files vs writable data.

The Mac app sets AUDIO_SEPARATOR_HOME so models and outputs stay in
~/Library/Application Support/Audio Separator, outside the .app bundle.
A checkout without that variable keeps everything next to the source.
"""
import os

_BLOCKED_HOMES = frozenset(
    {
        "/",
        "/tmp",
        "/var/tmp",
        "/private/tmp",
        "/private/var/tmp",
        "/etc",
        "/usr",
        "/bin",
        "/sbin",
        "/dev",
        "/System",
        "/Library",
        "/var",
        "/Users",
        "/home",
        "/root",
    }
)


def source_dir():
    return os.path.dirname(os.path.abspath(__file__))


def is_inside(path, root):
    """True when path is inside root after resolving symlinks."""
    try:
        path = os.path.realpath(path)
        root = os.path.realpath(root)
        return os.path.commonpath([path, root]) == root
    except (ValueError, OSError):
        return False


def _forbidden_home(path):
    real = os.path.realpath(path)
    if real in _BLOCKED_HOMES:
        return True
    home = os.path.expanduser("~")
    if home and os.path.realpath(home) == real:
        return True
    return False


def data_dir():
    override = os.environ.get("AUDIO_SEPARATOR_HOME", "").strip()
    if override:
        path = os.path.abspath(os.path.expanduser(override))
        if _forbidden_home(path):
            raise ValueError("AUDIO_SEPARATOR_HOME no es una carpeta válida.")
        os.makedirs(path, exist_ok=True)
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass
        return os.path.realpath(path)
    path = source_dir()
    os.makedirs(path, exist_ok=True)
    return path
