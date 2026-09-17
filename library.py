"""On-disk library of models and voice refs. No Hugging Face."""
import json
import os
import shutil
import time
import uuid

def running_in_app_bundle():
    from app_env import in_app_bundle

    return in_app_bundle()


def data_home():
    """AUDIO_SEPARATOR_DATA, or App Support inside .app, else package dir."""
    from app_env import data_dir

    return data_dir()


def _apply_paths():
    global ROOT, PATHS, INDEX
    ROOT = os.path.join(data_home(), "library")
    PATHS = {
        "uvr": os.path.join(ROOT, "models", "uvr"),
        "rvc": os.path.join(ROOT, "models", "rvc"),
        "xtts": os.path.join(ROOT, "models", "xtts"),
        "rvc_voices": os.path.join(ROOT, "models", "rvc_voices"),
        "voices": os.path.join(ROOT, "voices"),
    }
    INDEX = os.path.join(ROOT, "library.json")


_apply_paths()


def ensure_dirs():
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)
    if not os.path.isfile(INDEX):
        _save({"items": []})


def _save(data):
    os.makedirs(ROOT, exist_ok=True)
    with open(INDEX, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def _load():
    ensure_dirs()
    try:
        with open(INDEX, encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return {"items": []}
        data.setdefault("items", [])
        data.setdefault("session", {})
        return data
    except Exception:
        return {"items": [], "session": {}}


def set_session_meta(**fields):
    """Persist small session keys (e.g. last YouTube video path)."""
    data = _load()
    session = data.setdefault("session", {})
    for key, value in fields.items():
        if value is None:
            session.pop(key, None)
        else:
            session[key] = value
    _save(data)
    return session


def get_session_meta():
    return dict((_load().get("session") or {}))


def register(kind, src_path, name=None):
    ensure_dirs()
    if kind not in PATHS:
        raise ValueError("Tipo de biblioteca desconocido.")
    if not src_path or not os.path.isfile(src_path):
        raise ValueError("No hay archivo para guardar en la biblioteca.")
    dest_dir = PATHS[kind]
    filename = name or os.path.basename(src_path)
    dest = os.path.join(dest_dir, filename)
    if os.path.abspath(src_path) != os.path.abspath(dest):
        shutil.copy2(src_path, dest)
    data = _load()
    items = data.setdefault("items", [])
    stem = os.path.splitext(filename)[0]
    dest_abs = os.path.abspath(dest)
    # Replace only the same destination file (pth and index share stem).
    items = [
        old
        for old in items
        if not (
            old.get("kind") == kind
            and os.path.abspath(old.get("path") or "") == dest_abs
        )
    ]
    item = {
        "id": uuid.uuid4().hex[:12],
        "kind": kind,
        "name": stem,
        "path": dest,
        "created": int(time.time()),
    }
    items.append(item)
    data["items"] = items
    _save(data)
    return item


def _scan_dir(kind, extensions):
    ensure_dirs()
    folder = PATHS[kind]
    found = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and name.lower().endswith(extensions):
            found.append({"name": os.path.splitext(name)[0], "path": path, "kind": kind})
    return found


def find_index_for_model(model_path):
    if not model_path:
        return None
    base = os.path.splitext(model_path)[0]
    for candidate in (f"{base}.index", f"{base}_added.index"):
        if os.path.isfile(candidate):
            return candidate
    folder = os.path.dirname(model_path)
    stem = os.path.basename(base).lower()
    if not os.path.isdir(folder):
        return None
    for name in sorted(os.listdir(folder)):
        lower = name.lower()
        if lower.endswith(".index") and stem in lower:
            return os.path.join(folder, name)
    return None


def list_rvc_voices():
    items = _scan_dir("rvc_voices", (".pth", ".pt", ".safetensors"))
    for item in items:
        item["index"] = find_index_for_model(item["path"])
    return items


def list_voices():
    return _scan_dir("voices", (".wav", ".mp3", ".flac", ".m4a"))


def dropdown_choices(items):
    labeled = []
    for item in items:
        label = item["name"]
        if item.get("index"):
            label = f"{label} (+index)"
        labeled.append((label, item["path"]))
    return labeled


def rvc_support_dir():
    ensure_dirs()
    return PATHS["rvc"]


def xtts_dir():
    ensure_dirs()
    return PATHS["xtts"]


def uvr_dir():
    ensure_dirs()
    legacy = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mdx_models")
    if os.path.isdir(legacy):
        return legacy
    return PATHS["uvr"]
