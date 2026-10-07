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
    ROOT = os.path.join(data_home(), "Voces")
    PATHS = {
        "uvr": os.path.join(ROOT, "models", "uvr"),
        "rvc": os.path.join(ROOT, "models", "rvc"),
        "rvc_voices": os.path.join(ROOT, "models", "rvc_voices"),
        "voices": os.path.join(ROOT, "voices"),
    }
    INDEX = os.path.join(ROOT, "library.json")


_apply_paths()


def seed_support_weights():
    """Copy bundled RVC support weights into DATA on first .app launch."""
    from app_env import data_dir, home

    src = os.path.join(home(), "library", "models", "rvc")
    dst = os.path.join(data_dir(), "Voces", "models", "rvc")
    src_mark = os.path.join(src, "hubert_base", "config.json")
    dst_mark = os.path.join(dst, "hubert_base", "config.json")
    if not os.path.isfile(src_mark):
        return
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if os.path.isfile(dst_mark):
        return
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def ensure_dirs():
    seed_support_weights()
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
    raw = str(name or os.path.basename(src_path)).replace("\\", "/")
    if ".." in raw or raw.startswith("/") or (len(raw) > 1 and raw[1] == ":"):
        raise ValueError("Nombre de archivo inválido.")
    filename = os.path.basename(raw)
    if not filename or filename in (".", ".."):
        raise ValueError("Nombre de archivo inválido.")
    dest = os.path.join(dest_dir, filename)
    dest_abs = os.path.abspath(dest)
    dest_root = os.path.abspath(dest_dir)
    if os.path.commonpath([dest_root, dest_abs]) != dest_root:
        raise ValueError("Nombre de archivo inválido.")
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


def is_installed_voice(path):
    """True when path is already Voces/<nombre>/<nombre>.pth or the legacy flat folder."""
    if not path or not os.path.isfile(path):
        return False
    root = os.path.abspath(ROOT)
    abs_path = os.path.abspath(path)
    try:
        if os.path.commonpath([root, abs_path]) != root:
            return False
    except ValueError:
        return False
    parts = os.path.relpath(abs_path, root).split(os.sep)
    if len(parts) == 3 and parts[0] == "models" and parts[1] == "rvc_voices":
        return True
    if len(parts) == 2 and os.path.splitext(parts[1])[0] == parts[0]:
        return parts[1].lower().endswith((".pth", ".pt", ".safetensors"))
    return False


def place_named(src_path, stem, suffix):
    """Copy into Voces/<stem>/<stem><suffix>. Same path is a no-op."""
    if not src_path or not os.path.isfile(src_path):
        raise ValueError("No hay archivo para guardar en la biblioteca.")
    if suffix not in (".pth", ".index"):
        raise ValueError("Nombre de archivo inválido.")
    raw = str(stem or "").replace("\\", "/").strip()
    if not raw or raw in (".", "..") or "/" in raw or ".." in raw:
        raise ValueError("Nombre de archivo inválido.")
    dest_dir = os.path.join(ROOT, raw)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, raw + suffix)
    if os.path.abspath(src_path) != os.path.abspath(dest):
        shutil.copy2(src_path, dest)
    return dest


def place_voice_file(src_path, stem):
    """Copy a model into Voces/<stem>/<stem>.pth. Same path is a no-op."""
    return place_named(src_path, stem, ".pth")


def place_beside(model_path, src_path, suffix):
    """Copy src next to an installed model, using the model's stem."""
    if not model_path or not os.path.isfile(model_path):
        raise ValueError("No hay modelo para acompañar el índice.")
    if not src_path or not os.path.isfile(src_path):
        raise ValueError("No hay archivo para guardar en la biblioteca.")
    if suffix not in (".index", ".pth"):
        raise ValueError("Nombre de archivo inválido.")
    stem = os.path.splitext(os.path.basename(model_path))[0]
    if not stem or stem in (".", "..") or "/" in stem or ".." in stem:
        raise ValueError("Nombre de archivo inválido.")
    dest = os.path.join(os.path.dirname(os.path.abspath(model_path)), stem + suffix)
    if os.path.abspath(src_path) != os.path.abspath(dest):
        shutil.copy2(src_path, dest)
    return dest


def list_rvc_voices():
    items = _scan_dir("rvc_voices", (".pth", ".pt", ".safetensors"))
    by_name = {item["name"]: item for item in items}
    if os.path.isdir(ROOT):
        for name in sorted(os.listdir(ROOT)):
            if name == "models" or name.startswith("."):
                continue
            path = os.path.join(ROOT, name, f"{name}.pth")
            if os.path.isfile(path):
                by_name[name] = {"name": name, "path": path, "kind": "rvc_voices"}
    found = [by_name[name] for name in sorted(by_name)]
    for item in found:
        item["index"] = find_index_for_model(item["path"])
    return found


def list_voices():
    return _scan_dir("voices", (".wav", ".mp3", ".flac", ".m4a"))


_RECENT_EXTS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
}


def recent_media(limit=12):
    """Newest local audio/video the train tab can pick without a new upload."""
    found = {}

    def add(path, label=None):
        if not path or not os.path.isfile(path):
            return
        if os.path.splitext(path)[1].lower() not in _RECENT_EXTS:
            return
        abspath = os.path.abspath(path)
        try:
            mtime = os.path.getmtime(abspath)
        except OSError:
            return
        found[abspath] = (mtime, label or os.path.basename(abspath))

    meta = get_session_meta()
    for key, prefix in (
        ("last_vocal_path", "Voz separada"),
        ("last_audio_path", "Último audio"),
        ("last_video_path", "Último video"),
    ):
        path = meta.get(key)
        if path:
            add(path, f"{prefix}: {os.path.basename(path)}")
    downloads = os.path.join(data_home(), "downloads")
    if os.path.isdir(downloads):
        for name in os.listdir(downloads):
            add(os.path.join(downloads, name))
    voices = PATHS.get("voices")
    if voices and os.path.isdir(voices):
        for name in os.listdir(voices):
            add(os.path.join(voices, name), f"Biblioteca: {name}")
    ranked = sorted(found.items(), key=lambda item: item[1][0], reverse=True)
    return [(label, path) for path, (_mtime, label) in ranked[:limit]]


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


def uvr_dir():
    ensure_dirs()
    legacy = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mdx_models")
    if os.path.isdir(legacy):
        return legacy
    return PATHS["uvr"]
