"""Thin Python API over the vendored RVC stack. No pip rvc / rvc-python.

Uses library/models/rvc_voices/ only (same as Convertir in the app).
Mac Intel: infer_rvc_python BaseLoader with only_cpu=True.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Must run before importing torch via rvc_engine (OpenMP / desktop clash).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    os.chdir(ROOT)
except OSError:
    pass

from library import PATHS, find_index_for_model, list_rvc_voices
from rvc_engine import convert_voice

VOICE_DIR = Path(PATHS["rvc_voices"])
OUT_DIR = Path.home() / "Downloads" / "Audio Separator"


def list_voices() -> list[str]:
    """Voice names available under library/models/rvc_voices/."""
    return [item["name"] for item in list_rvc_voices()]


def resolve_voice(voice: str) -> tuple[Path, Path | None]:
    """Return (pth, index) for a voice name. Index may be None."""
    stem = voice.strip()
    if stem.lower().endswith(".pth"):
        stem = Path(stem).stem
    pth = VOICE_DIR / f"{stem}.pth"
    if not pth.is_file():
        raise FileNotFoundError(
            f"No está {pth}. Voces: {', '.join(list_voices()) or '(ninguna)'}"
        )
    index = find_index_for_model(str(pth))
    if index is None:
        # Prefer added_* sibling if find_index missed it.
        added = sorted(VOICE_DIR.glob(f"{stem}*added*.index")) or sorted(
            VOICE_DIR.glob(f"{stem}*.index")
        )
        index = str(added[0]) if added else None
    return pth, Path(index) if index else None


def _conflicting_rvc_processes() -> list[str]:
    """Other Audio_separator torch processes that clash with this CLI on Intel Mac."""
    try:
        import subprocess

        out = subprocess.check_output(["pgrep", "-lf", "python"], text=True)
    except Exception:
        return []
    me = str(os.getpid())
    hits = []
    for line in out.splitlines():
        if me in line.split(None, 1)[:1]:
            continue
        if "Audio_separator" not in line:
            continue
        if "desktop.py" in line or "rvc_api.py" in line:
            # skip this process if pgrep matched ourselves oddly
            if f" {me} " in f" {line} " or line.startswith(me + " "):
                continue
            hits.append(line.strip())
    return hits


def require_exclusive_cli() -> None:
    """Abort before loading RMVPE if desktop/another CLI is already up.

    Two torch/OpenMP processes on this Mac Intel → intermittent SIGSEGV.
    """
    import occupancy

    snap = occupancy.snapshot()
    if snap is not None and snap.holder == occupancy.HOLD_TRAIN:
        print(
            "ERROR: "
            + occupancy.blocked_message(snap)
            + "\nCerrá Convertir en la app o esperá a que termine el train.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
    hits = _conflicting_rvc_processes()
    if not hits:
        return
    print(
        "ERROR: hay otro proceso de Audio Separator usando RVC/torch:\n  "
        + "\n  ".join(hits)
        + "\n\nCerrá la app (desktop.py) o el otro rvc_api y volvé a intentar.\n"
        "Con la app abierta, usá solo Convertir en la UI — no el CLI.\n\n"
        "Comando correcto:\n"
        "  ./convert_rvc.sh <audio.wav> <voz> 0",
        file=sys.stderr,
        flush=True,
    )
    raise SystemExit(1)


def rvc_convert(
    input_wav: str | Path,
    voice: str = "smoke_voice",
    *,
    pitch: int = 0,
    index_rate: float = 0.66,
    f0_method: str = "rmvpe",
    protect: float = 0.33,
    output_wav: str | Path | None = None,
) -> Path:
    """Convert a wav with a trained library voice.

    Reuses the shared BaseLoader (HuBERT/RMVPE stay warm across calls).
    Writes under ~/Downloads/Audio Separator/ unless output_wav is set.
    """
    input_path = Path(input_wav)
    if not input_path.is_absolute():
        input_path = (ROOT / input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    pth, index = resolve_voice(voice)
    if output_wav is None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = OUT_DIR / f"{input_path.stem}_{pth.stem}.wav"
    else:
        out = Path(output_wav)
        out.parent.mkdir(parents=True, exist_ok=True)

    raw = convert_voice(
        str(input_path),
        str(pth),
        str(index) if index else None,
        pitch=pitch,
        index_rate=index_rate,
        f0_method=f0_method,
        protect=protect,
        copy_downloads=False,
    )
    raw_path = Path(raw)
    if raw_path.resolve() != out.resolve():
        out.write_bytes(raw_path.read_bytes())
    return out


def warm_converter() -> None:
    from vc_runner import ensure_vc_engine

    ensure_vc_engine()


def _print_usage() -> None:
    print("Uso (NO ejecutes la carpeta; hacé cd al repo o al .app):")
    print("  ./convert_rvc.sh <input.wav> [voice_name] [pitch]")
    print("  # o: python rvc_api.py <input.wav> [voice_name] [pitch]")
    print("Voces:", ", ".join(list_voices()) or "(ninguna)")
    print("Pitch: rmvpe (no harvest). Cerrá desktop.py antes del CLI.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _print_usage()
        raise SystemExit(2)
    require_exclusive_cli()
    inp = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "smoke_voice"
    pitch_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    result = rvc_convert(inp, voice=name, pitch=pitch_arg)
    print(result)
