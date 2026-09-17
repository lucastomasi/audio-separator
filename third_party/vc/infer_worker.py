#!/usr/bin/env python3
"""Long-lived conversion worker. JSON lines on stdin, JSON lines on stdout.

Logs go to stderr. Do not print engine chatter on stdout.
"""
from __future__ import annotations

import json
import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

_REPLY = sys.stdout
sys.stdout = sys.stderr


def _reply(payload: dict) -> None:
    _REPLY.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _REPLY.flush()


def main() -> int:
    from rvc.infer.infer import VoiceConverter

    converter = VoiceConverter()
    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            job = json.loads(raw)
        except json.JSONDecodeError as exc:
            _reply({"ok": False, "error": f"json: {exc}"})
            continue
        cmd = job.get("cmd") or "infer"
        if cmd == "quit":
            _reply({"ok": True, "quit": True})
            return 0
        if cmd == "ping":
            _reply({"ok": True, "ping": True})
            continue
        try:
            out = os.path.abspath(job["output"])
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            converter.convert_audio(
                audio_input_path=os.path.abspath(job["input"]),
                audio_output_path=out,
                model_path=os.path.abspath(job["pth"]),
                index_path=os.path.abspath(job["index"]) if job.get("index") else "",
                pitch=int(job.get("pitch") or 0),
                f0_method=job.get("f0_method") or "rmvpe",
                index_rate=float(job.get("index_rate") or 0.75),
                protect=float(job.get("protect") or 0.5),
                split_audio=bool(job.get("split_audio", True)),
                embedder_model=job.get("embedder") or "contentvec",
                export_format="WAV",
                clean_audio=False,
                post_process=False,
            )
            if not os.path.isfile(out):
                wav = os.path.splitext(out)[0] + ".wav"
                out = wav if os.path.isfile(wav) else out
            if not os.path.isfile(out) or os.path.getsize(out) <= 44:
                _reply({"ok": False, "error": "no audio file"})
                continue
            _reply({"ok": True, "path": out})
        except Exception as exc:
            traceback.print_exc()
            _reply({"ok": False, "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
