#!/usr/bin/env python3
"""Headless voice conversion CLI. Run with .venv-vc, cwd = this directory."""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pth", required=True)
    parser.add_argument("--index", default="")
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--index-rate", type=float, default=0.75)
    parser.add_argument("--protect", type=float, default=0.5)
    parser.add_argument("--f0-method", default="rmvpe")
    parser.add_argument("--embedder", default="contentvec")
    parser.add_argument("--split-audio", action="store_true")
    args = parser.parse_args()

    for path, label in ((args.input, "input"), (args.pth, "pth")):
        if not os.path.isfile(path):
            print(f"missing {label}: {path}", file=sys.stderr)
            return 2

    from rvc.infer.infer import VoiceConverter

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    converter = VoiceConverter()
    converter.convert_audio(
        audio_input_path=os.path.abspath(args.input),
        audio_output_path=os.path.abspath(args.output),
        model_path=os.path.abspath(args.pth),
        index_path=os.path.abspath(args.index) if args.index else "",
        pitch=args.pitch,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
        protect=args.protect,
        split_audio=args.split_audio,
        embedder_model=args.embedder,
        export_format="WAV",
        clean_audio=False,
        post_process=False,
    )
    if not os.path.isfile(args.output):
        # converter may rewrite extension
        wav = os.path.splitext(args.output)[0] + ".wav"
        if os.path.isfile(wav):
            return 0
        print("conversion produced no file", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
