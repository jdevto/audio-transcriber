#!/usr/bin/env python3
"""Transcribe audio to plain text using faster-whisper."""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

_SUPPORTED_PY = ((3, 10), (3, 11), (3, 12), (3, 13))


def _python_ok() -> bool:
    v = sys.version_info[:2]
    return v in _SUPPORTED_PY


def _default_output_path(audio_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return audio_path.with_name(f"{audio_path.stem}_{ts}.txt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio to a .txt file (local faster-whisper; requires ffmpeg on PATH)."
    )
    parser.add_argument(
        "audio",
        type=Path,
        help="Path to the audio file (e.g. M4A, WAV, MP3).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output text file (default: <input_stem>_YYYYMMDD-HHMMSS.txt beside the input).",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Use <input_stem>.txt as the default output (overwrites same path on rerun).",
    )
    parser.add_argument(
        "--pause-breaks",
        action="store_true",
        help="Blank line between segments when the gap between them is at least --pause-gap seconds.",
    )
    parser.add_argument(
        "--pause-gap",
        type=float,
        default=1.0,
        metavar="SEC",
        help="With --pause-breaks, minimum silence (seconds) to insert a paragraph break. Default: 1.0.",
    )
    parser.add_argument(
        "--flush-minutes",
        type=float,
        default=0.0,
        metavar="M",
        help="If > 0, append new text to the output every M minutes while transcribing; 0 = write only when done (default).",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="base",
        metavar="NAME",
        help="Whisper model size (tiny, base, small, medium, large-v2, large-v3, …). Default: base.",
    )
    parser.add_argument(
        "-l",
        "--language",
        default=None,
        metavar="CODE",
        help="Language code (e.g. en). Omit for automatic detection.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Inference device (e.g. cpu, cuda). Default: cpu.",
    )
    parser.add_argument(
        "--compute-type",
        default="default",
        metavar="TYPE",
        help="Compute type (e.g. int8, float16). Use float16 with cuda. Default: default.",
    )
    args = parser.parse_args()

    if not _python_ok():
        v = f"{sys.version_info.major}.{sys.version_info.minor}"
        print(
            f"error: Python {v} is not supported (need 3.10–3.13; PyPI wheels for "
            "ctranslate2/onnxruntime/av are not available for newer interpreters).\n"
            f"This process: {sys.executable}\n"
            "Run the script with your 3.12 venv (from the project directory), not system python:\n"
            "  .venv/bin/python transcribe.py …\n"
            "If .venv is missing or wrong: rm -rf .venv && python3.12 -m venv .venv\n"
            "  .venv/bin/python -m pip install -U pip && .venv/bin/python -m pip install -r requirements.txt\n"
            "If `python` or `which python` still shows /usr/bin/… after `source .venv/bin/activate`, "
            "your shell is not using the venv—use `.venv/bin/python` explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)

    audio_path = args.audio.expanduser().resolve()
    if not audio_path.is_file():
        print(f"error: file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    if shutil.which("ffmpeg") is None:
        print(
            "error: ffmpeg not found on PATH. Install ffmpeg to decode M4A and other formats.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.pause_gap < 0 or args.flush_minutes < 0:
        print("error: --pause-gap and --flush-minutes must be >= 0.", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        output_path = (
            audio_path.with_suffix(".txt")
            if args.no_timestamp
            else _default_output_path(audio_path)
        )
    else:
        output_path = args.output.expanduser().resolve()

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        print(
            f"error: could not import faster-whisper ({exc}).\n"
            "Install into this exact interpreter: python -m pip install -r requirements.txt\n"
            f"(this process is: {sys.executable})",
            file=sys.stderr,
        )
        sys.exit(1)

    transcribe_kw: dict = {}
    if args.language is not None:
        transcribe_kw["language"] = args.language

    flush_interval_sec = (
        args.flush_minutes * 60.0 if args.flush_minutes and args.flush_minutes > 0 else None
    )

    try:
        model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
        )
        segments, _info = model.transcribe(str(audio_path), **transcribe_kw)

        chunks: list[str] = []
        prev_end: float | None = None
        flushed_up_to = 0
        last_flush = time.monotonic()

        if flush_interval_sec is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("", encoding="utf-8")

        for seg in segments:
            text = (seg.text or "").strip()
            gap = seg.start - prev_end if prev_end is not None else 0.0
            if text:
                if prev_end is not None:
                    if args.pause_breaks and gap >= args.pause_gap:
                        chunks.append("\n\n")
                    else:
                        chunks.append(" ")
                chunks.append(text)
            prev_end = seg.end

            if flush_interval_sec is not None:
                now = time.monotonic()
                if now - last_flush >= flush_interval_sec:
                    new_part = "".join(chunks[flushed_up_to:])
                    if new_part:
                        with open(output_path, "a", encoding="utf-8") as f:
                            f.write(new_part)
                            f.flush()
                    flushed_up_to = len(chunks)
                    last_flush = now

        full = "".join(chunks)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if flush_interval_sec is None:
                output_path.write_text(full.strip() + "\n", encoding="utf-8")
            else:
                tail = "".join(chunks[flushed_up_to:])
                with open(output_path, "a", encoding="utf-8") as f:
                    if tail:
                        f.write(tail)
                        f.flush()
                raw = output_path.read_text(encoding="utf-8")
                output_path.write_text(raw.strip() + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"error: could not write {output_path}: {exc}", file=sys.stderr)
            sys.exit(1)
    except Exception as exc:
        print(f"error: transcription failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
