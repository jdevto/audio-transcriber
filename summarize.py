#!/usr/bin/env python3
"""Turn a transcript .txt into meeting-style notes via local Ollama (offline)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_SUPPORTED_PY = ((3, 10), (3, 11), (3, 12), (3, 13))

MAP_PROMPT = """The following is an excerpt from a meeting transcript (may start or end mid-conversation).
Extract the important content only: topics discussed, decisions, concerns, names, dates, and action-like items.
Use short bullets. Do not invent facts. If the excerpt is mostly filler or unclear, say "No substantive content in this excerpt."
Do not add a preamble — start directly with bullets or the single line above.

--- Transcript excerpt ---
{chunk}
--- End excerpt ---"""

REDUCE_PROMPT = """You are given partial bullet notes from the same meeting (in order). Merge them into concise meeting minutes.

Use this Markdown structure:
## Overview
(2–4 sentences)

## Topics discussed
(Bullet list grouped loosely by theme if helpful)

## Decisions / agreements
(Bullets, or "None noted.")

## Action items
(Bullets with owner/name in parentheses when the transcript implies one, else omit)

## Open questions / follow-ups
(Bullets, or "None noted.")

Remove duplicates from overlapping excerpts. Do not invent attendees or actions not supported by the notes.

--- Partial notes ---
{partial}
--- End notes ---"""


def _python_ok() -> bool:
    return sys.version_info[:2] in _SUPPORTED_PY


def _default_output_path(transcript_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return transcript_path.with_name(f"{transcript_path.stem}_minutes_{ts}.md")


def _chunk_text(text: str, max_chars: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - overlap)
    return chunks


def _ollama_chat(
    base_url: str,
    model: str,
    user_content: str,
    timeout_sec: float,
) -> str:
    base = base_url.rstrip("/")
    chat_url = base + "/api/chat"
    chat_body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "stream": False,
        }
    ).encode("utf-8")
    chat_req = Request(
        chat_url,
        data=chat_body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(chat_req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        msg = data.get("message") or {}
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Unexpected Ollama /api/chat response: {raw[:500]}")
        return content.strip()
    except HTTPError as exc:
        err_body = ""
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        if "model" in err_body.lower() and "not found" in err_body.lower():
            raise RuntimeError(
                f"Ollama model {model!r} is not available locally.\n"
                f"Pull it first: ollama pull {model}\n"
                "Check installed models: curl -s http://127.0.0.1:11434/api/tags"
            ) from exc
        # Some Ollama builds expose /api/generate but not /api/chat.
        if exc.code != 404:
            raise RuntimeError(
                f"Ollama request failed at {chat_url!r} (HTTP {exc.code}). {err_body[:200]}"
            ) from exc
        gen_url = base + "/api/generate"
        gen_body = json.dumps(
            {
                "model": model,
                "prompt": user_content,
                "stream": False,
            }
        ).encode("utf-8")
        gen_req = Request(
            gen_url,
            data=gen_body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(gen_req, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
            data = json.loads(raw)
            content = data.get("response")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError(
                    f"Unexpected Ollama /api/generate response: {raw[:500]}"
                )
            return content.strip()
        except HTTPError as gen_exc:
            gen_body = ""
            try:
                gen_body = gen_exc.read().decode("utf-8", errors="replace")
            except Exception:
                gen_body = ""
            if "model" in gen_body.lower() and "not found" in gen_body.lower():
                raise RuntimeError(
                    f"Ollama model {model!r} is not available locally.\n"
                    f"Pull it first: ollama pull {model}\n"
                    "Check installed models: curl -s http://127.0.0.1:11434/api/tags"
                ) from gen_exc
            raise RuntimeError(
                f"Ollama reachable but /api/chat returned 404 and fallback "
                f"/api/generate failed (HTTP {gen_exc.code}). {gen_body[:200]}"
            ) from gen_exc
        except URLError as gen_exc:
            raise RuntimeError(
                f"Ollama reachable but /api/chat returned 404 and fallback "
                f"/api/generate also failed ({gen_exc})."
            ) from gen_exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {chat_url!r} ({exc}).\n"
            "1) Install: https://ollama.com/download/linux (or your OS installer)\n"
            "2) Start the server — one of: run `ollama serve` in another terminal; "
            "or `systemctl --user start ollama` if your install registered a user service\n"
            "3) Check: curl -s http://127.0.0.1:11434/api/tags\n"
            "4) Pull a model once: ollama pull llama3.2  (or pass --model to match what you pulled)\n"
            "If Ollama listens elsewhere, use --base-url (see env OLLAMA_HOST in Ollama docs)."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize a transcript using local Ollama (no cloud; requires ollama serve)."
    )
    parser.add_argument(
        "transcript",
        type=Path,
        help="Path to the transcript .txt file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output Markdown file (default: <stem>_minutes_YYYYMMDD-HHMMSS.md beside input).",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:11434",
        metavar="URL",
        help="Ollama API base URL. Default: http://127.0.0.1:11434",
    )
    parser.add_argument(
        "--model",
        default="llama3.2",
        metavar="NAME",
        help="Ollama model name (must be pulled locally). Default: llama3.2",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=14000,
        metavar="N",
        help="Max characters per transcript chunk for the map step. Default: 14000",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=400,
        metavar="N",
        help="Overlap between chunks to reduce missed context at boundaries. Default: 400",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        metavar="SEC",
        help="HTTP timeout per Ollama request. Default: 600",
    )
    args = parser.parse_args()

    if not _python_ok():
        print(
            f"error: Python {sys.version_info.major}.{sys.version_info.minor} is not supported "
            "(use 3.10–3.13, same as transcribe.py).",
            file=sys.stderr,
        )
        sys.exit(1)

    path = args.transcript.expanduser().resolve()
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        print("error: transcript is empty.", file=sys.stderr)
        sys.exit(1)

    if args.output is None:
        out_path = _default_output_path(path)
    else:
        out_path = args.output.expanduser().resolve()

    if args.chunk_chars < 2000:
        print("error: --chunk-chars should be at least 2000 for useful context.", file=sys.stderr)
        sys.exit(1)
    if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_chars:
        print("error: --chunk-overlap must be >= 0 and < --chunk-chars.", file=sys.stderr)
        sys.exit(1)

    chunks = _chunk_text(text, args.chunk_chars, args.chunk_overlap)
    print(f"Transcript: {path} ({len(text)} chars, {len(chunks)} chunk(s))", file=sys.stderr)
    print(f"Ollama: {args.base_url} model={args.model}", file=sys.stderr)

    partial_notes: list[str] = []
    try:
        for i, chunk in enumerate(chunks, start=1):
            print(f"Map chunk {i}/{len(chunks)} …", file=sys.stderr)
            prompt = MAP_PROMPT.format(chunk=chunk)
            partial_notes.append(_ollama_chat(args.base_url, args.model, prompt, args.timeout))

        combined = "\n\n".join(
            f"### Part {i + 1}\n{n}" for i, n in enumerate(partial_notes)
        )
        print("Reduce → meeting minutes …", file=sys.stderr)
        final_prompt = REDUCE_PROMPT.format(partial=combined)
        minutes = _ollama_chat(args.base_url, args.model, final_prompt, args.timeout)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(minutes.strip() + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"error: could not write {out_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
