"""Record parsing and deterministic split helpers for SFT datasets."""

from __future__ import annotations

from hashlib import sha1
from pathlib import Path
from typing import Any, Iterator
import contextlib
import json
import os
import re

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


SMART_PUNCT_MAP = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "«": '"',
        "»": '"',
        "‹": "'",
        "›": "'",
        "–": "-",
        "—": "-",
        "‑": "-",
        "‒": "-",
        "−": "-",
        "…": "...",
        "∕": "/",
        "⁄": "/",
        "：": ":",
        "，": ",",
        "；": ";",
        "（": "(",
        "）": ")",
        "？": "?",
        "！": "!",
        "\u00A0": " ",
        "\u2007": " ",
        "\u202F": " ",
    }
)
_ZERO_WIDTH_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    return " ".join(value.strip().split())


def normalize_ascii_text(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    text = _ZERO_WIDTH_RE.sub("", text)
    return normalize_text(text.translate(SMART_PUNCT_MAP))


def extract_prompt(record: dict[str, Any]) -> str:
    if "prompt" in record:
        return normalize_ascii_text(record.get("prompt"))
    if "question" in record:
        return normalize_ascii_text(record.get("question"))
    if "instruction" in record and "output" in record:
        ins = normalize_ascii_text(record.get("instruction"))
        inp = normalize_ascii_text(record.get("input"))
        return ins if not inp else f"{ins}\n{inp}"
    if "messages" in record and isinstance(record.get("messages"), list):
        for m in record["messages"]:
            if isinstance(m, dict) and normalize_text(m.get("role")).lower() == "user":
                prompt = normalize_ascii_text(m.get("content"))
                if prompt:
                    return prompt
    return ""


def extract_response(record: dict[str, Any]) -> str:
    if "response" in record:
        return normalize_ascii_text(record.get("response"))
    if "answer" in record:
        return normalize_ascii_text(record.get("answer"))
    if "output" in record:
        return normalize_ascii_text(record.get("output"))
    if "messages" in record and isinstance(record.get("messages"), list):
        for m in record["messages"]:
            if isinstance(m, dict) and normalize_text(m.get("role")).lower() == "assistant":
                resp = normalize_ascii_text(m.get("content"))
                if resp:
                    return resp
    return ""


def record_id(record: dict[str, Any], fallback_prompt: str, line_no: int) -> str:
    explicit = normalize_text(record.get("id", ""))
    if explicit:
        return explicit
    h = sha1(f"{line_no}:{fallback_prompt}".encode("utf-8")).hexdigest()[:16]
    return f"auto_{h}"


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield line_no, obj


def append_jsonl(path: str | Path, obj: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(obj, ensure_ascii=True) + "\n")
        f.flush()
        with contextlib.suppress(OSError):
            os.fsync(f.fileno())
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def split_bucket(key: str, train_ratio: float, val_ratio: float) -> str:
    digest = sha1(key.encode("utf-8")).hexdigest()[:8]
    x = int(digest, 16) / 0xFFFFFFFF
    if x < train_ratio:
        return "train"
    if x < (train_ratio + val_ratio):
        return "val"
    return "test"
