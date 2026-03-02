"""Deterministic fingerprints for SFT jsonl inputs."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json


def file_descriptor(path: str | Path) -> dict[str, str | int]:
    p = Path(path)
    st = p.stat()
    return {
        "path": str(p.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def dataset_fingerprint(train_jsonl: str | Path, val_jsonl: str | Path, test_jsonl: str | Path = "") -> str:
    payload: dict[str, object] = {
        "train": file_descriptor(train_jsonl),
        "val": file_descriptor(val_jsonl),
    }
    if str(test_jsonl):
        payload["test"] = file_descriptor(test_jsonl)

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
