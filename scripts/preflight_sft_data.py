#!/usr/bin/env python3
"""Preflight validation for SFT JSONL splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.fingerprint import dataset_fingerprint
from src.data.sft_loader import summarize_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate SFT JSONL inputs")
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--val_jsonl", required=True)
    p.add_argument("--test_jsonl", required=True)
    p.add_argument("--output", default="outputs/reports/preflight_sft_data.json")
    p.add_argument("--min_valid_rows", type=int, default=100)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    failures: list[str] = []
    summaries = {}
    for name, path in {
        "train": args.train_jsonl,
        "val": args.val_jsonl,
        "test": args.test_jsonl,
    }.items():
        p = Path(path)
        if not p.exists():
            failures.append(f"missing:{name}:{p}")
            continue
        s = summarize_jsonl(p)
        summaries[name] = s.to_dict()
        if s.valid_rows < args.min_valid_rows:
            failures.append(f"too_few_valid_rows:{name}:{s.valid_rows}")

    fp = ""
    if not failures:
        fp = dataset_fingerprint(args.train_jsonl, args.val_jsonl, args.test_jsonl)

    out = {
        "ok": not failures,
        "failures": failures,
        "dataset_fingerprint": fp,
        "summaries": summaries,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
