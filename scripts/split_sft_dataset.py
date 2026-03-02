#!/usr/bin/env python3
"""Deterministically split cleaned SFT jsonl into train/val/test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sft_records import append_jsonl, extract_prompt, iter_jsonl, split_bucket, normalize_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split SFT dataset")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--val_jsonl", required=True)
    p.add_argument("--test_jsonl", required=True)
    p.add_argument("--train_ratio", type=float, default=0.9)
    p.add_argument("--val_ratio", type=float, default=0.05)
    p.add_argument("--report_json", default="")
    return p.parse_args()


def _key_for_split(record: dict) -> str:
    rid = normalize_text(record.get("id", ""))
    if rid:
        return rid
    prompt = extract_prompt(record)
    return prompt or json.dumps(record, sort_keys=True, ensure_ascii=True)


def main() -> int:
    args = parse_args()

    in_path = Path(args.input_jsonl)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    train_ratio = float(args.train_ratio)
    val_ratio = float(args.val_ratio)
    if train_ratio <= 0 or val_ratio < 0 or (train_ratio + val_ratio) >= 1:
        raise ValueError("Invalid split ratios; require 0<train, 0<=val, train+val<1")

    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    test_path = Path(args.test_jsonl)
    report_path = Path(args.report_json) if args.report_json else test_path.with_suffix(".split_report.json")

    for p in (train_path, val_path, test_path):
        if p.exists():
            p.unlink()

    stats = {"total": 0, "train": 0, "val": 0, "test": 0}

    for _, rec in iter_jsonl(in_path):
        stats["total"] += 1
        key = _key_for_split(rec)
        bucket = split_bucket(key, train_ratio=train_ratio, val_ratio=val_ratio)
        if bucket == "train":
            append_jsonl(train_path, rec)
        elif bucket == "val":
            append_jsonl(val_path, rec)
        else:
            append_jsonl(test_path, rec)
        stats[bucket] += 1

    report = {
        "input": str(in_path),
        "train_jsonl": str(train_path),
        "val_jsonl": str(val_path),
        "test_jsonl": str(test_path),
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": float(1.0 - train_ratio - val_ratio),
        },
        "counts": stats,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
