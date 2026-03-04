#!/usr/bin/env python3
"""Clean, normalize, and deduplicate generated SFT jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sft_cleaning import CleanConfig, clean_decision, dedup_key, to_clean_record
from src.data.sft_records import extract_prompt, extract_response, iter_jsonl, append_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean generated SFT records")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--rejects_jsonl", default="")
    p.add_argument("--report_json", default="")
    p.add_argument("--min_prompt_chars", type=int, default=6)
    p.add_argument("--min_response_chars", type=int, default=24)
    p.add_argument("--max_response_chars", type=int, default=6000)
    p.add_argument("--max_consecutive_repeat_words", type=int, default=12)
    p.add_argument("--min_unique_word_ratio", type=float, default=0.12)
    p.add_argument("--max_english_ratio_for_rw", type=float, default=0.45)
    p.add_argument("--allow_role_prefix_leakage", action="store_true", default=False)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    in_path = Path(args.input_jsonl)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    out_path = Path(args.output_jsonl)
    rej_path = Path(args.rejects_jsonl) if args.rejects_jsonl else out_path.with_suffix(".rejects.jsonl")
    report_path = Path(args.report_json) if args.report_json else out_path.with_suffix(".report.json")

    cfg = CleanConfig(
        min_prompt_chars=int(args.min_prompt_chars),
        min_response_chars=int(args.min_response_chars),
        max_response_chars=int(args.max_response_chars),
        max_consecutive_repeat_words=int(args.max_consecutive_repeat_words),
        min_unique_word_ratio=float(args.min_unique_word_ratio),
        max_english_ratio_for_rw=float(args.max_english_ratio_for_rw),
        reject_role_prefix_leakage=not bool(args.allow_role_prefix_leakage),
    )

    seen: set[str] = set()
    stats = {
        "processed": 0,
        "kept": 0,
        "rejected": 0,
        "reasons": {},
    }

    if out_path.exists():
        out_path.unlink()
    if rej_path.exists():
        rej_path.unlink()

    for line_no, raw in iter_jsonl(in_path):
        stats["processed"] += 1

        prompt = extract_prompt(raw)
        response = extract_response(raw)
        d = clean_decision(prompt, response, cfg, raw=raw)

        if not d.keep:
            stats["rejected"] += 1
            stats["reasons"][d.reason] = stats["reasons"].get(d.reason, 0) + 1
            append_jsonl(rej_path, {"line_no": line_no, "reason": d.reason, "record": raw})
            continue

        key = dedup_key(prompt, response)
        if key in seen:
            reason = "duplicate_prompt_response"
            stats["rejected"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
            append_jsonl(rej_path, {"line_no": line_no, "reason": reason, "record": raw})
            continue

        seen.add(key)
        cleaned = to_clean_record(raw, prompt=prompt, response=response)
        append_jsonl(out_path, cleaned)
        stats["kept"] += 1

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
