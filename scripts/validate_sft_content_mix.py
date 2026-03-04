#!/usr/bin/env python3
"""Validate SFT content-type and language mix before full training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sft_records import iter_jsonl, normalize_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate SFT content mix")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--plan_json", default="configs/sft_content_mix_v1.json")
    p.add_argument("--output", default="outputs/reports/sft_content_mix_report.json")
    return p.parse_args()


def _load_plan(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing mix plan: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("Mix plan must be a JSON object")
    return obj


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input jsonl: {input_path}")
    plan = _load_plan(Path(args.plan_json))

    min_rows = int(plan.get("min_rows", 0))
    max_single_topic_ratio = float(plan.get("max_single_topic_ratio", 1.0))
    min_content_ratio: dict[str, float] = {
        str(k): float(v) for k, v in dict(plan.get("content_type_min_ratio", {})).items()
    }
    min_lang_ratio: dict[str, float] = {
        str(k): float(v) for k, v in dict(plan.get("lang_mode_min_ratio", {})).items()
    }

    total = 0
    content_counts: dict[str, int] = {}
    lang_counts: dict[str, int] = {}
    topic_counts: dict[str, int] = {}

    for _, rec in iter_jsonl(input_path):
        total += 1
        content = normalize_text(rec.get("content_type", "")) or normalize_text(rec.get("task_type", "")) or "unknown"
        lang = normalize_text(rec.get("lang_mode", "")) or "unknown"
        topic = normalize_text(rec.get("topic", "")) or "unknown"
        content_counts[content] = content_counts.get(content, 0) + 1
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    failures: list[str] = []
    if total < min_rows:
        failures.append(f"too_few_rows:{total}<{min_rows}")

    content_ratio = {k: (v / total if total else 0.0) for k, v in content_counts.items()}
    lang_ratio = {k: (v / total if total else 0.0) for k, v in lang_counts.items()}

    for key, min_r in min_content_ratio.items():
        got = content_ratio.get(key, 0.0)
        if got < min_r:
            failures.append(f"content_ratio_low:{key}:{got:.4f}<{min_r:.4f}")

    for key, min_r in min_lang_ratio.items():
        got = lang_ratio.get(key, 0.0)
        if got < min_r:
            failures.append(f"lang_ratio_low:{key}:{got:.4f}<{min_r:.4f}")

    if total > 0:
        top_topic = max(topic_counts.items(), key=lambda kv: kv[1])
        top_topic_ratio = top_topic[1] / total
    else:
        top_topic = ("unknown", 0)
        top_topic_ratio = 0.0
    if top_topic_ratio > max_single_topic_ratio:
        failures.append(
            f"topic_concentration_high:{top_topic[0]}:{top_topic_ratio:.4f}>{max_single_topic_ratio:.4f}"
        )

    report = {
        "ok": not failures,
        "failures": failures,
        "input": str(input_path),
        "plan": str(args.plan_json),
        "total_rows": total,
        "content_counts": content_counts,
        "content_ratio": content_ratio,
        "lang_counts": lang_counts,
        "lang_ratio": lang_ratio,
        "top_topic": {"topic": top_topic[0], "rows": top_topic[1], "ratio": top_topic_ratio},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
