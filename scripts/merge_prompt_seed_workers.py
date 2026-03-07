#!/usr/bin/env python3
"""Merge detached worker outputs into one deduped prompt-seed file."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge worker hybrid prompt outputs")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--output_jsonl", default="")
    p.add_argument("--summary_json", default="")
    return p.parse_args()


def _norm_prompt(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _score(rec: dict[str, object]) -> tuple[int, str]:
    richness = 0
    for key in ("content_type", "task_type", "lang_mode", "topic", "source", "teacher_model"):
        value = rec.get(key, "")
        if isinstance(value, str) and value.strip():
            richness += 1
    return richness, str(rec.get("created_at", ""))


def _iter_candidate_files(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(run_dir.glob("worker_*/prompts.seed.hybrid.final.jsonl")):
        if path.is_file():
            files.append(path)
    return files


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run dir: {run_dir}")

    output_jsonl = (
        Path(args.output_jsonl).resolve()
        if args.output_jsonl
        else (run_dir / "merged.final.jsonl").resolve()
    )
    summary_json = (
        Path(args.summary_json).resolve()
        if args.summary_json
        else (run_dir / "merged.summary.json").resolve()
    )

    best: dict[str, dict[str, object]] = {}
    kept_from: dict[str, int] = {}
    origin_by_prompt: dict[str, str] = {}
    input_lines = 0
    input_files = _iter_candidate_files(run_dir)

    for path in input_files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            input_lines += 1
            rec = json.loads(line)
            prompt = str(rec.get("prompt", ""))
            key = _norm_prompt(prompt)
            if not key:
                continue
            prev = best.get(key)
            if prev is None:
                best[key] = rec
                origin_by_prompt[key] = path.name
                kept_from[path.name] = kept_from.get(path.name, 0) + 1
            elif _score(rec) > _score(prev):
                prev_name = origin_by_prompt.get(key, "")
                if prev_name:
                    kept_from[prev_name] = max(0, kept_from.get(prev_name, 0) - 1)
                best[key] = rec
                origin_by_prompt[key] = path.name
                kept_from[path.name] = kept_from.get(path.name, 0) + 1

    records = sorted(
        best.values(),
        key=lambda r: (str(r.get("created_at", "")), _norm_prompt(str(r.get("prompt", "")))),
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "input_files": [str(p) for p in input_files],
        "input_lines": input_lines,
        "unique_prompts": len(records),
        "output_jsonl": str(output_jsonl),
        "source_kept_counts": kept_from,
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
