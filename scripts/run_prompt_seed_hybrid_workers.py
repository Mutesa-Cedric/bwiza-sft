#!/usr/bin/env python3
"""Launch multiple detached hybrid prompt-seed workers."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch detached hybrid seed workers")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--target", type=int, default=30000)
    p.add_argument("--overshoot_factor", type=float, default=1.2)
    p.add_argument("--flash_batch_size", type=int, default=4)
    p.add_argument("--gemini_model", default="gemini-3-flash-preview")
    p.add_argument("--organizer_model", default="gpt-5.2")
    p.add_argument("--disable_organizer", action="store_true")
    p.add_argument("--task_type_targeting", choices=["none", "soft", "strict"], default="soft")
    p.add_argument("--env_file", default=".env")
    p.add_argument("--base_output_dir", default="outputs/sft/multi")
    p.add_argument(
        "--shared_output_prefix",
        default="outputs/sft/prompts.seed.hybrid",
        help="Canonical shared prefix for append-only JSONL outputs.",
    )
    p.add_argument(
        "--shared_dedup_db",
        default="outputs/sft/prompts.seed.shared.dedup.sqlite",
        help="Canonical shared dedup DB used by all workers.",
    )
    p.add_argument("--topics_file", default="")
    p.add_argument("--sleep_sec", type=float, default=0.0)
    p.add_argument("--sleep_jitter_sec", type=float, default=0.5)
    p.add_argument("--failure_cooldown_sec", type=float, default=4.0)
    p.add_argument("--failure_cooldown_cap_sec", type=float, default=20.0)
    p.add_argument("--max_retries", type=int, default=8)
    p.add_argument("--retry_backoff_sec", type=float, default=2.0)
    p.add_argument("--request_timeout_sec", type=float, default=30.0)
    p.add_argument("--launch_stagger_sec", type=float, default=0.0)
    p.add_argument("--print_every", type=int, default=10)
    return p.parse_args()


def _worker_target(total_target: int, workers: int, overshoot_factor: float) -> int:
    return max(1, math.ceil((float(total_target) * float(overshoot_factor)) / max(1, int(workers))))


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_root = (root / args.base_output_dir).resolve()
    run_id = datetime.now(timezone.utc).strftime("hybrid_workers_%Y%m%d_%H%M%S")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    worker_target = _worker_target(args.target, args.workers, args.overshoot_factor)
    manifest: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "workers": int(args.workers),
        "requested_target": int(args.target),
        "overshoot_factor": float(args.overshoot_factor),
        "worker_target": int(worker_target),
        "gemini_model": args.gemini_model,
        "organizer_model": args.organizer_model,
        "disable_organizer": bool(args.disable_organizer),
        "task_type_targeting": args.task_type_targeting,
        "shared_output_prefix": str((root / args.shared_output_prefix).resolve()),
        "shared_dedup_db": str((root / args.shared_dedup_db).resolve()),
        "max_retries": int(args.max_retries),
        "retry_backoff_sec": float(args.retry_backoff_sec),
        "request_timeout_sec": float(args.request_timeout_sec),
        "sleep_sec": float(args.sleep_sec),
        "sleep_jitter_sec": float(args.sleep_jitter_sec),
        "failure_cooldown_sec": float(args.failure_cooldown_sec),
        "failure_cooldown_cap_sec": float(args.failure_cooldown_cap_sec),
        "launch_stagger_sec": float(args.launch_stagger_sec),
        "workers_launched": [],
    }

    script_path = root / "scripts" / "build_prompt_seed_hybrid.py"
    for idx in range(int(args.workers)):
        worker_name = f"worker_{idx + 1:02d}"
        worker_dir = run_dir / worker_name
        worker_dir.mkdir(parents=True, exist_ok=True)

        prefix = worker_dir / "prompts.seed.hybrid"
        dedup_db = (root / args.shared_dedup_db).resolve()
        shared_output_prefix = (root / args.shared_output_prefix).resolve()
        log_path = worker_dir / "worker.log"

        cmd = [
            sys.executable,
            str(script_path),
            "--env_file",
            args.env_file,
            "--output_prefix",
            str(prefix),
            "--shared_output_prefix",
            str(shared_output_prefix),
            "--dedup_db",
            str(dedup_db),
            "--target",
            str(worker_target),
            "--topic_index_start",
            str(idx),
            "--flash_batch_size",
            str(args.flash_batch_size),
            "--gemini_model",
            args.gemini_model,
            "--organizer_model",
            args.organizer_model,
            "--max_retries",
            str(args.max_retries),
            "--retry_backoff_sec",
            str(args.retry_backoff_sec),
            "--request_timeout_sec",
            str(args.request_timeout_sec),
            "--task_type_targeting",
            args.task_type_targeting,
            "--sleep_sec",
            str(args.sleep_sec),
            "--sleep_jitter_sec",
            str(args.sleep_jitter_sec),
            "--failure_cooldown_sec",
            str(args.failure_cooldown_sec),
            "--failure_cooldown_cap_sec",
            str(args.failure_cooldown_cap_sec),
            "--print_every",
            str(args.print_every),
        ]
        if args.disable_organizer:
            cmd.append("--disable_organizer")
        if args.topics_file:
            cmd.extend(["--topics_file", args.topics_file])

        with log_path.open("ab") as logf:
            proc = subprocess.Popen(
                cmd,
                cwd=str(root),
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        manifest["workers_launched"].append(
            {
                "worker_name": worker_name,
                "pid": proc.pid,
                "topic_index_start": idx,
                "output_prefix": str(prefix),
                "shared_output_prefix": str(shared_output_prefix),
                "dedup_db": str(dedup_db),
                "log_path": str(log_path),
            }
        )
        if idx + 1 < int(args.workers) and float(args.launch_stagger_sec) > 0:
            time.sleep(float(args.launch_stagger_sec))

    manifest_path = run_dir / "launch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
