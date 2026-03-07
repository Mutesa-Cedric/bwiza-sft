#!/usr/bin/env python3
"""Launch multiple detached OpenAI chat-seed workers."""

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
    p = argparse.ArgumentParser(description="Launch detached OpenAI chat seed workers")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--target_dialogues", type=int, default=2000)
    p.add_argument("--overshoot_factor", type=float, default=1.0)
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--env_file", default=".env")
    p.add_argument("--base_output_dir", default="outputs/sft/multi")
    p.add_argument(
        "--shared_output_prefix",
        default="outputs/sft/chat.seed.openai",
        help="Canonical shared prefix for append-only JSONL outputs.",
    )
    p.add_argument(
        "--shared_dedup_db",
        default="outputs/sft/chat.seed.shared.dedup.sqlite",
        help="Canonical shared dedup DB used by all workers.",
    )
    p.add_argument("--topics_file", default="")
    p.add_argument("--min_turns", type=int, default=2)
    p.add_argument("--max_turns", type=int, default=4)
    p.add_argument("--history_window_messages", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_output_tokens", type=int, default=0)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--retry_backoff_sec", type=float, default=1.5)
    p.add_argument("--request_timeout_sec", type=float, default=120.0)
    p.add_argument("--sleep_sec", type=float, default=0.0)
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument("--launch_stagger_sec", type=float, default=1.0)
    return p.parse_args()


def _worker_target(total_target: int, workers: int, overshoot_factor: float) -> int:
    return max(1, math.ceil((float(total_target) * float(overshoot_factor)) / max(1, int(workers))))


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_root = (root / args.base_output_dir).resolve()
    run_id = datetime.now(timezone.utc).strftime("chat_workers_%Y%m%d_%H%M%S")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    worker_target = _worker_target(args.target_dialogues, args.workers, args.overshoot_factor)
    manifest: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "workers": int(args.workers),
        "requested_target_dialogues": int(args.target_dialogues),
        "overshoot_factor": float(args.overshoot_factor),
        "worker_target_dialogues": int(worker_target),
        "model": args.model,
        "shared_output_prefix": str((root / args.shared_output_prefix).resolve()),
        "shared_dedup_db": str((root / args.shared_dedup_db).resolve()),
        "workers_launched": [],
    }

    script_path = root / "scripts" / "build_chat_seed_openai.py"
    for idx in range(int(args.workers)):
        worker_name = f"worker_{idx + 1:02d}"
        worker_dir = run_dir / worker_name
        worker_dir.mkdir(parents=True, exist_ok=True)

        prefix = worker_dir / "chat.seed.openai"
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
            "--target_dialogues",
            str(worker_target),
            "--topic_index_start",
            str(idx),
            "--model",
            args.model,
            "--min_turns",
            str(args.min_turns),
            "--max_turns",
            str(args.max_turns),
            "--history_window_messages",
            str(args.history_window_messages),
            "--temperature",
            str(args.temperature),
            "--max_retries",
            str(args.max_retries),
            "--retry_backoff_sec",
            str(args.retry_backoff_sec),
            "--request_timeout_sec",
            str(args.request_timeout_sec),
            "--sleep_sec",
            str(args.sleep_sec),
            "--print_every",
            str(args.print_every),
        ]
        if int(args.max_output_tokens) > 0:
            cmd.extend(["--max_output_tokens", str(args.max_output_tokens)])
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
