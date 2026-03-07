#!/usr/bin/env python3
"""Launch multiple detached OpenAI SFT answer-generation workers."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch detached OpenAI SFT generation workers")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--env_file", default=".env")
    p.add_argument("--base_output_dir", default="outputs/sft/multi")
    p.add_argument("--shared_output_jsonl", default="outputs/sft/answers.openai.raw.jsonl")
    p.add_argument("--shared_errors_jsonl", default="outputs/sft/answers.openai.errors.jsonl")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_output_tokens", type=int, default=900)
    p.add_argument("--max_retries", type=int, default=4)
    p.add_argument("--retry_backoff_sec", type=float, default=1.0)
    p.add_argument("--request_timeout_sec", type=float, default=120.0)
    p.add_argument("--sleep_sec", type=float, default=0.0)
    p.add_argument("--max_items_per_worker", type=int, default=0)
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument("--launch_stagger_sec", type=float, default=1.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_root = (root / args.base_output_dir).resolve()
    run_id = datetime.now(timezone.utc).strftime("sft_openai_workers_%Y%m%d_%H%M%S")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    shared_output = (root / args.shared_output_jsonl).resolve()
    shared_errors = (root / args.shared_errors_jsonl).resolve()

    manifest: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "workers": int(args.workers),
        "input_jsonl": str((root / args.input_jsonl).resolve()),
        "model": args.model,
        "shared_output_jsonl": str(shared_output),
        "shared_errors_jsonl": str(shared_errors),
        "workers_launched": [],
    }

    script_path = root / "scripts" / "generate_sft_openai.py"
    for idx in range(int(args.workers)):
        worker_name = f"worker_{idx + 1:02d}"
        worker_dir = run_dir / worker_name
        worker_dir.mkdir(parents=True, exist_ok=True)
        local_output = worker_dir / "answers.openai.raw.jsonl"
        local_state = worker_dir / "answers.openai.state.json"
        log_path = worker_dir / "worker.log"

        cmd = [
            sys.executable,
            str(script_path),
            "--env_file",
            args.env_file,
            "--input_jsonl",
            args.input_jsonl,
            "--output_jsonl",
            str(local_output),
            "--state_path",
            str(local_state),
            "--shared_output_jsonl",
            str(shared_output),
            "--shared_errors_jsonl",
            str(shared_errors),
            "--worker_index",
            str(idx),
            "--worker_stride",
            str(args.workers),
            "--model",
            args.model,
            "--temperature",
            str(args.temperature),
            "--max_output_tokens",
            str(args.max_output_tokens),
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
        if int(args.max_items_per_worker) > 0:
            cmd.extend(["--max_items", str(args.max_items_per_worker)])

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
                "worker_index": idx,
                "worker_stride": int(args.workers),
                "state_path": str(local_state),
                "local_output_jsonl": str(local_output),
                "shared_output_jsonl": str(shared_output),
                "shared_errors_jsonl": str(shared_errors),
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
