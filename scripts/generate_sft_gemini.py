#!/usr/bin/env python3
"""Generate SFT responses via Gemini from prompt JSONL."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.gemini_client import GeminiConfig, GeminiError, generate_text
from src.data.sft_records import append_jsonl, extract_prompt, iter_jsonl, record_id


DEFAULT_SYSTEM_PROMPT = (
    "You are generating high-quality assistant training data. "
    "Write one direct, helpful assistant response only. "
    "No JSON, no markdown fences, no role labels. "
    "Respect the language used by the user prompt. "
    "If the prompt is code-switched, respond naturally in that mix."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SFT data via Gemini")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--errors_jsonl", default="")
    p.add_argument("--state_path", default="")
    p.add_argument("--model", default="gemini-3.1-pro-preview")
    p.add_argument("--api_key_env", default="GEMINI_API_KEY")
    p.add_argument("--system_prompt_file", default="")
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_output_tokens", type=int, default=1024)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--retry_backoff_sec", type=float, default=1.5)
    p.add_argument("--sleep_sec", type=float, default=0.1)
    p.add_argument("--max_items", type=int, default=0)
    p.add_argument("--print_every", type=int, default=20)
    return p.parse_args()


def _load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "last_line": 0,
        "processed": 0,
        "succeeded": 0,
        "failed": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": "",
    }


def _save_state(path: Path, state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def _system_prompt(args: argparse.Namespace) -> str:
    if args.system_prompt_file:
        return Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
    return DEFAULT_SYSTEM_PROMPT


def main() -> int:
    args = parse_args()

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key in env var: {args.api_key_env}")

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input jsonl: {input_path}")

    output_path = Path(args.output_jsonl)
    errors_path = Path(args.errors_jsonl) if args.errors_jsonl else output_path.with_suffix(".errors.jsonl")
    state_path = Path(args.state_path) if args.state_path else output_path.with_suffix(".state.json")

    cfg = GeminiConfig(
        model=args.model,
        api_key=api_key,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_output_tokens=int(args.max_output_tokens),
        max_retries=int(args.max_retries),
        retry_backoff_sec=float(args.retry_backoff_sec),
    )

    sys_prompt = _system_prompt(args)
    state = _load_state(state_path)
    start_line = int(state.get("last_line", 0)) + 1

    generated = 0
    for line_no, obj in iter_jsonl(input_path):
        if line_no < start_line:
            continue
        if args.max_items > 0 and generated >= args.max_items:
            break

        prompt = extract_prompt(obj)
        rid = record_id(obj, prompt, line_no)

        state["processed"] = int(state.get("processed", 0)) + 1
        state["last_line"] = line_no

        if not prompt:
            state["failed"] = int(state.get("failed", 0)) + 1
            append_jsonl(
                errors_path,
                {
                    "id": rid,
                    "line_no": line_no,
                    "reason": "empty_prompt",
                    "source": "extract_prompt",
                },
            )
            _save_state(state_path, state)
            continue

        user_prompt = (
            "Generate one assistant response suitable for supervised fine-tuning.\n"
            "Return only the assistant response text.\n\n"
            f"User prompt:\n{prompt}"
        )

        try:
            response = generate_text(cfg, user_prompt=user_prompt, system_prompt=sys_prompt)
            append_jsonl(
                output_path,
                {
                    "id": rid,
                    "prompt": prompt,
                    "response": response,
                    "teacher_model": cfg.model,
                    "source": "gemini_distill",
                    "line_no": line_no,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            state["succeeded"] = int(state.get("succeeded", 0)) + 1
            generated += 1
        except GeminiError as e:
            state["failed"] = int(state.get("failed", 0)) + 1
            append_jsonl(
                errors_path,
                {
                    "id": rid,
                    "line_no": line_no,
                    "reason": "gemini_error",
                    "error": str(e),
                    "prompt": prompt,
                },
            )

        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)

        _save_state(state_path, state)
        if int(state.get("processed", 0)) % max(1, args.print_every) == 0:
            print(
                f"processed={state['processed']} succeeded={state['succeeded']} failed={state['failed']} last_line={state['last_line']}"
            )

    _save_state(state_path, state)
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
