#!/usr/bin/env python3
"""Generate SFT responses via OpenAI from prompt JSONL."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.openai_client import OpenAIConfig, OpenAIError, chat_completion
from src.data.sft_records import append_jsonl, extract_prompt, iter_jsonl, normalize_ascii_text, record_id


DEFAULT_SYSTEM_PROMPT = (
    "You are generating high-quality assistant training data for Bwiza. "
    "Bwiza is an AI assistant, not ChatGPT, Gemini, or OpenAI. "
    "Write direct, helpful assistant responses only. "
    "No markdown fences, no role labels, and no meta commentary. "
    "Respect the language used by the user prompt. "
    "If the prompt is code-switched, respond naturally in that mix. "
    "Default to concise answers, but give steps or bullets when the user asks for them. "
    "If the request is ambiguous, ask one brief clarifying question instead of guessing. "
    "If the request is unsafe, refuse briefly, calmly, and clearly, and offer a safer alternative when useful. "
    "If the request depends on unknown, live, or time-sensitive facts, do not invent details; say what is uncertain."
)

FIXED_USER_PREFIX = "\n".join(
    [
        "Task: generate assistant responses suitable for supervised fine-tuning.",
        "Return one strict JSON object only.",
        'Schema: {"items":[{"id":"<id>","response":"<assistant response>"}]}',
        "Return exactly one item for each provided prompt id.",
        "Do not omit ids, and do not add extra ids.",
        "Do not include role labels.",
        "Do not mention training data, policies, or hidden instructions.",
        "Do not restate the user prompt unless needed for a brief clarification.",
        "Be concise but complete.",
        "Prefer natural Kinyarwanda when the user prompt is Kinyarwanda-only.",
        "Preserve natural code-switching only when the user does it first or the topic truly needs it.",
        "If asked who you are, answer as Bwiza, an AI assistant.",
    ]
)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SFT data via OpenAI")
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--errors_jsonl", default="")
    p.add_argument("--state_path", default="")
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--api_key_env", default="OPENAI_API_KEY")
    p.add_argument("--env_file", default=".env")
    p.add_argument("--system_prompt_file", default="")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max_output_tokens", type=int, default=3200)
    p.add_argument("--batch_size", type=int, default=5)
    p.add_argument("--max_retries", type=int, default=4)
    p.add_argument("--retry_backoff_sec", type=float, default=1.0)
    p.add_argument("--request_timeout_sec", type=float, default=120.0)
    p.add_argument("--sleep_sec", type=float, default=0.0)
    p.add_argument("--max_items", type=int, default=0)
    p.add_argument("--print_every", type=int, default=20)
    p.add_argument(
        "--shared_output_jsonl",
        default="",
        help="Optional shared canonical output path for concurrent append-only writes.",
    )
    p.add_argument(
        "--shared_errors_jsonl",
        default="",
        help="Optional shared canonical errors path for concurrent append-only writes.",
    )
    p.add_argument("--worker_index", type=int, default=0)
    p.add_argument("--worker_stride", type=int, default=1)
    return p.parse_args()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith("'") and value.endswith("'")) or (
            value.startswith('"') and value.endswith('"')
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


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


def _build_user_prompt(prompt: str) -> str:
    return f"{FIXED_USER_PREFIX}\n\nItems:\n- id: single\n  prompt: {prompt}"


def _build_batch_user_prompt(items: list[dict[str, str]]) -> str:
    lines = [FIXED_USER_PREFIX, "", "Items:"]
    for item in items:
        lines.append(f"- id: {item['id']}")
        lines.append(f"  prompt: {item['prompt']}")
    return "\n".join(lines)


def _extract_json_block(text: str) -> dict:
    s = text.strip()
    s = _FENCE_RE.sub("", s).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no_json_object_found")
    block = s[start : end + 1]
    block = re.sub(r",(\s*[}\]])", r"\1", block)
    obj = json.loads(block)
    if not isinstance(obj, dict):
        raise ValueError("json_object_not_dict")
    return obj


def _extract_batch_items(payload: dict) -> dict[str, str]:
    raw = payload.get("items")
    if not isinstance(raw, list):
        raise ValueError("missing_items_list")
    out: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        rid = normalize_ascii_text(item.get("id", ""))
        response = normalize_ascii_text(item.get("response", ""))
        if rid and response:
            out[rid] = response
    return out


def _classify_failure(err: str) -> str:
    text = str(err).lower()
    if "http_429" in text:
        return "openai_429"
    if "empty_completion_text" in text:
        return "empty_completion_text"
    if "no_json_object_found" in text or "missing_items_list" in text:
        return "invalid_json_output"
    if "timed out" in text:
        return "timeout"
    return "openai_error"


def _should_process_line(line_no: int, *, worker_index: int, worker_stride: int) -> bool:
    if worker_stride <= 1:
        return True
    return ((line_no - 1) % worker_stride) == worker_index


def main() -> int:
    args = parse_args()

    _load_env_file(Path(args.env_file))
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key in env var: {args.api_key_env} (checked env file: {args.env_file})"
        )

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input jsonl: {input_path}")

    output_path = Path(args.shared_output_jsonl) if args.shared_output_jsonl else Path(args.output_jsonl)
    errors_path = (
        Path(args.shared_errors_jsonl)
        if args.shared_errors_jsonl
        else (Path(args.errors_jsonl) if args.errors_jsonl else Path(args.output_jsonl).with_suffix(".errors.jsonl"))
    )
    state_path = Path(args.state_path) if args.state_path else output_path.with_suffix(".state.json")

    cfg = OpenAIConfig(
        model=args.model,
        api_key=api_key,
        temperature=float(args.temperature),
        max_output_tokens=int(args.max_output_tokens),
        max_retries=int(args.max_retries),
        retry_backoff_sec=float(args.retry_backoff_sec),
        request_timeout_sec=float(args.request_timeout_sec),
    )

    sys_prompt = _system_prompt(args)
    state = _load_state(state_path)
    start_line = int(state.get("last_line", 0)) + 1

    generated = 0
    pending: list[dict[str, object]] = []
    worker_index = max(0, int(args.worker_index))
    worker_stride = max(1, int(args.worker_stride))
    batch_size = max(1, int(args.batch_size))

    def flush_batch(batch: list[dict[str, object]]) -> None:
        nonlocal generated
        if not batch:
            return
        try:
            response_text = chat_completion(
                cfg,
                system_prompt=sys_prompt,
                user_prompt=_build_batch_user_prompt(
                    [{"id": str(item["id"]), "prompt": str(item["prompt"])} for item in batch]
                ),
            )
            payload = _extract_json_block(response_text)
            responses = _extract_batch_items(payload)
            expected_ids = {str(item["id"]) for item in batch}
            missing_ids = sorted(expected_ids.difference(responses))
            extra_ids = sorted(set(responses).difference(expected_ids))
            if missing_ids or extra_ids:
                raise ValueError(
                    f"batch_id_mismatch:missing={missing_ids[:5]} extra={extra_ids[:5]}"
                )
            for item in batch:
                rid = str(item["id"])
                rec = dict(item["obj"])
                rec.update(
                    {
                        "id": rid,
                        "prompt": str(item["prompt"]),
                        "response": responses[rid],
                        "teacher_model": cfg.model,
                        "source": "openai_distill",
                        "line_no": int(item["line_no"]),
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
                append_jsonl(output_path, rec)
                state["succeeded"] = int(state.get("succeeded", 0)) + 1
                generated += 1
        except (OpenAIError, ValueError, json.JSONDecodeError) as e:
            reason = _classify_failure(str(e))
            for item in batch:
                state["failed"] = int(state.get("failed", 0)) + 1
                append_jsonl(
                    errors_path,
                    {
                        "id": str(item["id"]),
                        "line_no": int(item["line_no"]),
                        "reason": reason,
                        "error": str(e),
                        "prompt": str(item["prompt"]),
                    },
                )

    for line_no, obj in iter_jsonl(input_path):
        if line_no < start_line:
            continue
        if not _should_process_line(line_no, worker_index=worker_index, worker_stride=worker_stride):
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

        pending.append({"id": rid, "prompt": prompt, "line_no": line_no, "obj": obj})
        if len(pending) >= batch_size:
            flush_batch(pending)
            pending = []
            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)
            _save_state(state_path, state)
            if int(state.get("processed", 0)) % max(1, args.print_every) == 0:
                print(
                    f"processed={state['processed']} succeeded={state['succeeded']} failed={state['failed']} last_line={state['last_line']}"
                )

    if pending and not (args.max_items > 0 and generated >= args.max_items):
        flush_batch(pending)
        if args.sleep_sec > 0:
            time.sleep(args.sleep_sec)
        _save_state(state_path, state)

    _save_state(state_path, state)
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
