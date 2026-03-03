#!/usr/bin/env python3
"""Generate prompt seed JSONL via Gemini with resumability and quality checks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha1
import json
import re
from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.gemini_client import GeminiConfig, GeminiError, generate_text
from src.data.sft_records import append_jsonl, normalize_text

DEFAULT_TOPICS = [
    "uburezi mu Rwanda",
    "ubuhinzi n'ubworozi",
    "ubuzima rusange",
    "ikoranabuhanga",
    "ubukungu bw'umuryango",
    "amakuru yo ku mbuga nkoranyambaga",
    "uburere bw'abana",
    "kwiga no gutegura ibizamini",
    "akazi n'imyuga",
    "iterambere ry'icyaro",
    "ubwikorezi bwo mu mujyi",
    "imihindagurikire y'ibihe",
    "ubucuruzi buto",
    "imiyoborere myiza",
    "ubukerarugendo",
    "kubungabunga umuco",
    "serivisi za leta",
    "uburenganzira bw'abaturage",
]

ALLOWED_TASK_TYPES = {
    "rw_instruction",
    "code_switch_instruction",
    "multilingual_retention",
    "language_control",
}

ALLOWED_LANG_MODES = {
    "rw",
    "rw_mixed",
    "en",
    "fr",
    "sw",
    "control",
}

TASK_LANG_MODE_COMPAT: dict[str, set[str]] = {
    "rw_instruction": {"rw"},
    "code_switch_instruction": {"rw_mixed"},
    "multilingual_retention": {"en", "fr", "sw"},
    "language_control": {"control"},
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a senior native Kinyarwanda linguist and LLM data curator. "
    "Generate high-quality user prompts for supervised fine-tuning. "
    "Avoid grammar/spelling mistakes in Kinyarwanda. "
    "Use natural, common Rwanda usage. "
    "If you are uncertain about a wording, rewrite to simpler and safer Kinyarwanda instead of guessing. "
    "Keep prompts realistic and diverse. "
    "Output STRICT JSON only."
)

PROMPT_SEED_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "task_type": {
                        "type": "string",
                        "enum": [
                            "rw_instruction",
                            "code_switch_instruction",
                            "multilingual_retention",
                            "language_control",
                        ],
                    },
                    "lang_mode": {
                        "type": "string",
                        "enum": ["rw", "rw_mixed", "en", "fr", "sw", "control"],
                    },
                },
                "required": ["prompt", "task_type", "lang_mode"],
            },
        }
    },
    "required": ["items"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SFT prompt seed via Gemini")
    p.add_argument("--output_jsonl", default="outputs/sft/prompts.seed.gemini.jsonl")
    p.add_argument("--errors_jsonl", default="")
    p.add_argument("--state_path", default="")
    p.add_argument("--topics_file", default="")
    p.add_argument("--target", type=int, default=500)
    p.add_argument("--batch_prompts", type=int, default=8)
    p.add_argument("--model", default="gemini-3.1-pro-preview")
    p.add_argument("--api_key_env", default="GEMINI_API_KEY")
    p.add_argument("--env_file", default=".env")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_output_tokens", type=int, default=2048)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--retry_backoff_sec", type=float, default=1.5)
    p.add_argument("--sleep_sec", type=float, default=0.2)
    p.add_argument("--print_every", type=int, default=10)
    return p.parse_args()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
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


def _load_topics(path: str) -> list[str]:
    if not path:
        return DEFAULT_TOPICS

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing topics file: {p}")

    topics: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        t = normalize_text(raw)
        if t:
            topics.append(t)
    if not topics:
        raise RuntimeError("Topics file is empty after normalization")
    return topics


def _load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "topic_index": 0,
        "generated": 0,
        "accepted": 0,
        "failed_calls": 0,
        "rejected_prompts": 0,
        "updated_at": "",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def _save_state(path: Path, state: dict) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_existing_prompts(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                p = normalize_text(obj.get("prompt", "")).lower()
                if p:
                    seen.add(p)
    return seen


def _prompt_id(prompt: str) -> str:
    return sha1(prompt.encode("utf-8")).hexdigest()[:16]


def _extract_json_block(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
        if s.startswith("json"):
            s = s[4:].strip()

    # Fast path: already valid JSON.
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Fallback: extract largest JSON object span.
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError("no_json_object_found")
    block = s[start : end + 1]

    try:
        obj = json.loads(block)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        # Common LLM glitch: trailing commas before closing braces/brackets.
        repaired = re.sub(r",(\s*[}\]])", r"\1", block)
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return obj

    raise ValueError("json_object_not_dict")


def _build_user_prompt(topic: str, n: int) -> str:
    return (
        "Generate prompt seed items for SFT.\\n"
        "Topic: " + topic + "\\n"
        f"Need exactly {n} items.\\n"
        "Output JSON object with this exact schema:\\n"
        "{\"items\":[{\"prompt\":\"...\",\"task_type\":\"rw_instruction|code_switch_instruction|multilingual_retention|language_control\",\"lang_mode\":\"rw|rw_mixed|en|fr|sw|control\"}]}\\n"
        "Constraints:\\n"
        "- At least 50% items with lang_mode=rw.\\n"
        "- Include at least 1 code_switch item.\\n"
        "- Include at least 1 language_control item.\\n"
        "- Prompts must be realistic user requests.\\n"
        "- Keep each prompt <= 220 chars.\\n"
        "- Do not include answers, only prompts.\\n"
        "- Return JSON only."
    )


def _validate_item(item: dict) -> tuple[bool, str]:
    prompt = normalize_text(item.get("prompt", ""))
    task_type = normalize_text(item.get("task_type", ""))
    lang_mode = normalize_text(item.get("lang_mode", ""))

    if not prompt:
        return False, "empty_prompt"
    if len(prompt) < 10:
        return False, "prompt_too_short"
    if len(prompt) > 260:
        return False, "prompt_too_long"
    if task_type not in ALLOWED_TASK_TYPES:
        return False, "invalid_task_type"
    if lang_mode not in ALLOWED_LANG_MODES:
        return False, "invalid_lang_mode"
    allowed = TASK_LANG_MODE_COMPAT.get(task_type, set())
    if lang_mode not in allowed:
        return False, "task_lang_mode_mismatch"
    return True, "ok"


def main() -> int:
    args = parse_args()

    _load_env_file(Path(args.env_file))
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Missing API key in env var: {args.api_key_env} (checked env file: {args.env_file})"
        )

    output_path = Path(args.output_jsonl)
    errors_path = Path(args.errors_jsonl) if args.errors_jsonl else output_path.with_suffix(".errors.jsonl")
    state_path = Path(args.state_path) if args.state_path else output_path.with_suffix(".state.json")

    topics = _load_topics(args.topics_file)
    seen_prompts = _read_existing_prompts(output_path)

    cfg = GeminiConfig(
        model=args.model,
        api_key=api_key,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_output_tokens=int(args.max_output_tokens),
        max_retries=int(args.max_retries),
        retry_backoff_sec=float(args.retry_backoff_sec),
        response_mime_type="application/json",
        response_schema=PROMPT_SEED_RESPONSE_SCHEMA,
    )

    state = _load_state(state_path)
    topic_index = int(state.get("topic_index", 0))

    while int(state.get("accepted", 0)) < int(args.target):
        topic = topics[topic_index % len(topics)]
        user_prompt = _build_user_prompt(topic, int(args.batch_prompts))

        try:
            text = generate_text(cfg, user_prompt=user_prompt, system_prompt=DEFAULT_SYSTEM_PROMPT)
            try:
                payload = _extract_json_block(text)
            except (ValueError, json.JSONDecodeError) as parse_err:
                raise ValueError(f"invalid_json_output:{parse_err}; head={text[:320]!r}") from parse_err
            items = payload.get("items") if isinstance(payload, dict) else None
            if not isinstance(items, list):
                raise ValueError("items_not_list")

            for item in items:
                if int(state.get("accepted", 0)) >= int(args.target):
                    break
                if not isinstance(item, dict):
                    state["rejected_prompts"] = int(state.get("rejected_prompts", 0)) + 1
                    append_jsonl(errors_path, {"topic": topic, "reason": "item_not_object", "item": item})
                    continue

                ok, reason = _validate_item(item)
                if not ok:
                    state["rejected_prompts"] = int(state.get("rejected_prompts", 0)) + 1
                    append_jsonl(errors_path, {"topic": topic, "reason": reason, "item": item})
                    continue

                prompt = normalize_text(item.get("prompt", ""))
                key = prompt.lower()
                if key in seen_prompts:
                    state["rejected_prompts"] = int(state.get("rejected_prompts", 0)) + 1
                    append_jsonl(errors_path, {"topic": topic, "reason": "duplicate_prompt", "prompt": prompt})
                    continue

                seen_prompts.add(key)
                rec = {
                    "id": f"gm_{_prompt_id(prompt)}",
                    "prompt": prompt,
                    "task_type": item.get("task_type"),
                    "lang_mode": item.get("lang_mode"),
                    "topic": topic,
                    "source": "gemini_prompt_seed_v1",
                    "teacher_model": cfg.model,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                append_jsonl(output_path, rec)
                state["accepted"] = int(state.get("accepted", 0)) + 1

            state["generated"] = int(state.get("generated", 0)) + len(items)

        except (GeminiError, ValueError, json.JSONDecodeError) as e:
            state["failed_calls"] = int(state.get("failed_calls", 0)) + 1
            append_jsonl(
                errors_path,
                {
                    "topic": topic,
                    "reason": "gemini_prompt_generation_error",
                    "error": str(e),
                },
            )

        topic_index += 1
        state["topic_index"] = topic_index
        _save_state(state_path, state)

        if topic_index % max(1, int(args.print_every)) == 0:
            print(
                f"accepted={state['accepted']} generated={state['generated']} rejected={state['rejected_prompts']} failed_calls={state['failed_calls']} topic_index={state['topic_index']}"
            )

        if args.sleep_sec > 0:
            time.sleep(float(args.sleep_sec))

    _save_state(state_path, state)
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
