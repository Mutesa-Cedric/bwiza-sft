#!/usr/bin/env python3
"""Hybrid prompt seed pipeline: Gemini Flash generation + OpenAI organizer normalization."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha1
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dedup_index import PromptDedupIndex
from src.data.gemini_client import GeminiConfig, GeminiError, generate_text
from src.data.openai_client import OpenAIConfig, OpenAIError, chat_completion
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
    "followup_clarification",
    "safety_refusal",
    "transformation",
    "structured_output",
    "noisy_input_robustness",
    "open_domain_chat",
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
    "multilingual_retention": {"en", "fr", "sw", "rw_mixed"},
    "language_control": {"control", "rw", "rw_mixed", "en", "fr", "sw"},
    "followup_clarification": {"rw", "rw_mixed"},
    "safety_refusal": {"rw", "rw_mixed", "control"},
    "transformation": {"rw", "rw_mixed", "en", "fr", "sw"},
    "structured_output": {"rw", "rw_mixed", "en", "fr", "sw"},
    "noisy_input_robustness": {"rw", "rw_mixed"},
    "open_domain_chat": {"rw", "rw_mixed", "en", "fr", "sw"},
}

TASK_TYPE_CYCLE = [
    "rw_instruction",
    "followup_clarification",
    "transformation",
    "safety_refusal",
    "noisy_input_robustness",
    "code_switch_instruction",
    "multilingual_retention",
    "structured_output",
    "language_control",
    "open_domain_chat",
]

TASK_TYPE_ENUM = "|".join(sorted(ALLOWED_TASK_TYPES))

FLASH_SYSTEM_PROMPT = (
    "Generate one realistic user prompt for SFT data. "
    "Output ONLY one minified JSON object with keys: prompt, task_type, lang_mode. "
    "No markdown. No code fences. No extra text."
)

ORGANIZER_SYSTEM_PROMPT = (
    "You normalize prompt-seed outputs into valid JSON. "
    "Return ONLY one minified JSON object and nothing else. "
    "If unusable, return: {\"reject_reason\":\"<short_reason>\"}."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hybrid prompt seed pipeline (Flash + organizer)")
    p.add_argument("--output_prefix", default="outputs/sft/prompts.seed.hybrid")
    p.add_argument(
        "--shared_output_prefix",
        default="",
        help="Optional shared prefix for canonical JSONL outputs. State remains under output_prefix.",
    )
    p.add_argument("--topics_file", default="")
    p.add_argument("--target", type=int, default=30000)
    p.add_argument("--topic_index_start", type=int, default=0)
    p.add_argument("--flash_batch_size", type=int, default=4)
    p.add_argument("--gemini_model", default="gemini-3-flash-preview")
    p.add_argument("--organizer_model", default="gpt-5.2")
    p.add_argument(
        "--disable_organizer",
        action="store_true",
        help="Disable organizer repair pass and skip OpenAI requirements.",
    )
    p.add_argument("--env_file", default=".env")
    p.add_argument("--gemini_api_key_env", default="GEMINI_API_KEY")
    p.add_argument("--openai_api_key_env", default="OPENAI_API_KEY")
    p.add_argument("--dedup_db", default="outputs/sft/prompts.seed.shared.dedup.sqlite")
    p.add_argument("--sleep_sec", type=float, default=0.2)
    p.add_argument("--sleep_jitter_sec", type=float, default=0.5)
    p.add_argument("--failure_cooldown_sec", type=float, default=4.0)
    p.add_argument("--failure_cooldown_cap_sec", type=float, default=20.0)
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument(
        "--max_output_tokens",
        type=int,
        default=0,
        help="Gemini max output tokens. Set 0 to omit explicit cap.",
    )
    p.add_argument("--max_retries", type=int, default=8)
    p.add_argument("--retry_backoff_sec", type=float, default=2.0)
    p.add_argument("--request_timeout_sec", type=float, default=30.0)
    p.add_argument("--near_dup_threshold", type=float, default=0.9)
    p.add_argument("--near_dup_topic_limit", type=int, default=200)
    p.add_argument("--near_dup_global_limit", type=int, default=400)
    p.add_argument("--avoid_topic_recent", type=int, default=12)
    p.add_argument("--avoid_global_recent", type=int, default=12)
    p.add_argument("--avoid_pattern_topk", type=int, default=8)
    p.add_argument("--cb_consecutive_429", type=int, default=12)
    p.add_argument("--cb_consecutive_parse", type=int, default=12)
    p.add_argument("--cb_consecutive_fail", type=int, default=30)
    p.add_argument(
        "--task_type_targeting",
        choices=["none", "soft", "strict"],
        default="soft",
        help="none: no targeting; soft: request type but accept all valid; strict: enforce requested type.",
    )
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


def _prompt_id(prompt: str) -> str:
    return sha1(prompt.encode("utf-8")).hexdigest()[:16]


def _extract_json_block(text: str) -> dict[str, Any]:
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

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end <= start:
        partial = _extract_partial_item(s)
        if partial is not None:
            return partial
        raise ValueError("no_json_object_found")
    block = s[start : end + 1]
    repaired = re.sub(r",(\s*[}\]])", r"\1", block)
    try:
        obj = json.loads(repaired)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    partial = _extract_partial_item(s)
    if partial is not None:
        return partial
    raise ValueError("json_object_not_dict")


def _decode_json_string_fragment(raw: str) -> str:
    try:
        return normalize_text(json.loads(f"\"{raw}\""))
    except Exception:
        return normalize_text(raw.replace("\\n", " ").replace('\\"', '"'))


def _extract_partial_item(text: str) -> dict[str, str] | None:
    prompt_m = re.search(r'"prompt"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    task_m = re.search(
        rf'"task_type"\s*:\s*"({TASK_TYPE_ENUM})"',
        text,
    )
    lang_m = re.search(r'"lang_mode"\s*:\s*"(rw|rw_mixed|en|fr|sw|control)"', text)
    if not (prompt_m and task_m and lang_m):
        return None
    prompt = _decode_json_string_fragment(prompt_m.group(1))
    if not prompt:
        return None
    return {"prompt": prompt, "task_type": task_m.group(1), "lang_mode": lang_m.group(1)}


def _validate_item(item: dict[str, Any]) -> tuple[bool, str]:
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


def _build_flash_user_prompt(
    topic: str,
    n: int,
    *,
    required_task_type: str,
    targeting_mode: str,
    avoid_prompts: list[str],
    frequent_patterns: list[str],
) -> str:
    avoid_block = ""
    if avoid_prompts:
        avoid_block += "Avoid same/very similar prompts as:\n"
        avoid_block += "\n".join([f"- {p}" for p in avoid_prompts]) + "\n"
    if frequent_patterns:
        avoid_block += "Avoid these common openings:\n"
        avoid_block += "\n".join([f"- {p}" for p in frequent_patterns]) + "\n"
    return (
        "Create prompt-seed items for supervised fine-tuning.\n"
        f"Topic: {topic}\n"
        f"Need exactly {n} items.\n"
        "Return one JSON object exactly in this schema:\n"
        "{\"items\":[{\"prompt\":\"<string>\",\"task_type\":\""
        + TASK_TYPE_ENUM
        + "\","
        "\"lang_mode\":\"rw|rw_mixed|en|fr|sw|control\"}]}\n"
        "Rules:\n"
        "- prompt must be realistic and <= 220 chars\n"
        "- no answer text, prompt only\n"
        + (
            f"- preferred task_type is {required_task_type}\n"
            if targeting_mode in {"soft", "strict"}
            else ""
        )
        + avoid_block
        + "- output JSON only"
    )


def _build_organizer_user_prompt(
    *,
    topic: str,
    raw_text: str,
    failure_reason: str,
    n: int,
    required_task_type: str,
    targeting_mode: str,
    avoid_prompts: list[str],
    frequent_patterns: list[str],
) -> str:
    avoid_block = ""
    if avoid_prompts:
        avoid_block += "Do not return prompts same/very similar to:\n"
        avoid_block += "\n".join([f"- {p}" for p in avoid_prompts]) + "\n"
    if frequent_patterns:
        avoid_block += "Avoid these common openings:\n"
        avoid_block += "\n".join([f"- {p}" for p in frequent_patterns]) + "\n"
    return (
        "Normalize the raw model output into valid prompt-seed JSON.\n"
        f"Topic: {topic}\n"
        f"Target item count: up to {n}\n"
        f"Local parse failure: {failure_reason}\n"
        "Return exactly one minified JSON object in one of these forms:\n"
        "{\"items\":[{\"prompt\":\"<string>\",\"task_type\":\""
        + TASK_TYPE_ENUM
        + "\","
        "\"lang_mode\":\"rw|rw_mixed|en|fr|sw|control\"}]}\n"
        "OR\n"
        "{\"reject_reason\":\"<short_reason>\"}\n"
        + (
            f"- preferred task_type is {required_task_type}\n"
            if targeting_mode in {"soft", "strict"}
            else ""
        )
        + avoid_block
        + "Raw output follows:\n"
        f"{raw_text}"
    )


def _payload_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("items")
    if isinstance(items, list):
        return [x for x in items if isinstance(x, dict)]
    if {"prompt", "task_type", "lang_mode"}.issubset(set(payload.keys())):
        return [payload]
    return []


def _collect_avoid_memory(
    dedup: PromptDedupIndex,
    topic: str,
    *,
    topic_recent: int,
    global_recent: int,
    pattern_topk: int,
) -> tuple[list[str], list[str]]:
    topic_prompts = dedup.recent_prompts(limit=max(0, topic_recent), topic=topic)
    global_prompts = dedup.recent_prompts(limit=max(0, global_recent), topic="")
    merged: list[str] = []
    seen: set[str] = set()
    for p in topic_prompts + global_prompts:
        key = normalize_text(p).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(normalize_text(p))
    patterns = dedup.frequent_patterns(limit=max(0, pattern_topk))
    return merged, patterns


def _out_path(prefix: Path, suffix: str) -> Path:
    # Preserve literal prefix (e.g. prompts.seed.hybrid) instead of stripping extension.
    return Path(f"{prefix}.{suffix}")


def _classify_failure(err: str) -> tuple[bool, bool]:
    e = normalize_text(err).lower()
    is_429 = "http_429" in e
    is_parse = (
        "invalid_json_output" in e
        or "no_json_object_found" in e
        or "json_object_not_dict" in e
        or "invalid_payload_not_object" in e
    )
    return is_429, is_parse


def _circuit_break_reason(
    *,
    consecutive_429: int,
    consecutive_parse: int,
    consecutive_fail: int,
    max_consecutive_429: int,
    max_consecutive_parse: int,
    max_consecutive_fail: int,
) -> str:
    if max_consecutive_429 > 0 and consecutive_429 >= max_consecutive_429:
        return "consecutive_429_limit_reached"
    if max_consecutive_parse > 0 and consecutive_parse >= max_consecutive_parse:
        return "consecutive_parse_limit_reached"
    if max_consecutive_fail > 0 and consecutive_fail >= max_consecutive_fail:
        return "consecutive_fail_limit_reached"
    return ""


def _compute_post_iteration_sleep(
    *,
    base_sleep_sec: float,
    sleep_jitter_sec: float,
    had_api_failure: bool,
    consecutive_fail: int,
    failure_cooldown_sec: float,
    failure_cooldown_cap_sec: float,
) -> float:
    sleep_for = max(0.0, float(base_sleep_sec))
    if sleep_jitter_sec > 0:
        sleep_for += random.uniform(0.0, float(sleep_jitter_sec))
    if had_api_failure and consecutive_fail > 0 and failure_cooldown_sec > 0:
        extra = min(
            float(failure_cooldown_cap_sec),
            float(failure_cooldown_sec) * max(1, int(consecutive_fail)),
        )
        sleep_for += extra
    return sleep_for


def _load_state(path: Path, topic_index_start: int = 0) -> dict[str, Any]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "topic_index": max(0, int(topic_index_start)),
        "flash_calls": 0,
        "organizer_calls": 0,
        "accepted_local": 0,
        "accepted_organized": 0,
        "rejected": 0,
        "failed_calls": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": "",
    }


def _save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def _accept_record(
    *,
    item: dict[str, Any],
    topic: str,
    source: str,
    model_name: str,
    dedup: PromptDedupIndex,
    dedup_threshold: float,
    dedup_topic_limit: int,
    dedup_global_limit: int,
    required_task_type: str,
    targeting_mode: str,
    final_path: Path,
    local_path: Path,
    organized_path: Path,
) -> tuple[bool, str]:
    ok, reason = _validate_item(item)
    if not ok:
        return False, reason
    if targeting_mode == "strict" and normalize_text(item["task_type"]) != required_task_type:
        return False, "task_type_not_requested"
    prompt = normalize_text(item["prompt"])
    near_dup, near_match = dedup.has_near_duplicate(
        prompt=prompt,
        topic=topic,
        threshold=dedup_threshold,
        topic_limit=dedup_topic_limit,
        global_limit=dedup_global_limit,
    )
    if near_dup:
        return False, f"near_duplicate:{near_match}"

    key = prompt.lower()
    inserted = dedup.add_if_new(
        prompt_key=key,
        first_prompt=prompt,
        first_source=source,
        topic=topic,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    if not inserted:
        return False, "duplicate_prompt"

    rec = {
        "id": f"hy_{_prompt_id(prompt)}",
        "prompt": prompt,
        "task_type": normalize_text(item["task_type"]),
        "content_type": normalize_text(item["task_type"]),
        "lang_mode": normalize_text(item["lang_mode"]),
        "topic": topic,
        "source": source,
        "teacher_model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    append_jsonl(final_path, rec)
    if source == "flash_local":
        append_jsonl(local_path, rec)
    else:
        append_jsonl(organized_path, rec)
    return True, "ok"


def main() -> int:
    args = parse_args()
    _load_env_file(Path(args.env_file))

    gemini_key = os.environ.get(args.gemini_api_key_env, "").strip()
    if not gemini_key:
        raise RuntimeError(
            f"Missing Gemini key in env var: {args.gemini_api_key_env} (checked env file: {args.env_file})"
        )
    openai_key = ""
    if not args.disable_organizer:
        openai_key = os.environ.get(args.openai_api_key_env, "").strip()
        if not openai_key:
            raise RuntimeError(
                f"Missing OpenAI key in env var: {args.openai_api_key_env} (checked env file: {args.env_file})"
            )

    out_prefix = Path(args.output_prefix)
    shared_prefix = Path(args.shared_output_prefix) if args.shared_output_prefix else out_prefix
    final_path = _out_path(shared_prefix, "final.jsonl")
    local_path = _out_path(shared_prefix, "accepted_local.jsonl")
    organized_path = _out_path(shared_prefix, "accepted_organized.jsonl")
    rejects_path = _out_path(shared_prefix, "rejects.jsonl")
    errors_path = _out_path(shared_prefix, "errors.jsonl")
    raw_path = _out_path(shared_prefix, "raw.jsonl")
    state_path = _out_path(out_prefix, "state.json")

    topics = _load_topics(args.topics_file)
    state = _load_state(state_path, topic_index_start=int(args.topic_index_start))
    topic_index = int(state.get("topic_index", 0))

    flash_cfg = GeminiConfig(
        model=args.gemini_model,
        api_key=gemini_key,
        temperature=0.0,
        top_p=0.1,
        max_output_tokens=(int(args.max_output_tokens) if int(args.max_output_tokens) > 0 else None),
        max_retries=int(args.max_retries),
        retry_backoff_sec=float(args.retry_backoff_sec),
        request_timeout_sec=float(args.request_timeout_sec),
    )
    org_cfg = None
    if not args.disable_organizer:
        org_cfg = OpenAIConfig(
            model=args.organizer_model,
            api_key=openai_key,
            temperature=0.0,
            max_output_tokens=300,
            max_retries=int(args.max_retries),
            retry_backoff_sec=float(args.retry_backoff_sec),
        )

    consecutive_429 = 0
    consecutive_parse = 0
    consecutive_fail = 0

    with PromptDedupIndex(Path(args.dedup_db)) as dedup:
        while (int(state.get("accepted_local", 0)) + int(state.get("accepted_organized", 0))) < int(args.target):
            had_api_failure = False
            topic = topics[topic_index % len(topics)]
            required_task_type = TASK_TYPE_CYCLE[topic_index % len(TASK_TYPE_CYCLE)]
            accepted_before = int(state.get("accepted_local", 0)) + int(state.get("accepted_organized", 0))
            remaining = int(args.target) - accepted_before
            batch_n = max(1, min(int(args.flash_batch_size), remaining))
            avoid_prompts, frequent_patterns = _collect_avoid_memory(
                dedup,
                topic,
                topic_recent=int(args.avoid_topic_recent),
                global_recent=int(args.avoid_global_recent),
                pattern_topk=int(args.avoid_pattern_topk),
            )
            raw_text = ""
            parse_reason = ""
            local_failures: list[str] = []
            try:
                raw_text = generate_text(
                    flash_cfg,
                    user_prompt=_build_flash_user_prompt(
                        topic,
                        batch_n,
                        required_task_type=required_task_type,
                        targeting_mode=str(args.task_type_targeting),
                        avoid_prompts=avoid_prompts,
                        frequent_patterns=frequent_patterns,
                    ),
                    system_prompt=FLASH_SYSTEM_PROMPT,
                )
                state["flash_calls"] = int(state.get("flash_calls", 0)) + 1
                append_jsonl(
                    raw_path,
                    {
                        "topic": topic,
                        "raw_output": raw_text,
                        "model": args.gemini_model,
                        "requested_items": batch_n,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                )

                flash_items: list[dict[str, Any]] = []
                try:
                    payload = _extract_json_block(raw_text)
                    flash_items = _payload_items(payload)
                    if not flash_items:
                        parse_reason = "parsed_without_required_fields"
                except (ValueError, json.JSONDecodeError) as e:
                    parse_reason = f"invalid_json_output:{e}"

                for item in flash_items:
                    if (int(state.get("accepted_local", 0)) + int(state.get("accepted_organized", 0))) >= int(args.target):
                        break
                    accepted, reason = _accept_record(
                        item=item,
                        topic=topic,
                        source="flash_local",
                        model_name=args.gemini_model,
                        dedup=dedup,
                        dedup_threshold=float(args.near_dup_threshold),
                        dedup_topic_limit=int(args.near_dup_topic_limit),
                        dedup_global_limit=int(args.near_dup_global_limit),
                        required_task_type=required_task_type,
                        targeting_mode=str(args.task_type_targeting),
                        final_path=final_path,
                        local_path=local_path,
                        organized_path=organized_path,
                    )
                    if accepted:
                        state["accepted_local"] = int(state.get("accepted_local", 0)) + 1
                    else:
                        local_failures.append(reason)

                # Organizer pass for parse/validation failures or partial local rejects.
                # Organizer is a structural-repair fallback.
                # When Flash already returned parseable JSON, local failures are usually
                # dedup/topic/compat checks that organizer cannot reliably improve.
                need_organizer = bool(parse_reason) and not args.disable_organizer
                if need_organizer and ((int(state.get("accepted_local", 0)) + int(state.get("accepted_organized", 0))) < int(args.target)):
                    fail_summary = parse_reason or ",".join(sorted(set(local_failures))[:8])
                    org_text = chat_completion(
                        org_cfg,
                        system_prompt=ORGANIZER_SYSTEM_PROMPT,
                        user_prompt=_build_organizer_user_prompt(
                            topic=topic,
                            raw_text=raw_text,
                            failure_reason=fail_summary,
                            n=batch_n,
                            required_task_type=required_task_type,
                            targeting_mode=str(args.task_type_targeting),
                            avoid_prompts=avoid_prompts,
                            frequent_patterns=frequent_patterns,
                        ),
                    )
                    state["organizer_calls"] = int(state.get("organizer_calls", 0)) + 1
                    org_obj = _extract_json_block(org_text)
                    if "reject_reason" in org_obj:
                        state["rejected"] = int(state.get("rejected", 0)) + 1
                        consecutive_429 = 0
                        consecutive_parse = 0
                        consecutive_fail = 0
                        append_jsonl(
                            rejects_path,
                            {
                                "topic": topic,
                                "reason": normalize_text(org_obj.get("reject_reason", "organizer_reject")),
                                "raw_output": raw_text,
                                "organizer_output": org_text,
                            },
                        )
                    else:
                        org_items = _payload_items(org_obj)
                        if not org_items:
                            raise ValueError("organizer_items_not_list")
                        for item in org_items:
                            if (int(state.get("accepted_local", 0)) + int(state.get("accepted_organized", 0))) >= int(args.target):
                                break
                            accepted, reason = _accept_record(
                                item=item,
                                topic=topic,
                                source="organized_gpt",
                                model_name=args.organizer_model,
                                dedup=dedup,
                                dedup_threshold=float(args.near_dup_threshold),
                                dedup_topic_limit=int(args.near_dup_topic_limit),
                                dedup_global_limit=int(args.near_dup_global_limit),
                                required_task_type=required_task_type,
                                targeting_mode=str(args.task_type_targeting),
                                final_path=final_path,
                                local_path=local_path,
                                organized_path=organized_path,
                            )
                            if accepted:
                                state["accepted_organized"] = int(state.get("accepted_organized", 0)) + 1
                            else:
                                state["rejected"] = int(state.get("rejected", 0)) + 1
                                append_jsonl(
                                    rejects_path,
                                    {
                                        "topic": topic,
                                        "reason": reason,
                                        "raw_output": raw_text,
                                        "organizer_output": org_text,
                                        "item": item,
                                    },
                                )

            except (GeminiError, OpenAIError, ValueError, json.JSONDecodeError) as e:
                had_api_failure = True
                state["failed_calls"] = int(state.get("failed_calls", 0)) + 1
                is_429, is_parse = _classify_failure(str(e))
                consecutive_429 = consecutive_429 + 1 if is_429 else 0
                consecutive_parse = consecutive_parse + 1 if is_parse else 0
                consecutive_fail += 1
                append_jsonl(
                    errors_path,
                    {
                        "topic": topic,
                        "reason": "hybrid_generation_error",
                        "error": str(e),
                        "raw_output": raw_text,
                    },
                )

            accepted_after = int(state.get("accepted_local", 0)) + int(state.get("accepted_organized", 0))
            if accepted_after > accepted_before:
                consecutive_429 = 0
                consecutive_parse = 0
                consecutive_fail = 0

            topic_index += 1
            state["topic_index"] = topic_index
            state["dedup_db_entries"] = dedup.count()
            _save_state(state_path, state)

            if topic_index % max(1, int(args.print_every)) == 0:
                print(
                    "accepted_local="
                    f"{state['accepted_local']} accepted_organized={state['accepted_organized']} "
                    f"rejected={state['rejected']} failed_calls={state['failed_calls']} "
                    f"topic_index={state['topic_index']} dedup_entries={state['dedup_db_entries']} "
                    f"cb429={consecutive_429} cbparse={consecutive_parse} cbfail={consecutive_fail}"
                )

            cb_reason = _circuit_break_reason(
                consecutive_429=consecutive_429,
                consecutive_parse=consecutive_parse,
                consecutive_fail=consecutive_fail,
                max_consecutive_429=int(args.cb_consecutive_429),
                max_consecutive_parse=int(args.cb_consecutive_parse),
                max_consecutive_fail=int(args.cb_consecutive_fail),
            )
            if cb_reason:
                append_jsonl(
                    errors_path,
                    {
                        "reason": "circuit_breaker_stop",
                        "circuit_reason": cb_reason,
                        "topic_index": state["topic_index"],
                        "failed_calls": state["failed_calls"],
                        "accepted_local": state["accepted_local"],
                        "accepted_organized": state["accepted_organized"],
                    },
                )
                print(f"circuit_breaker_stop: {cb_reason}")
                break

            sleep_for = _compute_post_iteration_sleep(
                base_sleep_sec=float(args.sleep_sec),
                sleep_jitter_sec=float(args.sleep_jitter_sec),
                had_api_failure=had_api_failure,
                consecutive_fail=consecutive_fail,
                failure_cooldown_sec=float(args.failure_cooldown_sec),
                failure_cooldown_cap_sec=float(args.failure_cooldown_cap_sec),
            )
            if sleep_for > 0:
                time.sleep(sleep_for)

    _save_state(state_path, state)
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
