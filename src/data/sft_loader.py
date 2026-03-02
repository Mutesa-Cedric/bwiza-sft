"""SFT JSONL loading and tokenization helpers."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass(frozen=True)
class SFTSummary:
    path: str
    rows: int
    valid_rows: int
    invalid_rows: int
    avg_prompt_chars: float
    avg_response_chars: float

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def _normalize_text(x: object) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return " ".join(x.strip().split())


def _extract_prompt_response(obj: dict) -> tuple[str, str]:
    if "prompt" in obj and "response" in obj:
        return _normalize_text(obj.get("prompt")), _normalize_text(obj.get("response"))

    if "instruction" in obj and "output" in obj:
        instr = _normalize_text(obj.get("instruction"))
        inp = _normalize_text(obj.get("input"))
        prompt = instr if not inp else f"{instr}\n{inp}"
        return prompt, _normalize_text(obj.get("output"))

    if "messages" in obj and isinstance(obj["messages"], list):
        msgs = [m for m in obj["messages"] if isinstance(m, dict)]
        user = ""
        assistant = ""
        for m in msgs:
            role = _normalize_text(m.get("role", "")).lower()
            content = _normalize_text(m.get("content", ""))
            if role == "user" and content and not user:
                user = content
            elif role == "assistant" and content and not assistant:
                assistant = content
        return user, assistant

    return "", ""


def iter_prompt_response(path: str | Path):
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            prompt, response = _extract_prompt_response(obj)
            if prompt and response:
                yield i, prompt, response


def iter_tokenized_examples(
    path: str | Path,
    tokenizer,
    seq_len: int,
    user_prefix: str = "User: ",
    assistant_prefix: str = "Assistant: ",
):
    for _, prompt, response in iter_prompt_response(path):
        sample = build_sft_tokens(
            tokenizer=tokenizer,
            prompt=prompt,
            response=response,
            seq_len=seq_len,
            user_prefix=user_prefix,
            assistant_prefix=assistant_prefix,
        )
        if sample is not None:
            yield sample


def sample_prompts(path: str | Path, limit: int = 5) -> list[str]:
    prompts: list[str] = []
    for _, prompt, _ in iter_prompt_response(path):
        prompts.append(prompt)
        if len(prompts) >= limit:
            break
    return prompts


def summarize_jsonl(path: str | Path) -> SFTSummary:
    p = Path(path)
    rows = 0
    valid = 0
    bad = 0
    pchars = 0
    rchars = 0

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            rows += 1
            line = line.strip()
            if not line:
                bad += 1
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not isinstance(obj, dict):
                bad += 1
                continue
            prompt, response = _extract_prompt_response(obj)
            if not prompt or not response:
                bad += 1
                continue
            valid += 1
            pchars += len(prompt)
            rchars += len(response)

    return SFTSummary(
        path=str(p),
        rows=rows,
        valid_rows=valid,
        invalid_rows=bad,
        avg_prompt_chars=(pchars / valid) if valid else 0.0,
        avg_response_chars=(rchars / valid) if valid else 0.0,
    )


def build_sft_tokens(
    tokenizer,
    prompt: str,
    response: str,
    seq_len: int,
    user_prefix: str = "User: ",
    assistant_prefix: str = "Assistant: ",
) -> tuple[list[int], list[int], int] | None:
    prompt_text = f"{user_prefix}{prompt}\n{assistant_prefix}"
    full_text = f"{prompt_text}{response}{tokenizer.eos_token or ''}"

    p_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    f_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    if not f_ids:
        return None

    if seq_len > 0 and len(f_ids) > seq_len:
        f_ids = f_ids[:seq_len]

    labels = f_ids.copy()
    prompt_len = min(len(p_ids), len(labels))
    for i in range(prompt_len):
        labels[i] = -100

    supervised = sum(1 for x in labels if x != -100)
    if supervised <= 0:
        return None
    return f_ids, labels, supervised
