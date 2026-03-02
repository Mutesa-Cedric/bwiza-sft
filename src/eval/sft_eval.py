"""SFT evaluation helpers (base vs adapted)."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import math
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.sft_loader import iter_tokenized_examples, sample_prompts

EN_STOPWORDS = {
    "the",
    "and",
    "is",
    "are",
    "to",
    "of",
    "in",
    "that",
    "for",
    "with",
    "on",
    "this",
    "it",
    "as",
    "be",
    "by",
    "an",
    "from",
}


@dataclass
class LossMetrics:
    loss: float
    perplexity: float
    batches: int


@dataclass
class GenerationMetrics:
    avg_english_drift: float
    avg_rw_marker_density: float
    samples: list[dict[str, str | float]]


@dataclass
class ModelEvalReport:
    model_path: str
    val: LossMetrics
    test: LossMetrics
    generation: GenerationMetrics

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rw_marker_density(text: str) -> float:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 0.0
    markers = {
        "mu",
        "ku",
        "ni",
        "na",
        "ndi",
        "ariko",
        "kandi",
        "uko",
        "niba",
        "rwose",
        "cyangwa",
        "amakuru",
        "ubuzima",
    }
    return sum(1 for w in words if w in markers) / len(words)


def english_drift_score(text: str) -> float:
    words = re.findall(r"[A-Za-z']+", text.lower())
    if not words:
        return 0.0
    return sum(1 for w in words if w in EN_STOPWORDS) / len(words)


def _collate(batch, pad_id: int, device: torch.device):
    bsz = len(batch)
    max_len = max(len(x[0]) for x in batch)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, (ids, lbl, _) in enumerate(batch):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :n] = 1
        labels[i, :n] = torch.tensor(lbl, dtype=torch.long)
    return input_ids.to(device), attention_mask.to(device), labels.to(device)


def evaluate_loss(
    model,
    tokenizer,
    jsonl_path: str,
    pad_id: int,
    device: torch.device,
    seq_len: int,
    max_batches: int = 32,
    batch_size: int = 8,
) -> LossMetrics:
    model.eval()
    losses: list[float] = []
    b = 0
    batch = []

    with torch.no_grad():
        for sample in iter_tokenized_examples(
            jsonl_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            user_prefix="User: ",
            assistant_prefix="Assistant: ",
        ):
            batch.append(sample)
            if len(batch) < batch_size:
                continue

            inp, mask, labels = _collate(batch, pad_id, device)
            out = model(input_ids=inp, attention_mask=mask, labels=labels)
            losses.append(float(out.loss.detach().cpu().item()))
            b += 1
            batch = []
            if b >= max_batches:
                break

        if batch and b < max_batches:
            inp, mask, labels = _collate(batch, pad_id, device)
            out = model(input_ids=inp, attention_mask=mask, labels=labels)
            losses.append(float(out.loss.detach().cpu().item()))
            b += 1

    avg = float(sum(losses) / max(1, len(losses)))
    ppl = float(math.exp(min(avg, 20.0)))
    return LossMetrics(loss=avg, perplexity=ppl, batches=b)


def evaluate_generation(model, tokenizer, prompts: list[str], device: torch.device, max_new_tokens: int = 120) -> GenerationMetrics:
    model.eval()
    samples: list[dict[str, str | float]] = []
    with torch.no_grad():
        for p in prompts:
            input_text = f"User: {p}\nAssistant: "
            inputs = tokenizer(input_text, return_tensors="pt").to(device)
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            drift = english_drift_score(text)
            rw_density = _rw_marker_density(text)
            samples.append(
                {
                    "prompt": p,
                    "output": text,
                    "english_drift": drift,
                    "rw_marker_density": rw_density,
                }
            )

    avg_drift = sum(float(s["english_drift"]) for s in samples) / max(1, len(samples))
    avg_rw = sum(float(s["rw_marker_density"]) for s in samples) / max(1, len(samples))
    return GenerationMetrics(avg_english_drift=float(avg_drift), avg_rw_marker_density=float(avg_rw), samples=samples)


def evaluate_model(
    model_path: str,
    val_jsonl: str,
    test_jsonl: str,
    device: str = "cuda",
    max_eval_batches: int = 32,
    eval_batch_size: int = 8,
    seq_len: int = 2048,
) -> ModelEvalReport:
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype).to(dev)

    val = evaluate_loss(
        model=model,
        tokenizer=tokenizer,
        jsonl_path=val_jsonl,
        pad_id=tokenizer.pad_token_id,
        device=dev,
        seq_len=seq_len,
        max_batches=max_eval_batches,
        batch_size=eval_batch_size,
    )
    test = evaluate_loss(
        model=model,
        tokenizer=tokenizer,
        jsonl_path=test_jsonl,
        pad_id=tokenizer.pad_token_id,
        device=dev,
        seq_len=seq_len,
        max_batches=max_eval_batches,
        batch_size=eval_batch_size,
    )

    prompt_pool = sample_prompts(val_jsonl, limit=5)
    if not prompt_pool:
        prompt_pool = [
            "Muraho, umeze ute?",
            "Sobanura impamvu amazi ari ingenzi ku buzima.",
            "Ni gute twarwanya amakuru atari yo?",
            "Andika incamake y'inkuru y'uburezi.",
            "Sobanura ijambo ubwirakabiri.",
        ]

    generation = evaluate_generation(model, tokenizer, prompt_pool, dev)

    return ModelEvalReport(model_path=model_path, val=val, test=test, generation=generation)


def write_report(path: str | Path, payload: dict[str, Any]) -> None:
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
