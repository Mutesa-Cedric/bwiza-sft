"""SFT training loop for instruction datasets."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch
from transformers import Adafactor, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from src.data.sft_loader import iter_tokenized_examples, summarize_jsonl
from src.train.checkpointing import latest_checkpoint, load_resume_state, save_checkpoint


@dataclass
class SFTResult:
    final_step: int
    supervised_tokens_seen: int
    final_checkpoint: str


def _collate(batch: list[tuple[list[int], list[int], int]], pad_id: int, device: torch.device):
    bsz = len(batch)
    max_len = max(len(x[0]) for x in batch)

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)

    supervised = 0
    for i, (ids, lbl, sup) in enumerate(batch):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :n] = 1
        labels[i, :n] = torch.tensor(lbl, dtype=torch.long)
        supervised += int(sup)

    return input_ids.to(device), attention_mask.to(device), labels.to(device), supervised


def _eval_loss(
    model,
    val_jsonl: str,
    tokenizer,
    pad_id: int,
    device: torch.device,
    batch_size: int,
    max_batches: int,
    seq_len: int,
    user_prefix: str,
    assistant_prefix: str,
) -> float:
    model.eval()
    losses: list[float] = []
    b = 0
    batch: list[tuple[list[int], list[int], int]] = []

    with torch.no_grad():
        for sample in iter_tokenized_examples(
            val_jsonl,
            tokenizer=tokenizer,
            seq_len=seq_len,
            user_prefix=user_prefix,
            assistant_prefix=assistant_prefix,
        ):
            batch.append(sample)
            if len(batch) < batch_size:
                continue

            inp, mask, labels, _ = _collate(batch, pad_id=pad_id, device=device)
            out = model(input_ids=inp, attention_mask=mask, labels=labels)
            losses.append(float(out.loss.detach().cpu().item()))
            batch = []
            b += 1
            if b >= max_batches:
                break

        if batch and b < max_batches:
            inp, mask, labels, _ = _collate(batch, pad_id=pad_id, device=device)
            out = model(input_ids=inp, attention_mask=mask, labels=labels)
            losses.append(float(out.loss.detach().cpu().item()))

    model.train()
    return float(sum(losses) / max(1, len(losses)))


def run_sft(
    cfg: dict,
    train_jsonl: str,
    val_jsonl: str,
    output_dir: str,
    resume_from: str = "",
    logger=None,
) -> SFTResult:
    model_name = cfg.get("model", {}).get("name", "Qwen/Qwen3-8B")
    trust_remote_code = bool(cfg.get("model", {}).get("trust_remote_code", True))

    fcfg = cfg.get("format", {})
    user_prefix = str(fcfg.get("user_prefix", "User: "))
    assistant_prefix = str(fcfg.get("assistant_prefix", "Assistant: "))

    tcfg = cfg.get("training", {})
    seq_len = int(tcfg.get("seq_len", 2048))
    lr = float(tcfg.get("learning_rate", 1e-5))
    warmup_ratio = float(tcfg.get("warmup_ratio", 0.03))
    weight_decay = float(tcfg.get("weight_decay", 0.01))
    max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
    bf16 = bool(tcfg.get("bf16", True))
    gradient_checkpointing = bool(tcfg.get("gradient_checkpointing", True))
    optimizer_name = str(tcfg.get("optimizer", "adafactor")).lower().strip()
    batch_size = int(tcfg.get("train_batch_size", 1))
    grad_accum = int(tcfg.get("gradient_accumulation_steps", 8))
    epochs = int(tcfg.get("epochs", 3))
    max_steps = int(tcfg.get("max_steps", 0))
    eval_every_steps = int(tcfg.get("eval_every_steps", 500))
    eval_batches = int(tcfg.get("eval_batches", 32))
    eval_batch_size = int(tcfg.get("eval_batch_size", 2))
    save_every_steps = int(tcfg.get("save_every_steps", 1000))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = ""
    if str(resume_from).lower() == "latest":
        last = latest_checkpoint(output_dir)
        checkpoint_path = str(last) if last else ""
    elif resume_from:
        checkpoint_path = str(resume_from)
    else:
        checkpoint_path = ""

    load_from = checkpoint_path if checkpoint_path else model_name

    tokenizer = AutoTokenizer.from_pretrained(load_from, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = torch.bfloat16 if (bf16 and device.type == "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        load_from,
        trust_remote_code=trust_remote_code,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    if optimizer_name == "adafactor":
        optimizer = Adafactor(
            model.parameters(),
            lr=lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    train_summary = summarize_jsonl(train_jsonl)
    val_summary = summarize_jsonl(val_jsonl)
    if train_summary.valid_rows <= 0:
        raise RuntimeError("No valid SFT train examples")
    if val_summary.valid_rows <= 0:
        raise RuntimeError("No valid SFT val examples")

    steps_per_epoch = math.ceil(train_summary.valid_rows / max(1, batch_size * grad_accum))
    total_opt_steps = max(1, epochs * steps_per_epoch)
    if max_steps > 0:
        total_opt_steps = min(total_opt_steps, max_steps)

    warmup_steps = int(total_opt_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps,
    )

    global_step = 0
    supervised_tokens_seen = 0
    if checkpoint_path:
        rs = load_resume_state(checkpoint_path, optimizer=optimizer, scheduler=scheduler)
        global_step = rs.global_step
        supervised_tokens_seen = rs.supervised_tokens_seen

    save_path = ""
    start_t = time.time()
    epoch = 0

    while global_step < total_opt_steps:
        epoch += 1
        stream = iter_tokenized_examples(
            train_jsonl,
            tokenizer=tokenizer,
            seq_len=seq_len,
            user_prefix=user_prefix,
            assistant_prefix=assistant_prefix,
        )

        made_progress = False
        while global_step < total_opt_steps:
            optimizer.zero_grad(set_to_none=True)
            loss_acc = 0.0
            step_sup_tokens = 0
            did_work = False

            for _ in range(grad_accum):
                batch: list[tuple[list[int], list[int], int]] = []
                for _ in range(batch_size):
                    try:
                        batch.append(next(stream))
                    except StopIteration:
                        break
                if not batch:
                    break

                did_work = True
                inp, mask, labels, sup = _collate(batch, pad_id=tokenizer.pad_token_id, device=device)
                step_sup_tokens += sup

                out = model(input_ids=inp, attention_mask=mask, labels=labels)
                loss = out.loss / max(1, grad_accum)
                if torch.isnan(loss).any():
                    raise RuntimeError("NaN loss detected")
                loss.backward()
                loss_acc += float(loss.detach().cpu().item())

            if not did_work:
                break

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1
            supervised_tokens_seen += step_sup_tokens
            made_progress = True

            elapsed = max(1e-6, time.time() - start_t)
            if logger is not None:
                logger.log(
                    {
                        "train/loss": loss_acc,
                        "optim/lr": float(scheduler.get_last_lr()[0]),
                        "system/supervised_tokens_seen": supervised_tokens_seen,
                        "system/supervised_tokens_per_sec": float(supervised_tokens_seen / elapsed),
                        "system/epoch": epoch,
                    },
                    step=global_step,
                )

            if global_step % eval_every_steps == 0:
                vloss = _eval_loss(
                    model=model,
                    val_jsonl=val_jsonl,
                    tokenizer=tokenizer,
                    pad_id=tokenizer.pad_token_id,
                    device=device,
                    batch_size=eval_batch_size,
                    max_batches=eval_batches,
                    seq_len=seq_len,
                    user_prefix=user_prefix,
                    assistant_prefix=assistant_prefix,
                )
                if logger is not None:
                    logger.log(
                        {
                            "val/loss": vloss,
                            "val/perplexity": float(math.exp(min(vloss, 20.0))),
                        },
                        step=global_step,
                    )

            if global_step % save_every_steps == 0:
                ckpt = save_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    supervised_tokens_seen=supervised_tokens_seen,
                )
                save_path = str(ckpt)
                if logger is not None:
                    logger.log({"checkpoint/step": global_step}, step=global_step)

        if not made_progress:
            break

    if not save_path:
        ckpt = save_checkpoint(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            supervised_tokens_seen=supervised_tokens_seen,
        )
        save_path = str(ckpt)

    return SFTResult(
        final_step=global_step,
        supervised_tokens_seen=supervised_tokens_seen,
        final_checkpoint=save_path,
    )
