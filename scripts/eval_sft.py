#!/usr/bin/env python3
"""Compare base vs SFT-adapted model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SFT base vs adapted")
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapted_model", required=True)
    p.add_argument("--val_jsonl", required=True)
    p.add_argument("--test_jsonl", required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--max_eval_batches", type=int, default=32)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--output", default="outputs/reports/eval_sft.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from src.eval.sft_eval import evaluate_model, write_report
    except ModuleNotFoundError as e:
        raise RuntimeError("Missing eval dependency. Install project dependencies first (pip install -e .).") from e

    base = evaluate_model(
        model_path=args.base_model,
        val_jsonl=args.val_jsonl,
        test_jsonl=args.test_jsonl,
        device=args.device,
        max_eval_batches=args.max_eval_batches,
        eval_batch_size=args.eval_batch_size,
        seq_len=args.seq_len,
    )
    adapted = evaluate_model(
        model_path=args.adapted_model,
        val_jsonl=args.val_jsonl,
        test_jsonl=args.test_jsonl,
        device=args.device,
        max_eval_batches=args.max_eval_batches,
        eval_batch_size=args.eval_batch_size,
        seq_len=args.seq_len,
    )

    delta = {
        "val_loss_delta": adapted.val.loss - base.val.loss,
        "val_ppl_delta": adapted.val.perplexity - base.val.perplexity,
        "test_loss_delta": adapted.test.loss - base.test.loss,
        "test_ppl_delta": adapted.test.perplexity - base.test.perplexity,
        "english_drift_delta": adapted.generation.avg_english_drift - base.generation.avg_english_drift,
        "rw_marker_density_delta": adapted.generation.avg_rw_marker_density - base.generation.avg_rw_marker_density,
    }

    payload = {
        "base": base.to_dict(),
        "adapted": adapted.to_dict(),
        "delta": delta,
        "promotion_signal": {
            "better_val_ppl": delta["val_ppl_delta"] < 0,
            "better_test_ppl": delta["test_ppl_delta"] < 0,
            "lower_english_drift": delta["english_drift_delta"] < 0,
        },
    }

    write_report(args.output, payload)
    print(f"Wrote evaluation report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
