#!/usr/bin/env python3
"""Entrypoint for SFT runs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.fingerprint import dataset_fingerprint
from src.train.manifest import RunManifest
from src.train.wandb_logger import WandbLogger


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run SFT")
    p.add_argument("--config", required=True)
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--val_jsonl", required=True)
    p.add_argument("--test_jsonl", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume_from", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from src.train.sft_loop import run_sft
    except ModuleNotFoundError as e:
        raise RuntimeError("Missing dependency. Install project dependencies first (pip install -e .).") from e

    config_path = Path(args.config)
    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    out_dir = Path(args.output_dir)

    for p in [config_path, train_path, val_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    if args.test_jsonl and not Path(args.test_jsonl).exists():
        raise FileNotFoundError(f"Missing test jsonl: {args.test_jsonl}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg_fp = hashlib.sha256(config_path.read_bytes()).hexdigest()
    ds_fp = dataset_fingerprint(args.train_jsonl, args.val_jsonl, args.test_jsonl)

    run_name = cfg.get("run", {}).get("name", "bwiza-sft")
    model_name = cfg.get("model", {}).get("name", "Qwen/Qwen3-8B")

    manifest = RunManifest(
        run_id=hashlib.sha1(f"{run_name}:{cfg_fp}:{ds_fp}".encode("utf-8")).hexdigest()[:12],
        run_name=run_name,
        git_commit=_git_commit(),
        config_fingerprint=cfg_fp,
        dataset_fingerprint=ds_fp,
        tokenizer_name=model_name,
        model_name=model_name,
        extra={
            "train_jsonl": str(train_path),
            "val_jsonl": str(val_path),
            "test_jsonl": str(args.test_jsonl),
            "resume_from": args.resume_from,
        },
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.json"
    manifest.write_json(manifest_path)

    wcfg = cfg.get("wandb", {})
    logger = WandbLogger(
        enabled=bool(wcfg.get("enabled", True)),
        project=str(wcfg.get("project", "bwiza-sft")),
        run_name=run_name,
        run_group=str(wcfg.get("run_group", "qwen3-8b-rw-v1")),
        mode=str(wcfg.get("mode", "online")),
        output_dir=str(out_dir),
        config=cfg,
    )

    status = "failed"
    try:
        result = run_sft(
            cfg=cfg,
            train_jsonl=str(train_path),
            val_jsonl=str(val_path),
            output_dir=str(out_dir),
            resume_from=args.resume_from,
            logger=logger,
        )
        manifest.checkpoints.append(result.final_checkpoint)
        manifest.finish(
            status="completed",
            supervised_tokens_seen=result.supervised_tokens_seen,
            global_step=result.final_step,
        )
        status = "completed"
        print(f"SFT completed: step={result.final_step} supervised_tokens={result.supervised_tokens_seen}")
        print(f"Final checkpoint: {result.final_checkpoint}")
    except Exception:
        manifest.finish(status="failed", supervised_tokens_seen=manifest.supervised_tokens_seen, global_step=manifest.global_step)
        raise
    finally:
        manifest.write_json(manifest_path)
        logger.finish()

    print(f"Run status: {status}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
