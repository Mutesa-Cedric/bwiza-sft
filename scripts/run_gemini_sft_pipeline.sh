#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <prompts_jsonl|auto|gemini> <workdir> [model] [max_items]"
  exit 1
fi

PROMPTS="$1"
WORKDIR="$2"
MODEL="${3:-gemini-3.1-pro-preview}"
MAX_ITEMS="${4:-0}"

mkdir -p "$WORKDIR"

RAW="$WORKDIR/generated.raw.jsonl"
CLEAN="$WORKDIR/generated.clean.jsonl"
TRAIN="$WORKDIR/train.jsonl"
VAL="$WORKDIR/val.jsonl"
TEST="$WORKDIR/test.jsonl"

if [[ "$PROMPTS" == "auto" ]]; then
  PROMPTS="$WORKDIR/prompts.seed.jsonl"
  python scripts/build_prompt_seed.py \
    --output_jsonl "$PROMPTS" \
    --target 500 \
    --overwrite
fi

if [[ "$PROMPTS" == "gemini" ]]; then
  PROMPTS="$WORKDIR/prompts.seed.gemini.jsonl"
  python scripts/build_prompt_seed_gemini.py \
    --output_jsonl "$PROMPTS" \
    --target 500
fi

python scripts/generate_sft_gemini.py \
  --input_jsonl "$PROMPTS" \
  --output_jsonl "$RAW" \
  --model "$MODEL" \
  --max_items "$MAX_ITEMS"

python scripts/clean_sft_generated.py \
  --input_jsonl "$RAW" \
  --output_jsonl "$CLEAN"

python scripts/split_sft_dataset.py \
  --input_jsonl "$CLEAN" \
  --train_jsonl "$TRAIN" \
  --val_jsonl "$VAL" \
  --test_jsonl "$TEST"

python scripts/preflight_sft_data.py \
  --train_jsonl "$TRAIN" \
  --val_jsonl "$VAL" \
  --test_jsonl "$TEST" \
  --output "$WORKDIR/preflight_sft_data.json"

printf "\nPipeline complete:\n  %s\n  %s\n  %s\n" "$TRAIN" "$VAL" "$TEST"
