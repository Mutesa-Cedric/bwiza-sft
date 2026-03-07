#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <prompts_jsonl> <workdir> [model] [max_items]"
  exit 1
fi

PROMPTS="$1"
WORKDIR="$2"
MODEL="${3:-gpt-5.4}"
MAX_ITEMS="${4:-0}"

mkdir -p "$WORKDIR"

RAW="$WORKDIR/generated.raw.jsonl"
CLEAN="$WORKDIR/generated.clean.jsonl"
TRAIN="$WORKDIR/train.jsonl"
VAL="$WORKDIR/val.jsonl"
TEST="$WORKDIR/test.jsonl"

python scripts/generate_sft_openai.py \
  --input_jsonl "$PROMPTS" \
  --output_jsonl "$RAW" \
  --model "$MODEL" \
  --max_items "$MAX_ITEMS"

python scripts/clean_sft_generated.py \
  --input_jsonl "$RAW" \
  --output_jsonl "$CLEAN"

if [[ -f "configs/sft_content_mix_v1.json" ]]; then
  if [[ "${STRICT_MIX:-0}" == "1" ]]; then
    python scripts/validate_sft_content_mix.py \
      --input_jsonl "$CLEAN" \
      --plan_json configs/sft_content_mix_v1.json \
      --output "$WORKDIR/sft_content_mix_report.json"
  else
    python scripts/validate_sft_content_mix.py \
      --input_jsonl "$CLEAN" \
      --plan_json configs/sft_content_mix_v1.json \
      --output "$WORKDIR/sft_content_mix_report.json" || true
  fi
fi

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
