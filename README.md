# bwiza-sft

SFT training pipeline for `bwiza-1-instruct`.

## Expected SFT JSONL schema
Each line must include:
- `prompt` (string)
- `response` (string)

Optional fields are ignored by training.

## Commands

```bash
python scripts/preflight_sft_data.py \
  --train_jsonl /path/train.jsonl \
  --val_jsonl /path/val.jsonl \
  --test_jsonl /path/test.jsonl \
  --output outputs/reports/preflight_sft_data.json

python scripts/train_sft.py \
  --config configs/sft_qwen3_8b_sanity.yaml \
  --train_jsonl /path/train.jsonl \
  --val_jsonl /path/val.jsonl \
  --test_jsonl /path/test.jsonl \
  --output_dir outputs/runs/sanity

python scripts/eval_sft.py \
  --base_model /path/base_or_hf_model \
  --adapted_model /path/sft_checkpoint \
  --val_jsonl /path/val.jsonl \
  --test_jsonl /path/test.jsonl \
  --output outputs/reports/eval_sft.json

python scripts/eval_sft_gate.py \
  --eval_report outputs/reports/eval_sft.json \
  --output outputs/reports/eval_sft_gate.json \
  --require_better_val_ppl \
  --require_better_test_ppl \
  --max_english_drift_delta 0.0 \
  --min_rw_marker_density_delta 0.0
```

## Gemini Distillation Data Pipeline

Generate SFT responses from prompt JSONL, then clean and split.

```bash
export GEMINI_API_KEY='...'

python scripts/generate_sft_gemini.py \
  --input_jsonl /path/prompts.jsonl \
  --output_jsonl outputs/sft/generated.raw.jsonl \
  --model gemini-3.1-pro-preview

python scripts/clean_sft_generated.py \
  --input_jsonl outputs/sft/generated.raw.jsonl \
  --output_jsonl outputs/sft/generated.clean.jsonl

python scripts/split_sft_dataset.py \
  --input_jsonl outputs/sft/generated.clean.jsonl \
  --train_jsonl outputs/sft/train.jsonl \
  --val_jsonl outputs/sft/val.jsonl \
  --test_jsonl outputs/sft/test.jsonl
```

Or one command:

```bash
scripts/run_gemini_sft_pipeline.sh /path/prompts.jsonl outputs/sft gemini-3.1-pro-preview
```

Pilot first (recommended):

```bash
python scripts/generate_sft_gemini.py \
  --input_jsonl /path/prompts.jsonl \
  --output_jsonl outputs/sft/pilot.raw.jsonl \
  --model gemini-3.1-pro-preview \
  --max_items 100
```
