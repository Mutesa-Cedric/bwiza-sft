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
