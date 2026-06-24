# models

`models/` contains model-training command builders and heavy experiment
declarations. Entry points are dry-run by default unless `--execute` is passed.

## Files

- `dual_head_lightgbm.py`: builds the current LightGBM dual-head training
  command through the vendored trainer.
- `heavy.py`: lists opt-in heavy experiments such as BERT/CodeBERT, Qwen LoRA,
  LLM-logit judges, and full ablations.

## Examples

```bash
python -m final3.cli train dual-head
python -m final3.cli train dual-head --execute
python -m final3.cli train list-heavy
```

Do not load model weights or start training at import time.
