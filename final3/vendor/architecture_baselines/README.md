# Architecture Baseline Trainers

This directory contains the editable final3 copies of architecture-baseline code:

- `train_direct_dual_head_mlp.py`
- `bert_baselines/`
- `llm_logit_baselines/`
- `qwen_finetune_baselines/`

Direct MLP is complete on the current ICSE 16-fold split and its selected-policy CSV/JSON outputs are copied under `paper/experiments/rq_final_lightgbm_17/model_compare/direct_mlp_full16_ij`. It is included in `paper/icse_submission_draft/data/table_architecture_compare_full16.csv`.

Local LLM Logit Judge / Qwen LoRA LOMO is complete on the current 16-target split. Its copied sweep outputs live under `paper/experiments/rq_final_lightgbm_17/model_compare/llm_logit_lomo_full16` and feed `table_architecture_compare_full16.csv` plus `table_llm_logit_lomo_*.csv`.

CodeBERT finetune is present only as historical partial files and is excluded from the current paper scope. Use the scripts in `scripts/run_rq_final_09_*.sh` only for future reruns or new baselines.

The old capability-group generated results are intentionally not part of final3 current paper evidence.
