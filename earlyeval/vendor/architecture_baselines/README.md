# Architecture Baseline Trainers

This directory contains the editable earlyeval copies of architecture-baseline code:

- `train_direct_dual_head_mlp.py`
- `bert_baselines/`
- `llm_logit_baselines/`
- `qwen_finetune_baselines/`

Direct MLP is complete on the current SWE-bench Verified paper split and its selected-policy CSV/JSON outputs are copied under `paper/experiments/earlyeval_lightgbm/model_compare/direct_mlp_sweverify_ij`. It is included in `paper/icse_submission_draft/data/table_architecture_compare_sweverify.csv`.

Local LLM Logit Judge / Qwen LoRA LOMO is complete on the current SWE-bench Verified paper split. Its copied sweep outputs live under `paper/experiments/earlyeval_lightgbm/model_compare/llm_logit_lomo_sweverify` and feed `table_architecture_compare_sweverify.csv` plus `table_llm_logit_lomo_*.csv`.

CodeBERT finetune is present only as historical partial files and is excluded from the current paper scope. Use the scripts in `scripts/run_earlyeval_09_*.sh` only for future reruns or new baselines.

The old capability-group generated results are intentionally not part of earlyeval current paper evidence.
