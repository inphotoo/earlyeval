HEAVY_EXPERIMENTS = {
    "current_safe_stop_retrain": "Current final-split LightGBM dual-head retraining",
    "direct_mlp": "Direct dual-head MLP architecture baseline",
    "bert_codebert": "BERT/CodeBERT frozen or finetune baselines",
    "qwen_lora": "Qwen LoRA finetune dual-head baseline",
    "llm_logit_judge": "Local LLM Yes/No logit judge",
    "full_ablation_sweep": "Full task/gold/process ablation sweep",
}


def describe_heavy_experiments() -> dict[str, str]:
    return dict(HEAVY_EXPERIMENTS)
