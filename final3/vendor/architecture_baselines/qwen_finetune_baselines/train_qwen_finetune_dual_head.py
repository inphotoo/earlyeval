#!/usr/bin/env python3
"""Fine-tune a Qwen encoder (optionally LoRA) with a dual-head safe-stop objective.

This mirrors the CodeBERT Stage-B pipeline:
- consumes the same compact text cache (`bert_text_cache.parquet`)
- trains `safe_success` / `safe_failure` jointly
- calibrates + selects policies on valid only
- evaluates locked selected policies on test
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

THIS_DIR = Path(__file__).resolve().parent
MODEL_ARCH_ROOT = THIS_DIR.parent
if str(MODEL_ARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ARCH_ROOT))

from bert_baselines.bert_baseline_common import (
    DEFAULT_RESULTS_ROOT,
    DualHeadMLP,
    normalize_weights,
    resolve_device,
    safe_stop_outputs,
    sample_train_indices,
    set_threads,
    weighted_dual_loss,
    write_json,
)

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEFAULT_CACHE = DEFAULT_RESULTS_ROOT / "qwen_finetune_smoke_cache"
DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "qwen_finetune_smoke"


class QwenTextDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, indices: np.ndarray | None = None):
        self.text_a = frame["text_a"].fillna("").astype(str).to_numpy()
        self.text_b = frame["text_b"].fillna("").astype(str).to_numpy()
        self.success = frame["safe_success_label"].to_numpy(dtype=np.float32)
        self.failure = frame["safe_failure_label"].to_numpy(dtype=np.float32)
        self.weights = normalize_weights(frame["sample_weight"].to_numpy(dtype=np.float32))
        self.indices = indices if indices is not None else np.arange(len(frame), dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        return (
            self.text_a[idx],
            self.text_b[idx],
            self.success[idx],
            self.failure[idx],
            self.weights[idx],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--encoder-name", default=None)
    parser.add_argument("--predictor-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--truncation-strategy",
        choices=("longest_first", "only_first", "only_second"),
        default="only_first",
        help="How to truncate text_a/text_b pair when token length exceeds --max-length.",
    )
    parser.add_argument(
        "--truncation-side",
        choices=("left", "right"),
        default="right",
        help="Tokenizer truncation side. right keeps prefix; left keeps suffix.",
    )
    parser.add_argument("--pooling", choices=("last", "mean"), default=None)
    parser.add_argument("--hidden", nargs="+", type=int, default=[256])
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--max-cpu-threads", type=int, default=16)
    parser.add_argument("--policy-min-steps", nargs="+", type=int, default=[0, 5, 10, 15])
    parser.add_argument("--consecutive", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--success-thresholds", nargs="+", type=float, default=[0.80, 0.90, 0.95, float("inf")])
    parser.add_argument("--failure-thresholds", nargs="+", type=float, default=[0.80, 0.90, 0.95, float("inf")])
    parser.add_argument("--score-modes", nargs="+", choices=("raw", "calibrated"), default=["raw", "calibrated"])
    parser.add_argument("--max-valid-abs-drop-pp", type=float, default=2.0)
    parser.add_argument("--min-valid-decision-acc", type=float, default=0.90)
    parser.add_argument("--fallback-min-save-pct", type=float, default=5.0)
    return parser.parse_args()


def _load_cache_meta(cache_dir: Path) -> dict[str, Any]:
    path = cache_dir / "bert_cache_meta.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing cache metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_text_cache(cache_dir: Path) -> dict[str, pd.DataFrame]:
    path = cache_dir / "bert_text_cache.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing text cache: {path}")
    text_cache = pd.read_parquet(path)
    frames = {}
    for split_name in ("train", "valid", "test"):
        frame = text_cache[text_cache["split"] == split_name].sort_values("row_idx").reset_index(drop=True)
        if frame.empty:
            raise RuntimeError(f"Text cache has no rows for split={split_name}")
        frames[split_name] = frame
    return frames


def _predictor_name(encoder_name: str, use_lora: bool) -> str:
    lowered = encoder_name.lower()
    if "qwen" in lowered:
        base = "Qwen"
    elif "llama" in lowered:
        base = "Llama"
    elif "mistral" in lowered:
        base = "Mistral"
    else:
        base = "CausalLM"
    lora_tag = "_LoRA" if use_lora else ""
    return f"{base}{lora_tag}_FineTuneDualHead"


def _collate(tokenizer: Any, max_length: int, truncation_strategy: str):
    def inner(batch):
        text_a, text_b, success, failure, weights = zip(*batch)
        encoded = tokenizer(
            list(text_a),
            list(text_b),
            padding=True,
            truncation=truncation_strategy,
            max_length=max_length,
            return_tensors="pt",
        )
        return (
            encoded,
            torch.tensor(success, dtype=torch.float32),
            torch.tensor(failure, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
        )

    return inner


def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
    lengths = lengths.to(dtype=torch.long)
    gather_idx = lengths.view(-1, 1, 1).expand(-1, 1, last_hidden_state.size(-1))
    pooled = torch.gather(last_hidden_state, dim=1, index=gather_idx)
    return pooled.squeeze(1)


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    pooled = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return pooled / denom


class CausalDualHeadClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        *,
        encoder_dim: int,
        hidden_dims: list[int],
        dropout: float,
        pooling: str,
    ):
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.head = DualHeadMLP(encoder_dim, hidden_dims, dropout)

    def _encode(self, batch: dict[str, torch.Tensor]) -> Any:
        kwargs: dict[str, Any] = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_dict": True,
        }
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        try:
            return self.encoder(**kwargs)
        except TypeError:
            kwargs.pop("token_type_ids", None)
            return self.encoder(**kwargs)

    def forward(self, **batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self._encode(batch)
        hidden = outputs.last_hidden_state
        if self.pooling == "mean":
            pooled = _mean_pool(hidden, batch["attention_mask"])
        else:
            pooled = _last_token_pool(hidden, batch["attention_mask"])
        return self.head(pooled)


def _autocast_context(device: torch.device, amp_dtype: torch.dtype | None):
    if device.type == "cuda" and amp_dtype is not None:
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def _loss_on_loader(
    model: CausalDualHeadClassifier,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    with torch.inference_mode():
        for encoded, success, failure, weights in loader:
            encoded = {key: value.to(device) for key, value in encoded.items()}
            success = success.to(device)
            failure = failure.to(device)
            weights = weights.to(device)
            with _autocast_context(device, amp_dtype):
                success_logits, failure_logits = model(**encoded)
                loss = weighted_dual_loss(success_logits, failure_logits, success, failure, weights)
            batch_weight = float(weights.sum().detach().cpu())
            total_loss += float(loss.detach().cpu()) * max(batch_weight, 1.0)
            total_weight += max(batch_weight, 1.0)
    return total_loss / max(total_weight, 1.0)


def _predict_probs(
    model: CausalDualHeadClassifier,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    success_probs: list[np.ndarray] = []
    failure_probs: list[np.ndarray] = []
    with torch.inference_mode():
        for encoded, _, _, _ in loader:
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with _autocast_context(device, amp_dtype):
                success_logits, failure_logits = model(**encoded)
            success_probs.append(torch.sigmoid(success_logits).detach().float().cpu().numpy())
            failure_probs.append(torch.sigmoid(failure_logits).detach().float().cpu().numpy())
    return np.concatenate(success_probs), np.concatenate(failure_probs)


def _trainable_parameters(model: CausalDualHeadClassifier) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    encoder_params = [param for param in model.encoder.parameters() if param.requires_grad]
    head_params = [param for param in model.head.parameters() if param.requires_grad]
    return encoder_params, head_params


def main() -> int:
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("Choose either --bf16 or --fp16, not both.")

    set_threads(args.max_cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    cache_meta = _load_cache_meta(args.cache_dir)

    encoder_name = args.encoder_name or cache_meta.get("encoder_name") or DEFAULT_MODEL
    max_length = int(args.max_length or cache_meta.get("max_length", 2048))
    pooling = str(args.pooling or "last")
    predictor_name = args.predictor_name or _predictor_name(str(encoder_name), bool(args.use_lora))

    output_dir = args.output_dir
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        encoder_name,
        local_files_only=not args.allow_download,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = str(args.truncation_side)

    model_kwargs: dict[str, Any] = {
        "local_files_only": not args.allow_download,
        "trust_remote_code": args.trust_remote_code,
    }
    if device.type == "cuda":
        if args.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif args.fp16:
            model_kwargs["torch_dtype"] = torch.float16

    encoder = AutoModel.from_pretrained(encoder_name, **model_kwargs)
    if args.gradient_checkpointing and hasattr(encoder, "gradient_checkpointing_enable"):
        encoder.gradient_checkpointing_enable()
    if hasattr(encoder, "config") and hasattr(encoder.config, "use_cache"):
        encoder.config.use_cache = False

    lora_active = False
    if args.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError(
                "--use-lora requires `peft`. Install it or rerun with --no-lora."
            ) from exc
        lora_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            target_modules=list(args.lora_target_modules),
            bias="none",
        )
        encoder = get_peft_model(encoder, lora_cfg)
        lora_active = True

    encoder.to(device)
    encoder_dim = int(getattr(encoder.config, "hidden_size", 0))
    if encoder_dim <= 0:
        raise RuntimeError(f"Cannot infer hidden size from encoder: {encoder_name}")

    model = CausalDualHeadClassifier(
        encoder,
        encoder_dim=encoder_dim,
        hidden_dims=args.hidden,
        dropout=args.dropout,
        pooling=pooling,
    ).to(device)

    frames = _load_text_cache(args.cache_dir)
    train_indices = sample_train_indices(
        frames["train"]["sample_weight"].to_numpy(dtype=np.float32),
        args.max_train_rows,
        args.seed,
    )
    datasets = {
        "train": QwenTextDataset(frames["train"], train_indices),
        "valid": QwenTextDataset(frames["valid"]),
        "test": QwenTextDataset(frames["test"]),
    }
    collate_fn = _collate(tokenizer, max_length, args.truncation_strategy)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
        ),
    }

    encoder_params, head_params = _trainable_parameters(model)
    param_groups: list[dict[str, Any]] = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": args.lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.head_lr})
    if not param_groups:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    amp_dtype: torch.dtype | None = None
    if device.type == "cuda":
        if args.bf16:
            amp_dtype = torch.bfloat16
        elif args.fp16:
            amp_dtype = torch.float16
    use_grad_scaler = amp_dtype == torch.float16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    grad_accum = max(1, int(args.gradient_accumulation_steps))
    best_valid = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = models_dir / f"{predictor_name}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_weight = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, (encoded, success, failure, weights) in enumerate(loaders["train"], start=1):
            encoded = {key: value.to(device) for key, value in encoded.items()}
            success = success.to(device)
            failure = failure.to(device)
            weights = weights.to(device)

            with _autocast_context(device, amp_dtype):
                success_logits, failure_logits = model(**encoded)
                loss = weighted_dual_loss(success_logits, failure_logits, success, failure, weights)
                loss = loss / grad_accum

            if use_grad_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % grad_accum == 0 or step == len(loaders["train"]):
                if use_grad_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            batch_weight = float(weights.sum().detach().cpu())
            train_loss += float(loss.detach().cpu()) * grad_accum * max(batch_weight, 1.0)
            train_weight += max(batch_weight, 1.0)

        train_loss = train_loss / max(train_weight, 1.0)
        valid_loss = _loss_on_loader(model, loaders["valid"], device, amp_dtype)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})

        if valid_loss < best_valid:
            best_valid = valid_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "encoder_name": encoder_name,
                    "encoder_dim": encoder_dim,
                    "hidden": args.hidden,
                    "dropout": args.dropout,
                    "pooling": pooling,
                    "predictor_name": predictor_name,
                    "cache_dir": str(args.cache_dir),
                    "lora_active": bool(lora_active),
                    "lora_target_modules": list(args.lora_target_modules),
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    valid_success_raw, valid_failure_raw = _predict_probs(model, loaders["valid"], device, amp_dtype)
    test_success_raw, test_failure_raw = _predict_probs(model, loaders["test"], device, amp_dtype)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    all_params = int(sum(p.numel() for p in model.parameters()))

    pd.DataFrame(
        [
            {
                "predictor": predictor_name,
                "encoder_name": encoder_name,
                "stage": "qwen_finetune_dual_head",
                "max_length": int(max_length),
                "pooling": pooling,
                "truncation_strategy": args.truncation_strategy,
                "truncation_side": args.truncation_side,
                "train_rows": int(len(datasets["train"])),
                "valid_rows": int(len(datasets["valid"])),
                "test_rows": int(len(datasets["test"])),
                "best_epoch": int(best_epoch),
                "best_valid_loss": float(best_valid),
                "lora_active": bool(lora_active),
                "trainable_params": trainable_params,
                "all_params": all_params,
                "trainable_ratio": float(trainable_params / max(all_params, 1)),
            }
        ]
    ).to_csv(output_dir / "variant_manifest.csv", index=False)

    write_json(
        output_dir / "qwen_finetune_config.json",
        {
            "stage": "qwen_finetune_dual_head",
            "predictor_name": predictor_name,
            "cache_dir": str(args.cache_dir),
            "cache_meta": cache_meta,
            "encoder_name": encoder_name,
            "max_length": int(max_length),
            "pooling": pooling,
            "truncation_strategy": args.truncation_strategy,
            "truncation_side": args.truncation_side,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "epochs_requested": int(args.epochs),
            "best_epoch": int(best_epoch),
            "batch_size": int(args.batch_size),
            "eval_batch_size": int(args.eval_batch_size),
            "gradient_accumulation_steps": int(grad_accum),
            "lr": float(args.lr),
            "head_lr": float(args.head_lr),
            "weight_decay": float(args.weight_decay),
            "max_train_rows": int(args.max_train_rows),
            "device": str(device),
            "bf16": bool(args.bf16),
            "fp16": bool(args.fp16),
            "gradient_checkpointing": bool(args.gradient_checkpointing),
            "lora_active": bool(lora_active),
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "lora_target_modules": list(args.lora_target_modules),
            "trainable_params": trainable_params,
            "all_params": all_params,
        },
    )

    safe_stop_outputs(
        output_dir=output_dir,
        run_label=output_dir.name,
        predictor_name=predictor_name,
        valid_frame=frames["valid"],
        test_frame=frames["test"],
        valid_success_raw=valid_success_raw,
        valid_failure_raw=valid_failure_raw,
        test_success_raw=test_success_raw,
        test_failure_raw=test_failure_raw,
        score_modes=args.score_modes,
        success_thresholds=args.success_thresholds,
        failure_thresholds=args.failure_thresholds,
        policy_min_steps=args.policy_min_steps,
        consecutive=args.consecutive,
        max_valid_abs_drop_pp=args.max_valid_abs_drop_pp,
        min_valid_decision_acc=args.min_valid_decision_acc,
        fallback_min_save_pct=args.fallback_min_save_pct,
    )

    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
