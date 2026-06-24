#!/usr/bin/env python3
"""Fine-tune a BERT/CodeBERT dual-head safe-stop baseline.

This is the heavier Stage-B baseline.  It consumes the compact text cache
produced by ``build_bert_embedding_cache.py`` and keeps the same valid-only
policy selection discipline as the LightGBM/MLP runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from bert_baseline_common import (
    DEFAULT_ENCODER,
    DEFAULT_RESULTS_ROOT,
    BertDualHeadClassifier,
    normalize_weights,
    resolve_device,
    safe_stop_outputs,
    sample_train_indices,
    set_threads,
    weighted_dual_loss,
    write_json,
)


DEFAULT_CACHE = DEFAULT_RESULTS_ROOT / "bert_codebert_cache"
DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "bert_codebert_finetune_smoke"


class BertTextDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        indices: np.ndarray | None = None,
    ):
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
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--pooling", choices=("mean", "cls"), default=None)
    parser.add_argument("--hidden", nargs="+", type=int, default=[256])
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
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


def _predictor_name(encoder_name: str) -> str:
    lowered = encoder_name.lower()
    if "codebert" in lowered:
        base = "CodeBERT"
    elif "unixcoder" in lowered:
        base = "UniXcoder"
    else:
        base = "BERT"
    return f"{base}_FineTuneDualHead"


def _collate(tokenizer: Any, max_length: int):
    def inner(batch):
        text_a, text_b, success, failure, weights = zip(*batch)
        encoded = tokenizer(
            list(text_a),
            list(text_b),
            padding=True,
            truncation=True,
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


def _loss_on_loader(
    model: BertDualHeadClassifier,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
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
            with torch.cuda.amp.autocast(enabled=use_amp):
                success_logits, failure_logits = model(**encoded)
                loss = weighted_dual_loss(success_logits, failure_logits, success, failure, weights)
            batch_weight = float(weights.sum().detach().cpu())
            total_loss += float(loss.detach().cpu()) * max(batch_weight, 1.0)
            total_weight += max(batch_weight, 1.0)
    return total_loss / max(total_weight, 1.0)


def _predict_probs(
    model: BertDualHeadClassifier,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    success_probs: list[np.ndarray] = []
    failure_probs: list[np.ndarray] = []
    with torch.inference_mode():
        for encoded, _, _, _ in loader:
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                success_logits, failure_logits = model(**encoded)
            success_probs.append(torch.sigmoid(success_logits).detach().float().cpu().numpy())
            failure_probs.append(torch.sigmoid(failure_logits).detach().float().cpu().numpy())
    return np.concatenate(success_probs), np.concatenate(failure_probs)


def main() -> int:
    args = parse_args()
    set_threads(args.max_cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    use_amp = bool(args.fp16 and device.type == "cuda")

    cache_meta = _load_cache_meta(args.cache_dir)
    encoder_name = args.encoder_name or cache_meta.get("encoder_name") or DEFAULT_ENCODER
    max_length = int(args.max_length or cache_meta.get("max_length", 512))
    pooling = str(args.pooling or cache_meta.get("pooling", "mean"))
    predictor_name = args.predictor_name or _predictor_name(str(encoder_name))
    output_dir = args.output_dir
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        encoder_name,
        local_files_only=not args.allow_download,
        use_fast=True,
    )
    encoder = AutoModel.from_pretrained(
        encoder_name,
        local_files_only=not args.allow_download,
    )
    encoder.to(device)
    encoder_dim = int(getattr(encoder.config, "hidden_size", 0))
    if encoder_dim <= 0:
        raise RuntimeError(f"Cannot infer hidden size from encoder: {encoder_name}")
    model = BertDualHeadClassifier(
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
        "train": BertTextDataset(frames["train"], train_indices),
        "valid": BertTextDataset(frames["valid"]),
        "test": BertTextDataset(frames["test"]),
    }
    collate_fn = _collate(tokenizer, max_length)
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

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
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
            with torch.cuda.amp.autocast(enabled=use_amp):
                success_logits, failure_logits = model(**encoded)
                loss = weighted_dual_loss(success_logits, failure_logits, success, failure, weights)
                loss = loss / grad_accum
            scaler.scale(loss).backward()
            if step % grad_accum == 0 or step == len(loaders["train"]):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            batch_weight = float(weights.sum().detach().cpu())
            train_loss += float(loss.detach().cpu()) * grad_accum * max(batch_weight, 1.0)
            train_weight += max(batch_weight, 1.0)
        train_loss = train_loss / max(train_weight, 1.0)
        valid_loss = _loss_on_loader(model, loaders["valid"], device, use_amp)
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
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    valid_success_raw, valid_failure_raw = _predict_probs(model, loaders["valid"], device, use_amp)
    test_success_raw, test_failure_raw = _predict_probs(model, loaders["test"], device, use_amp)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "predictor": predictor_name,
                "encoder_name": encoder_name,
                "stage": "finetune_encoder_dual_head",
                "max_length": int(max_length),
                "pooling": pooling,
                "train_rows": int(len(datasets["train"])),
                "valid_rows": int(len(datasets["valid"])),
                "test_rows": int(len(datasets["test"])),
                "best_epoch": int(best_epoch),
                "best_valid_loss": float(best_valid),
            }
        ]
    ).to_csv(output_dir / "variant_manifest.csv", index=False)
    write_json(
        output_dir / "bert_config.json",
        {
            "stage": "finetune_encoder_dual_head",
            "predictor_name": predictor_name,
            "cache_dir": str(args.cache_dir),
            "cache_meta": cache_meta,
            "encoder_name": encoder_name,
            "max_length": int(max_length),
            "pooling": pooling,
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
            "fp16": bool(use_amp),
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
