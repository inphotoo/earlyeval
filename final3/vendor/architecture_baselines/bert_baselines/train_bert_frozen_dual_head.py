#!/usr/bin/env python3
"""Train a frozen-encoder BERT/CodeBERT dual-head safe-stop baseline."""

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


DEFAULT_CACHE = DEFAULT_RESULTS_ROOT / "bert_codebert_cache"
DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "bert_codebert_frozen"


class ArrayDualHeadDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        success: np.ndarray,
        failure: np.ndarray,
        weights: np.ndarray,
        indices: np.ndarray | None = None,
    ):
        self.features = features
        self.success = success.astype(np.float32)
        self.failure = failure.astype(np.float32)
        self.weights = normalize_weights(weights)
        self.indices = indices if indices is not None else np.arange(len(features), dtype=np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, item: int):
        idx = int(self.indices[item])
        return (
            self.features[idx].astype(np.float32, copy=False),
            self.success[idx],
            self.failure[idx],
            self.weights[idx],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--predictor-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--include-dense", action="store_true", help="Append dense arrays saved by the cache builder.")
    parser.add_argument("--hidden", nargs="+", type=int, default=[256])
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
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


def _predictor_name(cache_meta: dict[str, Any], include_dense: bool) -> str:
    encoder = str(cache_meta.get("encoder_name", "bert")).lower()
    if "codebert" in encoder:
        base = "CodeBERT"
    elif "unixcoder" in encoder:
        base = "UniXcoder"
    else:
        base = "BERT"
    suffix = "_Dense" if include_dense else ""
    return f"{base}_FrozenDualHead{suffix}"


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
        expected = np.arange(len(frame), dtype=np.int64)
        observed = frame["row_idx"].to_numpy(dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise RuntimeError(f"Non-contiguous row_idx for split={split_name}")
        frames[split_name] = frame
    return frames


def _load_features(cache_dir: Path, split_name: str, include_dense: bool) -> np.ndarray:
    emb_path = cache_dir / f"bert_embeddings_{split_name}.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embedding array: {emb_path}")
    embeddings = np.load(emb_path).astype(np.float32)
    if include_dense:
        dense_path = cache_dir / f"bert_dense_{split_name}.npy"
        if not dense_path.exists():
            raise FileNotFoundError(f"Missing dense array for --include-dense: {dense_path}")
        dense = np.load(dense_path).astype(np.float32)
        if len(dense) != len(embeddings):
            raise RuntimeError(f"Dense/embedding row mismatch for split={split_name}")
        return np.concatenate([embeddings, dense], axis=1)
    return embeddings


def _loss_on_loader(
    model: DualHeadMLP,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    with torch.inference_mode():
        for features, success, failure, weights in loader:
            features = features.to(device)
            success = success.to(device)
            failure = failure.to(device)
            weights = weights.to(device)
            success_logits, failure_logits = model(features)
            loss = weighted_dual_loss(success_logits, failure_logits, success, failure, weights)
            batch_weight = float(weights.sum().detach().cpu())
            total_loss += float(loss.detach().cpu()) * max(batch_weight, 1.0)
            total_weight += max(batch_weight, 1.0)
    return total_loss / max(total_weight, 1.0)


def _predict_probs(
    model: DualHeadMLP,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    success_probs: list[np.ndarray] = []
    failure_probs: list[np.ndarray] = []
    with torch.inference_mode():
        for features, _, _, _ in loader:
            features = features.to(device)
            success_logits, failure_logits = model(features)
            success_probs.append(torch.sigmoid(success_logits).detach().cpu().numpy())
            failure_probs.append(torch.sigmoid(failure_logits).detach().cpu().numpy())
    return np.concatenate(success_probs), np.concatenate(failure_probs)


def main() -> int:
    args = parse_args()
    set_threads(args.max_cpu_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.device)
    cache_meta = _load_cache_meta(args.cache_dir)
    predictor_name = args.predictor_name or _predictor_name(cache_meta, args.include_dense)
    output_dir = args.output_dir
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    frames = _load_text_cache(args.cache_dir)
    features = {
        split_name: _load_features(args.cache_dir, split_name, args.include_dense)
        for split_name in ("train", "valid", "test")
    }
    labels = {
        split_name: {
            "success": frames[split_name]["safe_success_label"].to_numpy(dtype=np.float32),
            "failure": frames[split_name]["safe_failure_label"].to_numpy(dtype=np.float32),
            "weights": frames[split_name]["sample_weight"].to_numpy(dtype=np.float32),
        }
        for split_name in ("train", "valid", "test")
    }
    train_indices = sample_train_indices(labels["train"]["weights"], args.max_train_rows, args.seed)
    datasets = {
        "train": ArrayDualHeadDataset(
            features["train"],
            labels["train"]["success"],
            labels["train"]["failure"],
            labels["train"]["weights"],
            train_indices,
        ),
        "valid": ArrayDualHeadDataset(
            features["valid"],
            labels["valid"]["success"],
            labels["valid"]["failure"],
            labels["valid"]["weights"],
        ),
        "test": ArrayDualHeadDataset(
            features["test"],
            labels["test"]["success"],
            labels["test"]["failure"],
            labels["test"]["weights"],
        ),
    }
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        ),
        "valid": DataLoader(
            datasets["valid"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        ),
    }

    input_dim = int(features["train"].shape[1])
    model = DualHeadMLP(input_dim=input_dim, hidden_dims=args.hidden, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_valid = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    checkpoint_path = models_dir / f"{predictor_name}.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_weight = 0.0
        for batch in loaders["train"]:
            features_batch, success, failure, weights = batch
            features_batch = features_batch.to(device)
            success = success.to(device)
            failure = failure.to(device)
            weights = weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            success_logits, failure_logits = model(features_batch)
            loss = weighted_dual_loss(success_logits, failure_logits, success, failure, weights)
            loss.backward()
            optimizer.step()
            batch_weight = float(weights.sum().detach().cpu())
            train_loss += float(loss.detach().cpu()) * max(batch_weight, 1.0)
            train_weight += max(batch_weight, 1.0)
        train_loss = train_loss / max(train_weight, 1.0)
        valid_loss = _loss_on_loader(model, loaders["valid"], device)
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
        if valid_loss < best_valid:
            best_valid = valid_loss
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": input_dim,
                    "hidden": args.hidden,
                    "dropout": args.dropout,
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
    valid_success_raw, valid_failure_raw = _predict_probs(model, loaders["valid"], device)
    test_success_raw, test_failure_raw = _predict_probs(model, loaders["test"], device)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "predictor": predictor_name,
                "encoder_name": cache_meta.get("encoder_name"),
                "stage": "frozen_encoder_dual_head",
                "include_dense": bool(args.include_dense),
                "input_dim": int(input_dim),
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
            "stage": "frozen_encoder_dual_head",
            "predictor_name": predictor_name,
            "cache_dir": str(args.cache_dir),
            "cache_meta": cache_meta,
            "include_dense": bool(args.include_dense),
            "hidden": args.hidden,
            "dropout": args.dropout,
            "epochs_requested": int(args.epochs),
            "best_epoch": int(best_epoch),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "max_train_rows": int(args.max_train_rows),
            "device": str(device),
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
