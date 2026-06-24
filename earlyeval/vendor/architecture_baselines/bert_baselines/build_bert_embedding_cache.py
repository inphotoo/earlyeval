#!/usr/bin/env python3
"""Build compact BERT text and frozen-encoder embedding caches.

The cache is keyed to the same train/valid/test split used by the current
safe-stop experiments.  It streams the large prefix parquet once, writes a
compact text-pair parquet, and optionally writes float16 embedding ``.npy``
files for frozen-head training.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from bert_baseline_common import (
    DEFAULT_ENCODER,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_RUN_NAME,
    DEFAULT_VERIFIED_JSONL,
    TEXT_CACHE_COLUMNS,
    TEXT_COLUMNS,
    add_safe_labels,
    build_bert_text_pair,
    load_hf_encoder,
    load_split_frames,
    mean_pool,
    pool_encoder_output,
    resolve_device,
    set_threads,
    write_json,
)


DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "bert_codebert_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--prefix-table", type=Path, default=None)
    parser.add_argument("--verified-jsonl", type=Path, default=DEFAULT_VERIFIED_JSONL)
    parser.add_argument("--holdout-models", default="auto_mid3")
    parser.add_argument(
        "--exclude-train-models",
        nargs="*",
        default=None,
        help="Drop configured-excluded model ids before building train/valid/test splits.",
    )
    parser.add_argument("--max-instances", type=int, default=500)
    parser.add_argument("--split-strategy", choices=("per_instance_model", "per_instance_traj"), default="per_instance_model")
    parser.add_argument("--valid-models-per-instance", type=int, default=3)
    parser.add_argument("--valid-traj-ratio", type=float, default=0.15)
    parser.add_argument("--valid-per-instance", type=int, default=0)
    parser.add_argument("--smoke-trajectories-per-split", type=int, default=0)
    parser.add_argument("--cache-splits", nargs="+", choices=("train", "valid", "test"), default=["train", "valid", "test"])
    parser.add_argument("--trajectory-sample-frac", type=float, default=0.0)
    parser.add_argument("--trajectory-sample-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--safe-label-min-step", type=int, default=10)
    parser.add_argument("--mask-train-model-id", action="store_true", default=True)
    parser.add_argument("--keep-train-model-id", dest="mask_train_model_id", action="store_false")
    parser.add_argument("--include-dense", action="store_true", help="Also save dense structured feature arrays.")
    parser.add_argument("--encoder-name", default=DEFAULT_ENCODER)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true", help="Allow HF downloads instead of local cache only.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--pooling", choices=("mean", "cls"), default="mean")
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--parquet-batch-size", type=int, default=4096)
    parser.add_argument("--parquet-compression", default="zstd")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--task-chars", type=int, default=600)
    parser.add_argument("--gold-chars", type=int, default=600)
    parser.add_argument("--last-action-chars", type=int, default=300)
    parser.add_argument("--last-feedback-chars", type=int, default=300)
    parser.add_argument("--last-thought-chars", type=int, default=300)
    parser.add_argument("--prefix-action-tail-chars", type=int, default=900)
    parser.add_argument("--prefix-feedback-tail-chars", type=int, default=900)
    parser.add_argument("--prefix-thought-tail-chars", type=int, default=700)
    parser.add_argument("--max-cpu-threads", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _maybe_clean_known_outputs(output_dir: Path) -> None:
    known = [
        "bert_text_cache.parquet",
        "bert_cache_meta.json",
        "split_metadata.json",
        "split_summary.csv",
        "bert_embeddings_train.npy",
        "bert_embeddings_valid.npy",
        "bert_embeddings_test.npy",
        "bert_dense_train.npy",
        "bert_dense_valid.npy",
        "bert_dense_test.npy",
    ]
    for name in known:
        path = output_dir / name
        if path.exists():
            path.unlink()


def _cache_exists(output_dir: Path, skip_embeddings: bool) -> bool:
    needed = [output_dir / "bert_text_cache.parquet", output_dir / "bert_cache_meta.json"]
    if not skip_embeddings:
        needed.extend(
            [
                output_dir / "bert_embeddings_train.npy",
                output_dir / "bert_embeddings_valid.npy",
                output_dir / "bert_embeddings_test.npy",
            ]
        )
    return all(path.exists() for path in needed)


def _embedding_batch(
    *,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    text_a: list[str],
    text_b: list[str],
    max_length: int,
    pooling: str,
) -> np.ndarray:
    encoded = tokenizer(
        text_a,
        text_b,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        outputs = model(**encoded, return_dict=True)
        if pooling == "mean":
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        else:
            pooled = pool_encoder_output(outputs, encoded["attention_mask"], pooling)
    return pooled.detach().float().cpu().numpy().astype(np.float16)


def _prepare_selected_frames(
    *,
    df_train: pd.DataFrame,
    df_valid: pd.DataFrame,
    df_test: pd.DataFrame,
    safe_label_min_step: int,
) -> dict[str, pd.DataFrame]:
    frames = {}
    for split_name, frame in (("train", df_train), ("valid", df_valid), ("test", df_test)):
        prepared = add_safe_labels(frame, safe_label_min_step).copy()
        prepared["row_idx"] = np.arange(len(prepared), dtype=np.int64)
        frames[split_name] = prepared
    return frames


def _filter_cache_frames(
    frames: dict[str, pd.DataFrame],
    *,
    cache_splits: list[str],
    trajectory_sample_frac: float,
    trajectory_sample_seed: int,
) -> dict[str, pd.DataFrame]:
    selected: dict[str, pd.DataFrame] = {}
    for split_name in dict.fromkeys(cache_splits):
        if split_name not in frames:
            continue
        frame = frames[split_name]
        if 0.0 < float(trajectory_sample_frac) < 1.0:
            traj_ids = np.array(sorted(frame["traj_id"].astype(str).unique()))
            n_keep = max(1, int(round(len(traj_ids) * float(trajectory_sample_frac))))
            rng = np.random.default_rng(int(trajectory_sample_seed))
            keep = set(rng.choice(traj_ids, size=n_keep, replace=False).tolist())
            frame = frame[frame["traj_id"].astype(str).isin(keep)].copy()
        frame = frame.reset_index(drop=True).copy()
        frame["row_idx"] = np.arange(len(frame), dtype=np.int64)
        if frame.empty:
            raise RuntimeError(f"No rows selected for cache split={split_name}.")
        selected[split_name] = frame
    if not selected:
        raise RuntimeError("No cache splits selected.")
    return selected


def _text_pairs_for_batch(
    batch_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list[str], list[str]]:
    text_a: list[str] = []
    text_b: list[str] = []
    for _, row in batch_df.iterrows():
        a, b = build_bert_text_pair(
            row,
            task_chars=args.task_chars,
            gold_chars=args.gold_chars,
            last_action_chars=args.last_action_chars,
            last_feedback_chars=args.last_feedback_chars,
            last_thought_chars=args.last_thought_chars,
            prefix_action_tail_chars=args.prefix_action_tail_chars,
            prefix_feedback_tail_chars=args.prefix_feedback_tail_chars,
            prefix_thought_tail_chars=args.prefix_thought_tail_chars,
        )
        text_a.append(a)
        text_b.append(b)
    return text_a, text_b


def _write_dense_arrays(
    output_dir: Path,
    frames: dict[str, pd.DataFrame],
    feature_engineer: Any,
) -> dict[str, str]:
    dense_files: dict[str, str] = {}
    if feature_engineer is None:
        return dense_files
    for split_name, frame in frames.items():
        dense = feature_engineer.transform_dense(frame).astype(np.float32)
        path = output_dir / f"bert_dense_{split_name}.npy"
        np.save(path, dense)
        dense_files[split_name] = str(path)
        del dense
        gc.collect()
    return dense_files


def main() -> int:
    args = parse_args()
    set_threads(args.max_cpu_threads)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if _cache_exists(output_dir, skip_embeddings=args.skip_embeddings) and not args.overwrite:
        print(output_dir)
        return 0
    if args.overwrite:
        _maybe_clean_known_outputs(output_dir)

    device = resolve_device(args.device) if not args.skip_embeddings else torch.device("cpu")
    tokenizer = model = None
    hidden_dim = None
    if not args.skip_embeddings:
        tokenizer, model = load_hf_encoder(
            encoder_name=args.encoder_name,
            device=device,
            local_files_only=not args.allow_download,
            fp16=args.fp16,
        )
        hidden_dim = int(getattr(model.config, "hidden_size", 0))
        if hidden_dim <= 0:
            raise RuntimeError(f"Cannot infer hidden size from encoder: {args.encoder_name}")

    df_train, df_valid, df_test, split_meta, split_summary, feature_engineer, prefix_path = load_split_frames(
        run_name=args.run_name,
        prefix_table=args.prefix_table,
        verified_jsonl=args.verified_jsonl,
        holdout_models=args.holdout_models,
        max_instances=args.max_instances,
        split_strategy=args.split_strategy,
        valid_traj_ratio=args.valid_traj_ratio,
        valid_per_instance=args.valid_per_instance,
        valid_models_per_instance=args.valid_models_per_instance,
        smoke_trajectories_per_split=args.smoke_trajectories_per_split,
        seed=args.seed,
        mask_train_model_id=args.mask_train_model_id,
        include_dense_columns=args.include_dense,
        exclude_train_models=args.exclude_train_models,
    )
    frames = _prepare_selected_frames(
        df_train=df_train,
        df_valid=df_valid,
        df_test=df_test,
        safe_label_min_step=args.safe_label_min_step,
    )
    frames = _filter_cache_frames(
        frames,
        cache_splits=args.cache_splits,
        trajectory_sample_frac=args.trajectory_sample_frac,
        trajectory_sample_seed=args.trajectory_sample_seed,
    )
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    write_json(output_dir / "split_metadata.json", split_meta)
    dense_files = _write_dense_arrays(output_dir, frames, feature_engineer) if args.include_dense else {}

    selected = pd.concat(frames.values(), ignore_index=True)
    selected_cols = [column for column in TEXT_CACHE_COLUMNS if column not in {"text_a", "text_b", "text_a_chars", "text_b_chars"}]
    selected_lookup = selected[selected_cols].copy()
    selected_lookup["prefix_id"] = selected_lookup["prefix_id"].astype(str)
    selected_lookup = selected_lookup.set_index("prefix_id", drop=False)
    selected_prefix_ids = set(selected_lookup.index)
    split_positions = {
        split_name: pd.Series(frame["row_idx"].to_numpy(dtype=np.int64), index=frame["prefix_id"].astype(str))
        for split_name, frame in frames.items()
    }

    embedding_paths: dict[str, str] = {}
    embeddings: dict[str, np.memmap] = {}
    filled: dict[str, np.ndarray] = {}
    if not args.skip_embeddings:
        for split_name, frame in frames.items():
            path = output_dir / f"bert_embeddings_{split_name}.npy"
            embeddings[split_name] = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.float16,
                shape=(len(frame), int(hidden_dim)),
            )
            filled[split_name] = np.zeros(len(frame), dtype=bool)
            embedding_paths[split_name] = str(path)

    text_cache_path = output_dir / "bert_text_cache.parquet"
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    available_columns = set(pq.ParquetFile(prefix_path).schema_arrow.names)
    text_columns = ["prefix_id"] + [column for column in TEXT_COLUMNS if column in available_columns]
    parquet_file = pq.ParquetFile(prefix_path)

    for batch in parquet_file.iter_batches(batch_size=args.parquet_batch_size, columns=text_columns):
        raw_batch = batch.to_pandas()
        raw_prefix_ids = raw_batch["prefix_id"].astype(str)
        keep_mask = raw_prefix_ids.isin(selected_prefix_ids).to_numpy()
        if not keep_mask.any():
            continue
        text_rows = raw_batch.loc[keep_mask].reset_index(drop=True)
        prefix_ids = text_rows["prefix_id"].astype(str).tolist()
        meta = selected_lookup.loc[prefix_ids].reset_index(drop=True)
        text_a, text_b = _text_pairs_for_batch(text_rows, args)
        cache_batch = meta.copy()
        cache_batch["text_a"] = text_a
        cache_batch["text_b"] = text_b
        cache_batch["text_a_chars"] = [len(item) for item in text_a]
        cache_batch["text_b_chars"] = [len(item) for item in text_b]
        cache_batch = cache_batch[TEXT_CACHE_COLUMNS]
        table = pa.Table.from_pandas(cache_batch, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(text_cache_path, table.schema, compression=args.parquet_compression)
        writer.write_table(table)
        rows_written += len(cache_batch)

        if not args.skip_embeddings:
            assert tokenizer is not None and model is not None
            for start in range(0, len(cache_batch), args.encoder_batch_size):
                end = min(start + args.encoder_batch_size, len(cache_batch))
                emb = _embedding_batch(
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    text_a=cache_batch["text_a"].iloc[start:end].tolist(),
                    text_b=cache_batch["text_b"].iloc[start:end].tolist(),
                    max_length=args.max_length,
                    pooling=args.pooling,
                )
                sub = cache_batch.iloc[start:end]
                split_values = sub["split"].to_numpy()
                row_indices = sub["row_idx"].to_numpy(dtype=np.int64)
                for split_name in frames:
                    split_mask = split_values == split_name
                    if split_mask.any():
                        embeddings[split_name][row_indices[split_mask]] = emb[split_mask]
                        filled[split_name][row_indices[split_mask]] = True
        del raw_batch, text_rows, meta, cache_batch

    if writer is None:
        raise RuntimeError("No selected rows were written to the BERT text cache.")
    writer.close()

    if rows_written != len(selected):
        raise RuntimeError(f"Text cache wrote {rows_written} rows, expected {len(selected)}.")

    if not args.skip_embeddings:
        for split_name, mask in filled.items():
            missing = int((~mask).sum())
            if missing:
                raise RuntimeError(f"Embedding cache missing {missing} rows for split={split_name}.")
            embeddings[split_name].flush()

    cache_meta = {
        "run_name": args.run_name,
        "prefix_table": str(prefix_path),
        "verified_jsonl": str(args.verified_jsonl),
        "holdout_models": args.holdout_models,
        "max_instances": int(args.max_instances),
        "split_strategy": args.split_strategy,
        "valid_models_per_instance": int(args.valid_models_per_instance),
        "smoke_trajectories_per_split": int(args.smoke_trajectories_per_split),
        "cache_splits": list(dict.fromkeys(args.cache_splits)),
        "trajectory_sample_frac": float(args.trajectory_sample_frac),
        "trajectory_sample_seed": int(args.trajectory_sample_seed),
        "safe_label_min_step": int(args.safe_label_min_step),
        "encoder_name": args.encoder_name,
        "max_length": int(args.max_length),
        "pooling": args.pooling,
        "fp16": bool(args.fp16),
        "device": str(device),
        "local_files_only": not args.allow_download,
        "include_dense": bool(args.include_dense),
        "text_cache_file": str(text_cache_path),
        "embedding_files": embedding_paths,
        "dense_files": dense_files,
        "row_counts": {split_name: int(len(frame)) for split_name, frame in frames.items()},
        "embedding_dim": int(hidden_dim or 0),
        "text_budget": {
            "task_chars": args.task_chars,
            "gold_chars": args.gold_chars,
            "last_action_chars": args.last_action_chars,
            "last_feedback_chars": args.last_feedback_chars,
            "last_thought_chars": args.last_thought_chars,
            "prefix_action_tail_chars": args.prefix_action_tail_chars,
            "prefix_feedback_tail_chars": args.prefix_feedback_tail_chars,
            "prefix_thought_tail_chars": args.prefix_thought_tail_chars,
        },
    }
    write_json(output_dir / "bert_cache_meta.json", cache_meta)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
