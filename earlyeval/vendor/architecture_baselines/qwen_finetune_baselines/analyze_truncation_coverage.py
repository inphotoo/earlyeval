#!/usr/bin/env python3
"""Estimate truncation coverage for Qwen fine-tune text pairs.

Reads `bert_text_cache.parquet`, tokenizes `text_a`/`text_b` without truncation,
then simulates HuggingFace truncation strategies (`only_first`, `only_second`,
`longest_first`) at a target `max_length`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--truncation-side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("only_first", "only_second", "longest_first"),
        default=["only_first", "longest_first"],
    )
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--sample-rows-per-split", type=int, default=20000)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def _load_split(cache_path: Path, split_name: str) -> pd.DataFrame:
    cols = ["split", "text_a", "text_b", "text_a_chars", "text_b_chars", "row_idx"]
    frame = pd.read_parquet(cache_path, columns=cols, filters=[("split", "==", split_name)])
    if frame.empty:
        raise RuntimeError(f"No rows for split={split_name} in {cache_path}")
    return frame.reset_index(drop=True)


def _sample(frame: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    if n_rows <= 0 or len(frame) <= n_rows:
        return frame.reset_index(drop=True)
    return frame.sample(n=n_rows, random_state=seed).sort_values("row_idx").reset_index(drop=True)


def _token_lengths(tokenizer: Any, texts: list[str], batch_size: int) -> np.ndarray:
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        encoded = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_length=True,
        )
        batch_lengths = encoded.get("length")
        if batch_lengths is None:
            ids = encoded["input_ids"]
            batch_lengths = [len(x) for x in ids]
        lengths.extend(int(x) for x in batch_lengths)
    return np.asarray(lengths, dtype=np.int32)


def _simulate(
    len_a: np.ndarray,
    len_b: np.ndarray,
    specials: int,
    max_length: int,
    strategy: str,
) -> dict[str, np.ndarray]:
    total = len_a + len_b + int(specials)
    overflow = np.maximum(total - int(max_length), 0)

    rem_a = np.zeros_like(overflow)
    rem_b = np.zeros_like(overflow)
    unresolved = np.zeros_like(overflow)

    if strategy == "only_first":
        can = len_a > overflow
        rem_a = np.where(can, overflow, 0)
        unresolved = np.where((overflow > 0) & (~can), overflow, 0)
    elif strategy == "only_second":
        can = len_b > overflow
        rem_b = np.where(can, overflow, 0)
        unresolved = np.where((overflow > 0) & (~can), overflow, 0)
    elif strategy == "longest_first":
        diff = np.abs(len_b - len_a)
        first_remove = np.minimum(diff, overflow)
        second_remove = overflow - first_remove
        a_gt_b = len_a > len_b

        rem_a = np.where(a_gt_b, first_remove + (second_remove // 2), (second_remove // 2))
        rem_b = np.where(a_gt_b, second_remove - (second_remove // 2), first_remove + second_remove - (second_remove // 2))

        # Guard numerical edge cases (should be zero in normal settings).
        over_a = np.maximum(rem_a - len_a, 0)
        over_b = np.maximum(rem_b - len_b, 0)
        unresolved = over_a + over_b
        rem_a = np.minimum(rem_a, len_a)
        rem_b = np.minimum(rem_b, len_b)
    else:
        raise ValueError(strategy)

    kept_a = len_a - rem_a
    kept_b = len_b - rem_b
    final_total = kept_a + kept_b + int(specials)

    return {
        "total": total,
        "overflow": overflow,
        "removed_a": rem_a,
        "removed_b": rem_b,
        "kept_a": kept_a,
        "kept_b": kept_b,
        "final_total": final_total,
        "unresolved": unresolved,
    }


def _pct(mask: np.ndarray) -> float:
    if len(mask) == 0:
        return 0.0
    return float(mask.mean() * 100.0)


def _q(x: np.ndarray, p: float) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.quantile(x, p))


def _summary_row(
    split_name: str,
    n_total_split: int,
    n_sample: int,
    strategy: str,
    max_length: int,
    truncation_side: str,
    specials: int,
    metrics: dict[str, np.ndarray],
) -> dict[str, Any]:
    total = metrics["total"]
    overflow = metrics["overflow"]
    rem_a = metrics["removed_a"]
    rem_b = metrics["removed_b"]
    kept_a = metrics["kept_a"]
    kept_b = metrics["kept_b"]
    final_total = metrics["final_total"]
    unresolved = metrics["unresolved"]

    is_truncated = overflow > 0
    unresolved_mask = unresolved > 0

    row = {
        "split": split_name,
        "n_total_split": int(n_total_split),
        "n_sample": int(n_sample),
        "sample_frac_pct": float(100.0 * n_sample / max(n_total_split, 1)),
        "strategy": strategy,
        "truncation_side": truncation_side,
        "max_length": int(max_length),
        "special_tokens_pair": int(specials),
        "pct_rows_over_max_raw": _pct(is_truncated),
        "pct_rows_unresolved": _pct(unresolved_mask),
        "pct_rows_final_still_over_max": _pct(final_total > max_length),
        "pct_rows_removed_first": _pct(rem_a > 0),
        "pct_rows_removed_second": _pct(rem_b > 0),
        "mean_raw_total_tokens": float(total.mean()),
        "p50_raw_total_tokens": _q(total, 0.50),
        "p90_raw_total_tokens": _q(total, 0.90),
        "p95_raw_total_tokens": _q(total, 0.95),
        "p99_raw_total_tokens": _q(total, 0.99),
        "mean_overflow_tokens_all": float(overflow.mean()),
        "mean_overflow_tokens_truncated": float(overflow[is_truncated].mean()) if is_truncated.any() else 0.0,
        "mean_removed_first_all": float(rem_a.mean()),
        "mean_removed_second_all": float(rem_b.mean()),
        "mean_removed_first_truncated": float(rem_a[is_truncated].mean()) if is_truncated.any() else 0.0,
        "mean_removed_second_truncated": float(rem_b[is_truncated].mean()) if is_truncated.any() else 0.0,
        "mean_kept_first_tokens": float(kept_a.mean()),
        "mean_kept_second_tokens": float(kept_b.mean()),
        "mean_final_total_tokens": float(final_total.mean()),
    }
    return row


def _report_md(
    out_path: Path,
    *,
    cache_dir: Path,
    model_name: str,
    max_length: int,
    truncation_side: str,
    sample_rows: int,
    summary: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Truncation Coverage Report")
    lines.append("")
    lines.append(f"- cache_dir: `{cache_dir}`")
    lines.append(f"- model_name: `{model_name}`")
    lines.append(f"- max_length: `{max_length}`")
    lines.append(f"- truncation_side: `{truncation_side}`")
    lines.append(f"- sample_rows_per_split: `{sample_rows}` (0 means all rows)")
    lines.append("")

    for strategy in summary["strategy"].unique().tolist():
        sub = summary[summary["strategy"] == strategy].copy()
        lines.append(f"## Strategy: `{strategy}`")
        lines.append("")
        show_cols = [
            "split",
            "n_sample",
            "pct_rows_over_max_raw",
            "pct_rows_unresolved",
            "pct_rows_removed_first",
            "pct_rows_removed_second",
            "mean_overflow_tokens_truncated",
            "mean_final_total_tokens",
        ]
        lines.append(sub[show_cols].to_markdown(index=False, floatfmt=".3f"))
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- `pct_rows_over_max_raw`: raw pair token length exceeds max_length before truncation.")
    lines.append("- `pct_rows_unresolved`: strategy cannot satisfy max_length for some rows (important for only_first/only_second).")
    lines.append("- `pct_rows_removed_first/second`: fraction of rows where the segment lost tokens after truncation.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir
    cache_path = cache_dir / "bert_text_cache.parquet"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing text cache: {cache_path}")

    output_dir = args.output_dir or cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=not args.allow_download,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = str(args.truncation_side)

    specials = int(tokenizer.num_special_tokens_to_add(pair=True))

    rows: list[dict[str, Any]] = []

    for split_name in args.splits:
        frame = _load_split(cache_path, split_name)
        total_rows = len(frame)
        sample = _sample(frame, args.sample_rows_per_split, args.sample_seed)

        text_a = sample["text_a"].fillna("").astype(str).tolist()
        text_b = sample["text_b"].fillna("").astype(str).tolist()

        len_a = _token_lengths(tokenizer, text_a, args.batch_size)
        len_b = _token_lengths(tokenizer, text_b, args.batch_size)

        for strategy in args.strategies:
            metrics = _simulate(
                len_a=len_a,
                len_b=len_b,
                specials=specials,
                max_length=args.max_length,
                strategy=strategy,
            )
            rows.append(
                _summary_row(
                    split_name=split_name,
                    n_total_split=total_rows,
                    n_sample=len(sample),
                    strategy=strategy,
                    max_length=args.max_length,
                    truncation_side=args.truncation_side,
                    specials=specials,
                    metrics=metrics,
                )
            )

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "truncation_coverage_summary.csv"
    summary.to_csv(summary_path, index=False)

    md_path = output_dir / "truncation_coverage_report.md"
    _report_md(
        md_path,
        cache_dir=cache_dir,
        model_name=args.model_name,
        max_length=args.max_length,
        truncation_side=args.truncation_side,
        sample_rows=args.sample_rows_per_split,
        summary=summary,
    )

    meta = {
        "cache_dir": str(cache_dir),
        "cache_path": str(cache_path),
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "max_length": int(args.max_length),
        "truncation_side": args.truncation_side,
        "strategies": list(args.strategies),
        "splits": list(args.splits),
        "sample_rows_per_split": int(args.sample_rows_per_split),
        "sample_seed": int(args.sample_seed),
        "batch_size": int(args.batch_size),
        "special_tokens_pair": specials,
        "summary_csv": str(summary_path),
        "report_md": str(md_path),
    }
    (output_dir / "truncation_coverage_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(summary_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
