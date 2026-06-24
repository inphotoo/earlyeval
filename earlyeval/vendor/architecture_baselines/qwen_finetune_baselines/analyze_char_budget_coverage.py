#!/usr/bin/env python3
"""Analyze character-level truncation caused by text budget caps.

This inspects rows selected into `bert_text_cache.parquet` and compares raw text
lengths from the original prefix table against the cache-builder caps
(`task_chars`, `gold_chars`, `prefix_*_tail_chars`, etc.).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SEGMENTS = [
    ("task_prompt_text", "task_chars", "task_head"),
    ("gold_answer_summary_text", "gold_chars", "gold_head"),
    ("last_action_text", "last_action_chars", "last_action_head"),
    ("last_feedback_text", "last_feedback_chars", "last_feedback_head"),
    ("last_thought_text", "last_thought_chars", "last_thought_head"),
    ("prefix_action_text", "prefix_action_tail_chars", "prefix_action_tail"),
    ("prefix_feedback_text", "prefix_feedback_tail_chars", "prefix_feedback_tail"),
    ("prefix_thought_text", "prefix_thought_tail_chars", "prefix_thought_tail"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefix-table", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--sample-rows-per-split", type=int, default=0)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--task-chars", type=int, default=None)
    parser.add_argument("--gold-chars", type=int, default=None)
    parser.add_argument("--last-action-chars", type=int, default=None)
    parser.add_argument("--last-feedback-chars", type=int, default=None)
    parser.add_argument("--last-thought-chars", type=int, default=None)
    parser.add_argument("--prefix-action-tail-chars", type=int, default=None)
    parser.add_argument("--prefix-feedback-tail-chars", type=int, default=None)
    parser.add_argument("--prefix-thought-tail-chars", type=int, default=None)
    return parser.parse_args()


def _load_cache_meta(cache_dir: Path) -> dict[str, Any]:
    meta_path = cache_dir / "bert_cache_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing cache metadata: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _load_selected(cache_dir: Path, sample_rows_per_split: int, sample_seed: int) -> pd.DataFrame:
    cache_path = cache_dir / "bert_text_cache.parquet"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing text cache: {cache_path}")
    selected = pd.read_parquet(cache_path, columns=["prefix_id", "split"])
    selected["prefix_id"] = selected["prefix_id"].astype(str)

    if sample_rows_per_split > 0:
        parts: list[pd.DataFrame] = []
        for split_name in sorted(selected["split"].astype(str).unique().tolist()):
            part = selected[selected["split"] == split_name]
            if len(part) > sample_rows_per_split:
                part = part.sample(n=sample_rows_per_split, random_state=sample_seed)
            parts.append(part)
        selected = pd.concat(parts, ignore_index=True)

    # ensure unique prefix_id mapping
    selected = selected.drop_duplicates(subset=["prefix_id"], keep="first").reset_index(drop=True)
    return selected


def _init_stat() -> dict[str, Any]:
    return {
        "rows": 0,
        "rows_truncated": 0,
        "sum_raw_chars": 0,
        "sum_kept_chars": 0,
        "sum_removed_chars": 0,
        "max_raw_chars": 0,
        "max_removed_chars": 0,
    }


def _update_stat(stat: dict[str, Any], raw: np.ndarray, kept: np.ndarray, removed: np.ndarray) -> None:
    stat["rows"] += int(len(raw))
    stat["rows_truncated"] += int((removed > 0).sum())
    stat["sum_raw_chars"] += int(raw.sum())
    stat["sum_kept_chars"] += int(kept.sum())
    stat["sum_removed_chars"] += int(removed.sum())
    if len(raw):
        stat["max_raw_chars"] = max(stat["max_raw_chars"], int(raw.max()))
        stat["max_removed_chars"] = max(stat["max_removed_chars"], int(removed.max()))


def _safe_len_series(series: pd.Series) -> np.ndarray:
    return series.fillna("").astype(str).str.len().to_numpy(dtype=np.int32)


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir
    output_dir = args.output_dir or cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_cache_meta(cache_dir)
    budgets = meta.get("text_budget", {})
    override_map = {
        "task_chars": args.task_chars,
        "gold_chars": args.gold_chars,
        "last_action_chars": args.last_action_chars,
        "last_feedback_chars": args.last_feedback_chars,
        "last_thought_chars": args.last_thought_chars,
        "prefix_action_tail_chars": args.prefix_action_tail_chars,
        "prefix_feedback_tail_chars": args.prefix_feedback_tail_chars,
        "prefix_thought_tail_chars": args.prefix_thought_tail_chars,
    }
    for key, value in override_map.items():
        if value is not None:
            budgets[key] = int(value)
    prefix_table = args.prefix_table or Path(meta.get("prefix_table", ""))
    if not prefix_table.exists():
        raise FileNotFoundError(f"Missing prefix table: {prefix_table}")

    selected = _load_selected(cache_dir, args.sample_rows_per_split, args.sample_seed)
    if selected.empty:
        raise RuntimeError("No selected rows found in cache.")

    split_by_id = dict(zip(selected["prefix_id"].tolist(), selected["split"].astype(str).tolist()))
    wanted_ids = set(split_by_id.keys())

    cols = ["prefix_id"] + [name for name, _, _ in SEGMENTS]
    parquet = pq.ParquetFile(prefix_table)

    per_segment: dict[tuple[str, str], dict[str, Any]] = defaultdict(_init_stat)
    per_split_row: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "rows": 0,
        "rows_any_segment_truncated": 0,
        "sum_removed_chars_all_segments": 0,
        "max_removed_chars_all_segments": 0,
    })

    seen_ids: set[str] = set()

    for batch in parquet.iter_batches(batch_size=args.batch_size, columns=cols):
        df = batch.to_pandas()
        ids = df["prefix_id"].astype(str)
        mask = ids.isin(wanted_ids).to_numpy()
        if not mask.any():
            continue

        sub = df.loc[mask].reset_index(drop=True)
        sub_ids = sub["prefix_id"].astype(str)
        sub_splits = sub_ids.map(split_by_id).astype(str)

        # Track row-level aggregate removed chars across segments.
        row_removed_total = np.zeros(len(sub), dtype=np.int64)

        for col_name, budget_key, segment_name in SEGMENTS:
            cap = int(budgets.get(budget_key, 0))
            raw = _safe_len_series(sub[col_name])
            if cap > 0:
                kept = np.minimum(raw, cap).astype(np.int32)
            else:
                kept = raw.copy()
            removed = (raw - kept).astype(np.int32)
            row_removed_total += removed.astype(np.int64)

            # overall (all splits)
            key_all = ("all", segment_name)
            _update_stat(per_segment[key_all], raw, kept, removed)

            for split_name in ("train", "valid", "test"):
                s_mask = (sub_splits == split_name).to_numpy()
                if not s_mask.any():
                    continue
                key = (split_name, segment_name)
                _update_stat(per_segment[key], raw[s_mask], kept[s_mask], removed[s_mask])

        # update per-row aggregate stats per split
        for split_name in ("train", "valid", "test"):
            s_mask = (sub_splits == split_name).to_numpy()
            if not s_mask.any():
                continue
            removed_split = row_removed_total[s_mask]
            stat = per_split_row[split_name]
            stat["rows"] += int(s_mask.sum())
            stat["rows_any_segment_truncated"] += int((removed_split > 0).sum())
            stat["sum_removed_chars_all_segments"] += int(removed_split.sum())
            if len(removed_split):
                stat["max_removed_chars_all_segments"] = max(
                    stat["max_removed_chars_all_segments"], int(removed_split.max())
                )

        seen_ids.update(sub_ids.tolist())

    missing = len(wanted_ids - seen_ids)
    if missing:
        raise RuntimeError(f"Could not match {missing} selected prefix_id rows in prefix table.")

    rows: list[dict[str, Any]] = []
    for (split_name, segment_name), stat in sorted(per_segment.items()):
        n = int(stat["rows"])
        sum_raw = int(stat["sum_raw_chars"])
        sum_removed = int(stat["sum_removed_chars"])
        rows.append(
            {
                "split": split_name,
                "segment": segment_name,
                "cap_chars": None,
                "rows": n,
                "rows_truncated": int(stat["rows_truncated"]),
                "pct_rows_truncated": float(100.0 * stat["rows_truncated"] / max(n, 1)),
                "sum_raw_chars": sum_raw,
                "sum_kept_chars": int(stat["sum_kept_chars"]),
                "sum_removed_chars": sum_removed,
                "pct_chars_removed_vs_raw": float(100.0 * sum_removed / max(sum_raw, 1)),
                "mean_raw_chars_per_row": float(sum_raw / max(n, 1)),
                "mean_removed_chars_per_row": float(sum_removed / max(n, 1)),
                "max_raw_chars": int(stat["max_raw_chars"]),
                "max_removed_chars": int(stat["max_removed_chars"]),
            }
        )

    summary = pd.DataFrame(rows)

    # Fill cap column from mapping.
    cap_map = {segment_name: int(budgets.get(budget_key, 0)) for _, budget_key, segment_name in SEGMENTS}
    summary["cap_chars"] = summary["segment"].map(cap_map).fillna(0).astype(int)

    out_csv = output_dir / "char_budget_truncation_summary.csv"
    summary.to_csv(out_csv, index=False)

    row_rows = []
    for split_name, stat in sorted(per_split_row.items()):
        n = int(stat["rows"])
        sum_removed = int(stat["sum_removed_chars_all_segments"])
        row_rows.append(
            {
                "split": split_name,
                "rows": n,
                "rows_any_segment_truncated": int(stat["rows_any_segment_truncated"]),
                "pct_rows_any_segment_truncated": float(100.0 * stat["rows_any_segment_truncated"] / max(n, 1)),
                "sum_removed_chars_all_segments": sum_removed,
                "mean_removed_chars_all_segments_per_row": float(sum_removed / max(n, 1)),
                "max_removed_chars_all_segments": int(stat["max_removed_chars_all_segments"]),
            }
        )
    row_summary = pd.DataFrame(row_rows)
    out_row_csv = output_dir / "char_budget_truncation_row_summary.csv"
    row_summary.to_csv(out_row_csv, index=False)

    md_lines: list[str] = []
    md_lines.append("# Character Budget Truncation Report")
    md_lines.append("")
    md_lines.append(f"- cache_dir: `{cache_dir}`")
    md_lines.append(f"- prefix_table: `{prefix_table}`")
    md_lines.append(f"- sample_rows_per_split: `{args.sample_rows_per_split}` (0 means all rows)")
    md_lines.append("")
    md_lines.append("## Row-Level Impact")
    md_lines.append("")
    if not row_summary.empty:
        md_lines.append(row_summary.to_markdown(index=False, floatfmt=".3f"))
    else:
        md_lines.append("(no rows)")
    md_lines.append("")

    md_lines.append("## Segment-Level Impact (all splits)")
    md_lines.append("")
    all_seg = summary[summary["split"] == "all"].copy()
    show_cols = [
        "segment",
        "cap_chars",
        "rows",
        "pct_rows_truncated",
        "pct_chars_removed_vs_raw",
        "mean_raw_chars_per_row",
        "mean_removed_chars_per_row",
        "max_removed_chars",
    ]
    if not all_seg.empty:
        md_lines.append(all_seg[show_cols].to_markdown(index=False, floatfmt=".3f"))
    else:
        md_lines.append("(no rows)")
    md_lines.append("")

    out_md = output_dir / "char_budget_truncation_report.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    out_meta = output_dir / "char_budget_truncation_meta.json"
    meta_payload = {
        "cache_dir": str(cache_dir),
        "prefix_table": str(prefix_table),
        "sample_rows_per_split": int(args.sample_rows_per_split),
        "sample_seed": int(args.sample_seed),
        "batch_size": int(args.batch_size),
        "budgets": budgets,
        "selected_rows": int(len(selected)),
        "summary_csv": str(out_csv),
        "row_summary_csv": str(out_row_csv),
        "report_md": str(out_md),
    }
    out_meta.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(out_csv)
    print(out_row_csv)
    print(out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
