from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from final3.core.io import ensure_dir
from final3.experiments.rq_final import (
    _default_output_dir,
    _excluded_models_from_config,
    _float_sequence,
    _markdown_table,
    load_rq_final_config,
)
from final3.policies.safe_stop import head_column


def _evaluate_complement_frame(
    frame: pd.DataFrame,
    *,
    predictor: str,
    score_mode: str,
    thresholds: list[float],
) -> pd.DataFrame:
    success_col = head_column("success", score_mode, predictor)
    failure_col = head_column("failure", score_mode, predictor)
    required = ["traj_id", "label", "prefix_step_idx", success_col, failure_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Prediction table is missing required columns: {missing}")

    own = np.asarray(thresholds, dtype=np.float64)
    opposite = 1.0 - own
    n = len(own)
    counts = {
        "decided_failure": np.zeros(n, dtype=np.int64),
        "decided_success": np.zeros(n, dtype=np.int64),
        "undecided": np.zeros(n, dtype=np.int64),
        "false_negatives": np.zeros(n, dtype=np.int64),
        "true_negatives": np.zeros(n, dtype=np.int64),
        "false_positives": np.zeros(n, dtype=np.int64),
        "true_positives": np.zeros(n, dtype=np.int64),
        "total_saved_steps": np.zeros(n, dtype=np.int64),
    }
    original_total = 0
    original_resolved = 0
    total_steps = 0

    for _, group in frame.groupby("traj_id", sort=False):
        group = group.sort_values("prefix_step_idx")
        label = int(group["label"].iloc[0])
        steps = group["prefix_step_idx"].to_numpy(dtype=np.int32)
        success_scores = group[success_col].to_numpy(dtype=np.float64)
        failure_scores = group[failure_col].to_numpy(dtype=np.float64)
        n_steps = int(len(group))
        saved_by_index = np.asarray([(steps > step).sum() for step in steps], dtype=np.int64)

        original_total += 1
        original_resolved += int(label == 1)
        total_steps += n_steps

        decided = np.zeros(n, dtype=bool)
        decided_success = np.zeros(n, dtype=bool)
        for idx, (success_score, failure_score) in enumerate(zip(success_scores, failure_scores)):
            if bool(decided.all()):
                break
            success_hit = (success_score >= own) & (failure_score < opposite)
            failure_hit = (failure_score >= own) & (success_score < opposite)
            new_decision = (~decided) & (success_hit | failure_hit)
            if not bool(new_decision.any()):
                continue
            new_success = new_decision & success_hit
            decided[new_decision] = True
            decided_success[new_success] = True
            counts["total_saved_steps"][new_decision] += int(saved_by_index[idx])

        decided_failure = decided & ~decided_success
        undecided = ~decided
        counts["decided_success"] += decided_success.astype(np.int64)
        counts["decided_failure"] += decided_failure.astype(np.int64)
        counts["undecided"] += undecided.astype(np.int64)
        if label == 1:
            counts["true_positives"] += decided_success.astype(np.int64)
            counts["false_negatives"] += decided_failure.astype(np.int64)
        else:
            counts["false_positives"] += decided_success.astype(np.int64)
            counts["true_negatives"] += decided_failure.astype(np.int64)

    original_rate = original_resolved / original_total if original_total else 0.0
    rows: list[dict[str, Any]] = []
    for i, threshold in enumerate(thresholds):
        tp = int(counts["true_positives"][i])
        tn = int(counts["true_negatives"][i])
        fp = int(counts["false_positives"][i])
        fn = int(counts["false_negatives"][i])
        decided_success_count = int(counts["decided_success"][i])
        decided_failure_count = int(counts["decided_failure"][i])
        n_decided = decided_success_count + decided_failure_count
        adjusted_resolved = int(original_resolved - fn + fp)
        adjusted_rate = adjusted_resolved / original_total if original_total else 0.0
        rows.append(
            {
                "policy_name": f"opposite_complement_t{threshold:.2f}",
                "threshold": float(threshold),
                "own_thr": float(threshold),
                "opposite_cap": float(1.0 - threshold),
                "success_thr": float(threshold),
                "failure_thr": float(threshold),
                "original_total": int(original_total),
                "original_resolved": int(original_resolved),
                "original_resolve_rate": float(original_rate),
                "decided_failure": decided_failure_count,
                "decided_success": decided_success_count,
                "undecided": int(counts["undecided"][i]),
                "false_negatives": fn,
                "true_negatives": tn,
                "false_positives": fp,
                "true_positives": tp,
                "n_decided": int(n_decided),
                "coverage_pct": 100.0 * n_decided / original_total if original_total else float("nan"),
                "decision_accuracy_pct": 100.0 * (tp + tn) / n_decided if n_decided else 100.0,
                "precision_success_pct": 100.0 * tp / decided_success_count if decided_success_count else float("nan"),
                "precision_failure_pct": 100.0 * tn / decided_failure_count if decided_failure_count else float("nan"),
                "adjusted_resolved": adjusted_resolved,
                "adjusted_resolve_rate": float(adjusted_rate),
                "resolve_rate_drop_pp": 100.0 * (original_rate - adjusted_rate),
                "resolve_rate_change_pp": 100.0 * (adjusted_rate - original_rate),
                "pct_steps_saved": 100.0 * int(counts["total_saved_steps"][i]) / total_steps if total_steps else float("nan"),
                "step_save_pct": 100.0 * int(counts["total_saved_steps"][i]) / total_steps if total_steps else float("nan"),
                "total_saved_steps": int(counts["total_saved_steps"][i]),
                "total_steps": int(total_steps),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_fold(
    fold_dir: str,
    *,
    predictor: str,
    score_mode: str,
    thresholds: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = Path(fold_dir)
    fold_id = path.name
    out: dict[str, list[dict[str, Any]]] = {}
    for split in ("valid", "test"):
        frame = pd.read_parquet(path / f"{split}_predictions_safe_stop.parquet")
        metrics = _evaluate_complement_frame(
            frame,
            predictor=predictor,
            score_mode=score_mode,
            thresholds=thresholds,
        )
        metrics["fold_id"] = fold_id
        metrics["test_model"] = fold_id
        metrics["split"] = split
        out[split] = metrics.to_dict("records")
    return out["valid"], out["test"]


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (threshold, opposite_cap), part in frame.groupby(["threshold", "opposite_cap"], sort=True):
        total = float(part["original_total"].sum())
        decided = float(part["n_decided"].sum())
        original_resolved = float(part["original_resolved"].sum())
        adjusted_resolved = float(part["adjusted_resolved"].sum())
        fn = float(part["false_negatives"].sum())
        fp = float(part["false_positives"].sum())
        tp = float(part["true_positives"].sum())
        tn = float(part["true_negatives"].sum())
        saved = float(part["total_saved_steps"].sum())
        total_steps = float(part["total_steps"].sum())
        fold_totals = part["original_total"].astype(float)
        fold_change = (
            (part["adjusted_resolved"].astype(float) - part["original_resolved"].astype(float))
            * 100.0
            / fold_totals.replace(0.0, float("nan"))
        )
        rows.append(
            {
                "threshold": float(threshold),
                "own_thr": float(threshold),
                "opposite_cap": float(opposite_cap),
                "folds": int(part["fold_id"].nunique()),
                "trajectories": int(total),
                "original_resolved": int(original_resolved),
                "adjusted_resolved": int(adjusted_resolved),
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "true_positives": int(tp),
                "original_resolve_rate_pct": original_resolved * 100.0 / total if total else 0.0,
                "adjusted_resolve_rate_pct": adjusted_resolved * 100.0 / total if total else 0.0,
                "resolve_rate_change_pp": (adjusted_resolved - original_resolved) * 100.0 / total if total else 0.0,
                "mean_abs_resolve_rate_change_pp": float((fold_change.abs() * fold_totals).sum() / total) if total else 0.0,
                "decided_trajectories": int(decided),
                "coverage_pct": decided * 100.0 / total if total else 0.0,
                "decision_accuracy_pct": (tp + tn) * 100.0 / decided if decided else 100.0,
                "saved_steps": int(saved),
                "total_steps": int(total_steps),
                "step_save_pct": saved * 100.0 / total_steps if total_steps else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("threshold")


def _write_plots(out_dir: Path, aggregate_test: pd.DataFrame) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    paths: list[str] = []
    x = aggregate_test["threshold"].astype(float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, aggregate_test["step_save_pct"].astype(float), marker="o", label="Step saving")
    ax.plot(x, aggregate_test["decision_accuracy_pct"].astype(float), marker="o", label="Decision accuracy")
    ax.set_xlabel("Own-head threshold")
    ax.set_ylabel("Percent")
    ax.set_title("Complement Opposite-Head Rule")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "aggregate_test_frontier.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, aggregate_test["resolve_rate_change_pp"].astype(float), marker="o", label="Resolve change")
    ax.plot(x, aggregate_test["mean_abs_resolve_rate_change_pp"].astype(float), marker="o", label="Mean abs shift")
    ax.axhline(0.0, color="#555555", linewidth=1)
    ax.set_xlabel("Own-head threshold")
    ax.set_ylabel("pp")
    ax.set_title("Resolve-Rate Shift")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "aggregate_test_resolve_shift.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def _write_readme(out_dir: Path, aggregate_test: pd.DataFrame, manifest: dict[str, Any]) -> None:
    lines = [
        "# Complement Opposite-Head Sweep",
        "",
        "Rule:",
        "",
        "- success stop: `success_score >= t AND failure_score < 1 - t`",
        "- failure stop: `failure_score >= t AND success_score < 1 - t`",
        "",
        "Resolve change uses both false negatives and false positives:",
        "",
        "`adjusted_resolved = original_resolved - false_negatives + false_positives`",
        "",
        f"- completed folds used: `{manifest['folds']}`",
        f"- workers: `{manifest['workers']}`",
        f"- threshold range: `{manifest['thresholds'][0]}` to `{manifest['thresholds'][-1]}`",
        "",
        "## Outputs",
        "",
        "- `per_fold_valid_metrics.csv`",
        "- `per_fold_test_metrics.csv`",
        "- `aggregate_valid_metrics.csv`",
        "- `aggregate_test_metrics.csv`",
        "",
        "## Aggregate Test",
        "",
    ]
    display = []
    for row in aggregate_test.to_dict("records"):
        display.append(
            {
                "t": f"{float(row['threshold']):.2f}",
                "opposite": f"{float(row['opposite_cap']):.2f}",
                "save": f"{float(row['step_save_pct']):.2f}%",
                "acc": f"{float(row['decision_accuracy_pct']):.2f}%",
                "change": f"{float(row['resolve_rate_change_pp']):+.2f}pp",
                "mean_abs": f"{float(row['mean_abs_resolve_rate_change_pp']):.2f}pp",
                "FN": int(row["false_negatives"]),
                "FP": int(row["false_positives"]),
            }
        )
    lines.extend(_markdown_table(display, ["t", "opposite", "save", "acc", "change", "mean_abs", "FN", "FP"]))
    lines.extend(["", "## Plots", "", "- `aggregate_test_frontier.png`", "- `aggregate_test_resolve_shift.png`"])
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(
    *,
    config: Path,
    output_dir: Path | None,
    out_subdir: str,
    workers: int,
    threshold_start: float,
    threshold_stop: float,
    threshold_step: float,
) -> dict[str, Any]:
    started = time.time()
    cfg = load_rq_final_config(config)
    root = ensure_dir(output_dir or _default_output_dir(cfg, cfg.run_id))
    run_dir = root / "lightgbm_main"
    out_dir = ensure_dir(run_dir / "policy_sweeps" / out_subdir)

    sweep_cfg = cfg.payload.get("policy_sweep") or {}
    predictor_values = sweep_cfg.get("prefix_models") or sweep_cfg.get("predictors") or ["I_LightGBM_Dense_AF"]
    predictor = str(predictor_values[0] if isinstance(predictor_values, list) else predictor_values)
    score_modes = sweep_cfg.get("score_modes", ["calibrated"])
    score_mode = str(score_modes[0] if isinstance(score_modes, list) else score_modes)
    thresholds = _float_sequence({"start": threshold_start, "stop": threshold_stop, "step": threshold_step}, None)

    excluded = _excluded_models_from_config(cfg)
    fold_dirs = [path.parent for path in sorted((run_dir / "folds").glob("*/_SUCCESS")) if path.parent.name not in excluded]
    if not fold_dirs:
        raise FileNotFoundError(f"No completed LightGBM folds found under {run_dir / 'folds'}")

    valid_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    max_workers = max(1, int(workers))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _evaluate_fold,
                str(fold_dir),
                predictor=predictor,
                score_mode=score_mode,
                thresholds=thresholds,
            ): fold_dir.name
            for fold_dir in fold_dirs
        }
        for future in as_completed(futures):
            fold_id = futures[future]
            fold_valid, fold_test = future.result()
            valid_rows.extend(fold_valid)
            test_rows.extend(fold_test)
            print(f"[opposite-complement] completed {fold_id}", flush=True)

    valid = pd.DataFrame(valid_rows).sort_values(["fold_id", "threshold"])
    test = pd.DataFrame(test_rows).sort_values(["fold_id", "threshold"])
    aggregate_valid = _aggregate(valid)
    aggregate_test = _aggregate(test)
    valid.to_csv(out_dir / "per_fold_valid_metrics.csv", index=False)
    test.to_csv(out_dir / "per_fold_test_metrics.csv", index=False)
    aggregate_valid.to_csv(out_dir / "aggregate_valid_metrics.csv", index=False)
    aggregate_test.to_csv(out_dir / "aggregate_test_metrics.csv", index=False)
    plots = _write_plots(out_dir, aggregate_test)
    manifest = {
        "output_dir": str(out_dir),
        "folds": len(fold_dirs),
        "workers": max_workers,
        "predictor": predictor,
        "score_mode": score_mode,
        "thresholds": thresholds,
        "rule": "own_head >= t and opposite_head < 1 - t",
        "plots": plots,
        "elapsed_sec": time.time() - started,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_readme(out_dir, aggregate_test, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a parallel complement opposite-head threshold sweep.")
    parser.add_argument("--config", type=Path, default=Path("configs/rq_final.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/experiments/rq_final_lightgbm_17"))
    parser.add_argument("--out-subdir", default="opposite_complement_075_095")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--threshold-start", type=float, default=0.75)
    parser.add_argument("--threshold-stop", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run_sweep(
        config=args.config,
        output_dir=args.output_dir,
        out_subdir=args.out_subdir,
        workers=args.workers,
        threshold_start=args.threshold_start,
        threshold_stop=args.threshold_stop,
        threshold_step=args.threshold_step,
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
