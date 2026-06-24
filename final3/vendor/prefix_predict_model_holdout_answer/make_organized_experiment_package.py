#!/usr/bin/env python3
"""Build a lightweight, paper-facing package for the messy holdout experiments.

The package intentionally copies only code, reports, small CSVs, and plots.
Large parquet/model/raw diagnostic files are recorded in manifests instead of
being duplicated.
"""

from __future__ import annotations

import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RUN_NAME = "model_holdout_answer_calibrated_full"
RUN_ROOT = ROOT / "runs" / RUN_NAME
REPORTS = RUN_ROOT / "reports"
VIS = REPORTS / "safe_stop_dual_head_visual_summary"
PKG = ROOT / "organized_experiment_package_20260503"

SMALL_CSV_LIMIT = 2 * 1024 * 1024
PLOT_LIMIT = 6 * 1024 * 1024


@dataclass
class CopiedArtifact:
    source: Path
    destination: Path | None
    status: str
    size_bytes: int
    note: str


COPIED: list[CopiedArtifact] = []


def rel(path: Path, base: Path = ROOT) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def ensure_dirs() -> None:
    for subdir in [
        "00_quick_start",
        "01_final_recommendation",
        "02_data_audit",
        "03_core_results/project_docs",
        "03_core_results/safe_stop_dual_head_visual_summary",
        "04_diagnostics/problem_diagnosis_summaries",
        "05_code_manifest/code_snapshot",
        "06_paper_tables",
        "99_source_manifests",
    ]:
        (PKG / subdir).mkdir(parents=True, exist_ok=True)


def record(source: Path, destination: Path | None, status: str, note: str = "") -> None:
    size = source.stat().st_size if source.exists() else 0
    COPIED.append(CopiedArtifact(source, destination, status, size, note))


def copy_file(source: Path, destination: Path, note: str = "") -> None:
    if not source.exists():
        record(source, None, "missing", note)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    record(source, destination, "copied", note)


def copy_tree_filtered(source_dir: Path, destination_dir: Path, note: str = "") -> None:
    if not source_dir.exists():
        record(source_dir, None, "missing_dir", note)
        return
    for source in source_dir.rglob("*"):
        if not source.is_file():
            continue
        suffix = source.suffix.lower()
        size = source.stat().st_size
        should_copy = (
            suffix == ".md"
            or (suffix == ".csv" and size <= SMALL_CSV_LIMIT)
            or (suffix in {".json", ".txt"} and size <= SMALL_CSV_LIMIT)
            or (suffix in {".png", ".jpg", ".jpeg", ".svg"} and size <= PLOT_LIMIT)
            or source.name in {"split_summary.csv", "safe_stop_report.md", "variant_manifest.csv"}
        )
        destination = destination_dir / source.relative_to(source_dir)
        if should_copy:
            copy_file(source, destination, note)
        else:
            record(source, None, "manifest_only_large_or_binary", note)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_table_from_md(md_path: Path, section_title: str) -> list[list[str]]:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    rows: list[list[str]] = []
    in_section = False
    for line in lines:
        if line.strip() == section_title:
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and line.startswith("|") and not line.startswith("|:") and "---" not in line:
            cells = [cell.strip().strip("`") for cell in line.strip("|").split("|")]
            if cells and cells[0] not in {"Split", "Policy"}:
                rows.append(cells)
    return rows


def build_universal_tables() -> None:
    universal_md = VIS / "universal_dual_head_strategy_latest.md"
    copy_file(
        universal_md,
        PKG / "01_final_recommendation" / "universal_dual_head_strategy_latest.md",
        "final universal valid-minimax report",
    )
    copy_file(
        VIS / "universal_dual_head_all_policy_test_posthoc.csv",
        PKG / "01_final_recommendation" / "universal_dual_head_all_policy_test_posthoc.csv",
        "test-posthoc all-policy grid; diagnostic, not selection source",
    )

    validation_rows = read_table_from_md(universal_md, 'Public-release English note.')
    test_rows = read_table_from_md(universal_md, 'Public-release English note.')
    validation = {
        row[0]: {
            "valid_save_pct": row[1].replace("%", ""),
            "valid_drop_pp": row[2].replace("pp", ""),
            "valid_acc_pct": row[3].replace("%", ""),
            "valid_worst_agent_abs_drop_pp": row[4].replace("pp", ""),
        }
        for row in validation_rows
    }
    test = {
        row[0]: {
            "test_save_pct": row[1].replace("%", ""),
            "test_drop_pp": row[2].replace("pp", ""),
            "test_acc_pct": row[3].replace("%", ""),
        }
        for row in test_rows
    }
    split_summary_rows = []
    for split in ["bottom3", "mid3", "top3"]:
        split_summary_rows.append({"split": split, **validation.get(split, {}), **test.get(split, {})})
    pd.DataFrame(split_summary_rows).to_csv(
        PKG / "06_paper_tables" / "universal_strategy_split_summary.csv",
        index=False,
    )

    per_agent_rows = read_table_from_md(universal_md, "## Per-Agent Test Shift")
    pd.DataFrame(
        [
            {
                "split": row[0],
                "agent": row[1],
                "delta_rate_pp": row[2].replace("pp", ""),
                "save_pct": row[3].replace("%", ""),
                "acc_pct": row[4].replace("%", ""),
            }
            for row in per_agent_rows
        ]
    ).to_csv(PKG / "06_paper_tables" / "universal_strategy_per_agent_test_shift.csv", index=False)

    selected_policy_rows = []
    for split in ["bottom3", "mid3", "top3"]:
        grid = REPORTS / (
            f"per_instance_model_valid3_{split}_no_model_id_strong_reg_"
            "safe_stop_dual_head_retrain/safe_stop_valid_policy_grid.csv"
        )
        if not grid.exists():
            continue
        df = pd.read_csv(grid)
        mask = (
            df["prefix_model"].eq("I_LightGBM_Dense_AF")
            & df["score_mode"].eq("calibrated")
            & df["success_thr"].eq(0.95)
            & df["failure_thr"].eq(0.95)
            & df["min_step"].eq(0)
            & df["consecutive"].eq(1)
        )
        row = df.loc[mask].iloc[0].to_dict()
        selected_policy_rows.append(
            {
                "split": split,
                "selection_source": "validation_only",
                "strategy": "no_model_id + strong_reg + dual-head",
                "prefix_model": "I_LightGBM_Dense_AF",
                "score_mode": "calibrated",
                "success_thr": 0.95,
                "failure_thr": 0.95,
                "min_step": 0,
                "consecutive": 1,
                "valid_save_pct": row["pct_steps_saved"],
                "valid_drop_pp": row["resolve_rate_drop"] * 100.0,
                "valid_acc_pct": row["decision_accuracy"] * 100.0,
                "valid_fn": int(row["false_negatives"]),
                "valid_fp": int(row["false_positives"]),
                "valid_coverage_pct": row["coverage"] * 100.0,
            }
        )
    pd.DataFrame(selected_policy_rows).to_csv(
        PKG / "06_paper_tables" / "final_valid_selected_policy_by_split.csv",
        index=False,
    )

    all_policy = VIS / "universal_dual_head_all_policy_test_posthoc.csv"
    if all_policy.exists():
        df = pd.read_csv(all_policy)
        mask = (
            df["prefix_model"].eq("I_LightGBM_Dense_AF")
            & df["score_mode"].eq("calibrated")
            & df["success_thr"].eq(0.95)
            & df["failure_thr"].eq(0.95)
            & df["min_step"].eq(0)
            & df["consecutive"].eq(1)
        )
        final = df.loc[mask].copy()
        final["selection_source"] = "validation_minimax_locked_before_test_interpretation"
        final.to_csv(PKG / "06_paper_tables" / "final_universal_policy_test_locked.csv", index=False)


def build_core_results() -> None:
    for split in ["bottom3", "mid3", "top3"]:
        source = REPORTS / (
            f"per_instance_model_valid3_{split}_no_model_id_strong_reg_"
            "safe_stop_dual_head_retrain"
        )
        destination = PKG / "03_core_results" / f"{split}_no_model_id_strong_reg_safe_stop_dual_head"
        copy_tree_filtered(source, destination, f"latest {split} no_model_id strong_reg dual-head run")

    for name in [
        "safe_stop_visual_summary.md",
        "strategy_stability_takeaways.md",
        "selected_policy_valid_test.csv",
        "valid_policy_grid_all.csv",
        "best_valid_dual_candidates.csv",
        "dual_head_valid_common_policy_stability.csv",
        "dual_head_common_policy_test_stability.csv",
        "dual_head_candidate_policy_test_check.csv",
        "calibration_summary_all.csv",
        "single_head_common_strategy_stability.csv",
    ]:
        copy_file(
            VIS / name,
            PKG / "03_core_results" / "safe_stop_dual_head_visual_summary" / name,
            "safe-stop visual/common-policy summary",
        )
    copy_tree_filtered(
        VIS / "plots",
        PKG / "03_core_results" / "safe_stop_dual_head_visual_summary" / "plots",
        "safe-stop visual plots",
    )

    for doc in [
        "README.md",
        "PROJECT_PAPER_VALUE_STRATEGY_REPORT.md",
        "RECENT_MODEL_HOLDOUT_EXPERIMENTS_AUDIT.md",
        "SHADOW_VALID_RETRAIN_README.md",
        "MODEL_HOLDOUT_ANSWER_FEATURES_FLOW.md",
        "MODEL_HOLDOUT_ANSWER_FINAL_CONSOLIDATED_REPORT.md",
        "ABLATION_DESIGN.md",
        "feature_dictionary.md",
    ]:
        copy_file(ROOT / doc, PKG / "03_core_results" / "project_docs" / doc, "root project documentation")


def build_diagnostics() -> None:
    for name in ["gpt52codex_source_trace.md", "gpt52codex_raw_data_diagnosis.md"]:
        copy_file(VIS / name, PKG / "02_data_audit" / name, "data-quality audit")

    diagnosis_root = VIS / "problem_diagnosis"
    copy_tree_filtered(
        diagnosis_root,
        PKG / "04_diagnostics" / "problem_diagnosis_summaries",
        "diagnostic reports/plots; large raw CSVs are manifest-only",
    )

    data_quality_rows = [
        {
            "scope": "grouped_json",
            "model_count": 20,
            "meaning": "output_bash_only_trajs_flat_by_model/stats_summary.json; excludes gpt-5-2-codex",
        },
        {
            "scope": "clean_non_v2_grouped_json",
            "model_count": 18,
            "meaning": "grouped JSON minus two mini-v2.0.0 models; matches the intended clean set",
        },
        {
            "scope": "raw_parquet",
            "model_count": 21,
            "meaning": "tool-bash-trajs-flat.parquet; includes patch-only gpt-5-2-codex",
        },
    ]
    pd.DataFrame(data_quality_rows).to_csv(PKG / "06_paper_tables" / "data_quality_model_counts.csv", index=False)


def build_code_snapshot() -> None:
    code_rows = []
    descriptions = {
        "run_all.py": "original end-to-end prefix/feature/training pipeline",
        "model_holdout_shadow_valid_retrain.py": "model-holdout retraining with train/valid/test split controls",
        "safe_stop_dual_head_retrain.py": "dual-head safe-success/safe-failure retraining and valid-selected safe-stop policies",
        "valid_policy_tuning_posthoc.py": "single-head valid policy tuning diagnostics",
        "dual_head_conjunctive_gate_valid_test_posthoc.py": "posthoc conjunctive dual-head gate analysis",
        "plot_safe_stop_dual_head_summary.py": "safe-stop summary tables and plots",
        "plot_dual_head_threshold_agent_curves_all_splits.py": "threshold curves by split/agent",
        "other_model_prior_auc_posthoc.py": "other-model prior AUC baseline",
        "hard_subset_auc_posthoc.py": "hard subset AUC diagnostics",
        "hard_subset_policy_auc_rescue_all_splits_posthoc.py": "hard-subset policy gain sweeps across bottom/mid/top",
        "top3_policy_auc_gain_sweep_fast_posthoc.py": "top3 AUC gain sweep diagnostics",
        "ambiguous_policy_auc_rescue_posthoc.py": "ambiguous-band AUC rescue diagnostics",
        "prefix_builder.py": "prefix table construction",
        "step_builder.py": "step-level trajectory construction",
        "feature_engineer.py": "dense/TF-IDF feature construction",
        "trainer.py": "model training helpers",
        "evaluator.py": "metric/evaluation helpers",
        "config.py": "path and model configuration",
    }
    for source in sorted(ROOT.glob("*.py")):
        destination = PKG / "05_code_manifest" / "code_snapshot" / source.name
        copy_file(source, destination, "top-level Python source snapshot")
        code_rows.append(
            {
                "script": source.name,
                "copied_to": rel(destination, PKG),
                "category": "core" if source.name in descriptions else "support_or_posthoc",
                "description": descriptions.get(source.name, "support/posthoc script retained for reproducibility"),
            }
        )
    for source in [ROOT / "requirements.txt", ROOT / "action_taxonomy.yaml"]:
        copy_file(source, PKG / "05_code_manifest" / "code_snapshot" / source.name, "runtime/config file")
        code_rows.append(
            {
                "script": source.name,
                "copied_to": rel(PKG / "05_code_manifest" / "code_snapshot" / source.name, PKG),
                "category": "runtime_config",
                "description": "runtime dependency/configuration file",
            }
        )
    pd.DataFrame(code_rows).to_csv(PKG / "05_code_manifest" / "code_manifest.csv", index=False)
    write_text(
        PKG / "05_code_manifest" / "code_index.md",
        """# Code Index

This folder is a snapshot of the top-level code used by the experiment line.

## Most Important Scripts

- `safe_stop_dual_head_retrain.py`: current main training/evaluation script for no_model_id + strong_reg + dual-head safe-stop.
- `model_holdout_shadow_valid_retrain.py`: earlier model-holdout retrain and ablation backbone.
- `plot_safe_stop_dual_head_summary.py`: common-policy and visualization summary.
- `dual_head_conjunctive_gate_valid_test_posthoc.py`: diagnostic dual-head gate sweeps.
- `other_model_prior_auc_posthoc.py`, `hard_subset_auc_posthoc.py`, `hard_subset_policy_auc_rescue_all_splits_posthoc.py`: prior-baseline / hard-subset diagnostics.

See `code_manifest.csv` for the full script list and one-line purpose.
""",
    )


def build_quick_start() -> None:
    write_text(
        PKG / "00_quick_start" / "commands.md",
        f"""# Quick Start Commands

Run from:

```bash
cd {ROOT}
```

## Current Final Strategy

Use this result as the paper-facing main policy:

```text
no_model_id + strong_reg + dual-head + calibrated I + s0.95/f0.95 + min0 + k1
```

Important leakage rule:

- Select policy by validation only.
- Treat test-posthoc grids as diagnostic upper bounds, not final-policy selection.

## Re-run Bottom/Mid/Top Safe-Stop Experiments

The three commands below use the same configuration and only change heldout split/output folder.
They include `0.95` thresholds because the final valid-minimax policy uses `s0.95/f0.95`.

```bash
export SWE_MAX_CPU_THREADS=96
export OMP_NUM_THREADS=96
export OPENBLAS_NUM_THREADS=96
export MKL_NUM_THREADS=96
export NUMEXPR_NUM_THREADS=96

python safe_stop_dual_head_retrain.py \\
  --run-name {RUN_NAME} \\
  --holdout-models auto_bottom3 \\
  --max-instances 500 \\
  --split-strategy per_instance_model \\
  --valid-models-per-instance 3 \\
  --output-subdir per_instance_model_valid3_bottom3_no_model_id_strong_reg_safe_stop_dual_head_retrain \\
  --variants i j \\
  --lgbm-preset strong_reg \\
  --mask-train-model-id \\
  --success-thresholds 0.50 0.60 0.70 0.80 0.90 0.95 inf \\
  --failure-thresholds 0.50 0.60 0.70 0.80 0.90 0.95 inf \\
  --policy-min-steps 0 5 10 15 \\
  --consecutive 1 2 3 \\
  --score-modes raw calibrated \\
  --max-cpu-threads 96 \\
  --text-batch-size 4096

python safe_stop_dual_head_retrain.py \\
  --run-name {RUN_NAME} \\
  --holdout-models auto_mid3 \\
  --max-instances 500 \\
  --split-strategy per_instance_model \\
  --valid-models-per-instance 3 \\
  --output-subdir per_instance_model_valid3_mid3_no_model_id_strong_reg_safe_stop_dual_head_retrain \\
  --variants i j \\
  --lgbm-preset strong_reg \\
  --mask-train-model-id \\
  --success-thresholds 0.50 0.60 0.70 0.80 0.90 0.95 inf \\
  --failure-thresholds 0.50 0.60 0.70 0.80 0.90 0.95 inf \\
  --policy-min-steps 0 5 10 15 \\
  --consecutive 1 2 3 \\
  --score-modes raw calibrated \\
  --max-cpu-threads 96 \\
  --text-batch-size 4096

python safe_stop_dual_head_retrain.py \\
  --run-name {RUN_NAME} \\
  --holdout-models auto_top3 \\
  --max-instances 500 \\
  --split-strategy per_instance_model \\
  --valid-models-per-instance 3 \\
  --output-subdir per_instance_model_valid3_top3_no_model_id_strong_reg_safe_stop_dual_head_retrain \\
  --variants i j \\
  --lgbm-preset strong_reg \\
  --mask-train-model-id \\
  --success-thresholds 0.50 0.60 0.70 0.80 0.90 0.95 inf \\
  --failure-thresholds 0.50 0.60 0.70 0.80 0.90 0.95 inf \\
  --policy-min-steps 0 5 10 15 \\
  --consecutive 1 2 3 \\
  --score-modes raw calibrated \\
  --max-cpu-threads 96 \\
  --text-batch-size 4096
```

## Refresh Summary Plots

```bash
python plot_safe_stop_dual_head_summary.py \\
  --reports-root runs/{RUN_NAME}/reports \\
  --output-dir runs/{RUN_NAME}/reports/safe_stop_dual_head_visual_summary
```

## Where To Look First

- `../01_final_recommendation/final_strategy_summary.md`
- `../06_paper_tables/universal_strategy_split_summary.csv`
- `../02_data_audit/gpt52codex_source_trace.md`
- `../99_source_manifests/artifact_manifest.csv`
""",
    )


def build_final_summary() -> None:
    write_text(
        PKG / "01_final_recommendation" / "final_strategy_summary.md",
        """# Final Strategy Summary

## Recommendation

Use the unified valid-minimax policy:

```text
no_model_id + strong_reg + dual-head + calibrated I + s0.95/f0.95 + min0 + k1
```

Meaning:

- `no_model_id`: train/valid/test model ID is masked, reducing direct agent-ID memorization.
- `strong_reg`: stronger LightGBM regularization.
- `dual-head`: separate safe-success and safe-failure heads; a stop decision requires one side to be confident enough.
- `calibrated I`: use calibrated probabilities from `I_LightGBM_Dense_AF`.
- `s0.95/f0.95`: success threshold 0.95 and failure threshold 0.95.
- `min0/k1`: no extra minimum-step gate and one consecutive trigger is enough.

## Current Numbers

From `universal_dual_head_strategy_latest.md`:

- Validation weighted save: `22.83%`
- Validation absolute drop: `0.18pp`
- Validation worst-agent absolute drop: `2.44pp`
- Test weighted save: `23.59%`
- Test absolute drop: `0.21pp`
- Test max per-agent shift: `1.84pp`

## Why This Is The Main Result

- It is selected by validation only, so it is defensible as the locked policy.
- It sacrifices some save rate to keep per-agent resolve-rate drift small.
- It avoids the earlier trap of choosing asymmetric/test-specific thresholds after seeing test.

## What Not To Claim

- Do not claim the test-posthoc `s0.95/f0.90` candidate is the final policy; it is an exploratory upper bound.
- Do not treat `gpt-5-2-codex` as a normal trajectory model; it is patch-only/p0-only in the raw parquet.
- Do not mix the older split-specific selected policies with the final unified valid-minimax policy.
""",
    )
    write_text(
        PKG / "README.md",
        'Public-release English note.',
    )


def build_experiment_index() -> None:
    write_text(
        PKG / "experiment_index.md",
        """# Experiment Index

## Paper-Facing Main Line

- Strategy: `no_model_id + strong_reg + dual-head + calibrated I + s0.95/f0.95 + min0 + k1`
- Selection: validation-only valid-minimax across bottom3/mid3/top3.
- Final report: `01_final_recommendation/universal_dual_head_strategy_latest.md`
- Paper tables: `06_paper_tables/`

## Latest Core Runs

- `03_core_results/bottom3_no_model_id_strong_reg_safe_stop_dual_head/`
- `03_core_results/mid3_no_model_id_strong_reg_safe_stop_dual_head/`
- `03_core_results/top3_no_model_id_strong_reg_safe_stop_dual_head/`

## Diagnostic Lines

- Single-head / dual-head stability: `03_core_results/safe_stop_dual_head_visual_summary/`
- Hard-subset AUC and prior baseline: `04_diagnostics/problem_diagnosis_summaries/other_model_prior_auc/`
- Symmetric thresholds and CIs: `04_diagnostics/problem_diagnosis_summaries/symmetric_threshold_agent_analysis/`
- Conjunctive dual-head gate: `04_diagnostics/problem_diagnosis_summaries/dual_head_conjunctive_gate_posthoc/`
- Data quality / codex anomaly: `02_data_audit/`

## Use / Avoid

- Use `valid-safe` policies for paper claims.
- Use `test-posthoc` only as diagnostics or upper-bound discussion.
- Avoid making final claims from `gpt-5-2-codex` trajectory behavior unless regenerated from real action trajectories.
""",
    )


def build_manifests() -> None:
    report_rows = []
    for source in sorted(REPORTS.rglob("*")):
        if not source.is_file():
            continue
        size = source.stat().st_size
        suffix = source.suffix.lower()
        report_rows.append(
            {
                "source_path": rel(source),
                "size_bytes": size,
                "size_mb": round(size / 1048576, 3),
                "suffix": suffix,
                "artifact_kind": (
                    "model_or_binary"
                    if suffix in {".parquet", ".pkl", ".lgb", ".joblib"}
                    else "report_or_table_or_plot"
                ),
            }
        )
    pd.DataFrame(report_rows).to_csv(PKG / "99_source_manifests" / "all_reports_artifact_manifest.csv", index=False)

    copied_rows = []
    for item in COPIED:
        copied_rows.append(
            {
                "source_path": rel(item.source),
                "destination_path": rel(item.destination, PKG) if item.destination else "",
                "status": item.status,
                "size_bytes": item.size_bytes,
                "size_mb": round(item.size_bytes / 1048576, 3),
                "note": item.note,
            }
        )
    pd.DataFrame(copied_rows).to_csv(PKG / "99_source_manifests" / "artifact_manifest.csv", index=False)

    large_rows = [
        row
        for row in copied_rows
        if row["status"] == "manifest_only_large_or_binary" or row["size_bytes"] > SMALL_CSV_LIMIT
    ]
    with (PKG / "99_source_manifests" / "large_artifacts_not_copied.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_path",
                "destination_path",
                "status",
                "size_bytes",
                "size_mb",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(large_rows)


def main() -> int:
    if PKG.exists():
        shutil.rmtree(PKG)
    ensure_dirs()
    build_universal_tables()
    build_core_results()
    build_diagnostics()
    build_code_snapshot()
    build_quick_start()
    build_final_summary()
    build_experiment_index()
    build_manifests()
    print(f"Wrote organized package: {PKG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
