from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from final3.core.io import ensure_dir, write_json, write_table
from final3.experiments.rq_final import (
    RqFinalConfig,
    _aggregate_selected_policy,
    _default_output_dir,
    _eligible_lightgbm_folds,
    _excluded_models_from_config,
    _fit_feature_engineer_on_train,
    _legacy_trainer_path,
    _markdown_table,
    _resolve_project_path,
    _safe_label_min_step,
    _shared_feature_engineer_path,
    load_rq_final_config,
)


PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "feature_groups": {
        "description": (
            "Main SWEVerify feature-group ablation: I/J plus no-task, no-task-prompt-TFIDF, "
            "no-gold, and no-task+no-gold variants."
        ),
        "variants": [
            "i",
            "j",
            "no_task_signal",
            "no_task_tfidf",
            "no_gold_answer",
            "no_task_signal_no_gold_answer",
        ],
        "lgbm_preset": "strong_reg",
        "mask_train_model_id": True,
    },
    "component_with_model_id": {
        "description": "Component ablation for model-id masking: train keeps model_id visible.",
        "variants": ["i"],
        "lgbm_preset": "strong_reg",
        "mask_train_model_id": False,
    },
    "component_default_reg": {
        "description": "Component ablation for regularization: default LightGBM preset with model_id masked.",
        "variants": ["i"],
        "lgbm_preset": "default",
        "mask_train_model_id": True,
    },
    "fine_grained_process": {
        "description": (
            "Fine-grained process-feature ablation: remove action, feedback, thought, "
            "or task/model/gold signals from the Dense+AF+Thought base."
        ),
        "variants": ["no_feedback", "no_action", "no_thought", "process_only"],
        "lgbm_preset": "strong_reg",
        "mask_train_model_id": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final RQ SWEVerify LightGBM ablations.")
    parser.add_argument("--config", type=Path, default=Path("configs/rq_final.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-subdir", default="sweverify_lightgbm_ablation")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument(
        "--sample-folds",
        type=int,
        default=None,
        help="Randomly sample this many folds from the filtered eligible fold list before running.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--test-models", nargs="*", default=None)
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["feature_groups", "component_with_model_id", "component_default_reg"],
        choices=tuple(PROFILE_SPECS),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Override variants for all selected profiles. Useful for smoke, e.g. --variants i no_task_tfidf.",
    )
    parser.add_argument("--max-instances", type=int, default=500)
    parser.add_argument(
        "--safe-label-min-step",
        type=int,
        default=None,
        help=(
            "Override the safe-stop label min-step. When omitted, falls back "
            "to configs/rq_final.yaml main_model.safe_label_min_step."
        ),
    )
    parser.add_argument("--policy-min-steps", nargs="+", type=int, default=[0, 5, 10])
    parser.add_argument("--consecutive", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--success-thresholds", nargs="+", type=float, default=[0.80, 0.90, 0.95])
    parser.add_argument("--failure-thresholds", nargs="+", type=float, default=[0.80, 0.90, 0.95])
    parser.add_argument("--score-modes", nargs="+", choices=("raw", "calibrated"), default=["raw", "calibrated"])
    parser.add_argument("--max-valid-abs-drop-pp", type=float, default=2.0)
    parser.add_argument("--min-valid-decision-acc", type=float, default=0.90)
    parser.add_argument("--fallback-min-save-pct", type=float, default=0.0)
    parser.add_argument("--smoke-trajectories-per-split", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--fit-feature-engineer-on-train",
        action="store_true",
        help=(
            "Override the yaml main_model.feature_engineer_fit_on_train flag "
            "and force every ablation fold to fit a fresh FeatureEngineer "
            "on its own train split. Strict no-leak but ~5-10x slower."
        ),
    )
    parser.add_argument(
        "--max-parallel-folds",
        type=int,
        default=1,
        help=(
            "Run up to this many ablation fold subprocesses concurrently. "
            "Each parallel fold spawns its own LightGBM training subprocess; "
            "set --threads to a smaller value when raising this so total "
            "CPU usage stays bounded."
        ),
    )
    parser.add_argument("--no-low-memory", action="store_true")
    return parser.parse_args()


def _python_executable(cfg: RqFinalConfig) -> Path:
    paths_cfg = yaml.safe_load(_resolve_project_path("configs/paths.yaml").read_text(encoding="utf-8")) or {}
    runtime = paths_cfg.get("runtime") or {}
    value = runtime.get("python_executable") or (cfg.payload.get("runtime") or {}).get("python_executable") or sys.executable
    path = Path(str(value))
    return path if path.is_absolute() else _resolve_project_path(path)


def _profile_variants(profile: str, override: list[str] | None) -> list[str]:
    if override:
        return [str(item) for item in override]
    return [str(item) for item in PROFILE_SPECS[profile]["variants"]]


def _ablation_command(
    cfg: RqFinalConfig,
    *,
    profile: str,
    test_model: str,
    fold_output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    dataset = (cfg.payload.get("datasets") or {})["sweverify"]
    main = cfg.payload.get("main_model") or {}
    resources = cfg.payload.get("resources") or {}
    spec = PROFILE_SPECS[profile]
    threads = int(args.threads or resources.get("lightgbm_num_threads", resources.get("max_cpu_threads", 8)))
    excluded_train_models = sorted(_excluded_models_from_config(cfg) - {test_model})
    command = [
        str(_python_executable(cfg)),
        str(_legacy_trainer_path(cfg)),
        "--run-name",
        str(main.get("run_name", "model_holdout_answer_calibrated_full")),
        "--prefix-table",
        str(_resolve_project_path(dataset["prefix_table"])),
        "--verified-jsonl",
        str(_resolve_project_path(dataset["verified_jsonl"])),
        "--feature-engineer-path",
        str(_shared_feature_engineer_path()),
        "--holdout-models",
        test_model,
        "--max-instances",
        str(args.max_instances),
        "--split-strategy",
        "per_instance_model",
        "--seed",
        str(cfg.seed),
        "--output-subdir",
        str(fold_output_dir.resolve()),
        "--variants",
        *_profile_variants(profile, args.variants),
        "--lgbm-preset",
        str(spec["lgbm_preset"]),
        "--safe-label-min-step",
        str(args.safe_label_min_step if args.safe_label_min_step is not None else _safe_label_min_step(cfg)),
        "--policy-min-steps",
        *(str(value) for value in args.policy_min_steps),
        "--consecutive",
        *(str(value) for value in args.consecutive),
        "--success-thresholds",
        *(str(value) for value in args.success_thresholds),
        "--failure-thresholds",
        *(str(value) for value in args.failure_thresholds),
        "--score-modes",
        *(str(value) for value in args.score_modes),
        "--max-valid-abs-drop-pp",
        str(args.max_valid_abs_drop_pp),
        "--min-valid-decision-acc",
        str(args.min_valid_decision_acc),
        "--fallback-min-save-pct",
        str(args.fallback_min_save_pct),
        "--max-cpu-threads",
        str(threads),
    ]
    if not args.no_low_memory:
        command.append("--low-memory")
    if bool(spec["mask_train_model_id"]):
        command.append("--mask-train-model-id")
    if excluded_train_models:
        command.append("--exclude-train-models")
        command.extend(excluded_train_models)
    # Per-fold fit-on-train is opt-in; default is the shared pkl path.
    fit_on_train = bool(getattr(args, "fit_feature_engineer_on_train", False)) or _fit_feature_engineer_on_train(cfg)
    if fit_on_train:
        command.append("--fit-feature-engineer-on-train")
        ablation_run_dir = fold_output_dir.parent.parent.parent
        command.extend(["--ram-peak-lock-path", str((ablation_run_dir / "ram_peak.lock").resolve())])
    if int(args.smoke_trajectories_per_split) > 0:
        command.extend(["--smoke-trajectories-per-split", str(int(args.smoke_trajectories_per_split))])
    return command


def _selected_folds(cfg: RqFinalConfig, args: argparse.Namespace) -> list[dict[str, Any]]:
    folds = [row for row in _eligible_lightgbm_folds(cfg, dataset="sweverify") if row["eligible"]]
    if args.test_models:
        wanted = set(str(item) for item in args.test_models)
        folds = [row for row in folds if str(row["test_model"]) in wanted or str(row["fold_id"]) in wanted]
    elif args.sample_folds is not None:
        sample_n = int(args.sample_folds)
        if sample_n <= 0:
            raise ValueError("--sample-folds must be positive when provided.")
        rng = random.Random(int(args.sample_seed))
        folds = list(folds)
        rng.shuffle(folds)
        folds = folds[: min(sample_n, len(folds))]
    if args.max_folds is not None:
        folds = folds[: int(args.max_folds)]
    return folds


def _build_plan(cfg: RqFinalConfig, args: argparse.Namespace, run_dir: Path) -> tuple[list[dict[str, Any]], list[list[str]]]:
    rows: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    folds = _selected_folds(cfg, args)
    for profile in args.profiles:
        profile_dir = run_dir / profile
        logs_dir = ensure_dir(profile_dir / "logs")
        for order, fold in enumerate(folds, start=1):
            fold_dir = profile_dir / "folds" / str(fold["fold_id"])
            command = _ablation_command(
                cfg,
                profile=profile,
                test_model=str(fold["test_model"]),
                fold_output_dir=fold_dir,
                args=args,
            )
            commands.append(command)
            marker = fold_dir / "safe_stop_test_selected.csv"
            rows.append(
                {
                    "profile": profile,
                    "profile_description": PROFILE_SPECS[profile]["description"],
                    "order": order,
                    "fold_id": fold["fold_id"],
                    "test_model": fold["test_model"],
                    "trajectories": fold["trajectories"],
                    "variants": " ".join(_profile_variants(profile, args.variants)),
                    "lgbm_preset": PROFILE_SPECS[profile]["lgbm_preset"],
                    "mask_train_model_id": bool(PROFILE_SPECS[profile]["mask_train_model_id"]),
                    "output_dir": str(fold_dir),
                    "log": str(logs_dir / f"{fold['fold_id']}.log"),
                    "status": "skipped_existing" if marker.exists() and not args.force else "pending",
                    "command": " ".join(command),
                }
            )
    return rows, commands


def _fmt_float(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return ""


def _summarize(run_dir: Path) -> dict[str, Any]:
    import pandas as pd

    test_rows = []
    valid_rows = []
    calibration_rows = []
    for profile_dir in sorted(path for path in run_dir.iterdir() if path.is_dir() and path.name != "summary"):
        profile = profile_dir.name
        for success_marker in sorted((profile_dir / "folds").glob("*/_SUCCESS")):
            fold_dir = success_marker.parent
            fold_id = fold_dir.name
            test_path = fold_dir / "safe_stop_test_selected.csv"
            valid_path = fold_dir / "safe_stop_selected_policies.csv"
            calibration_path = fold_dir / "safe_stop_calibration_summary.csv"
            if test_path.exists():
                frame = pd.read_csv(test_path)
                frame.insert(0, "profile", profile)
                frame.insert(1, "fold_id", fold_id)
                frame.insert(2, "test_model", fold_id)
                test_rows.append(frame)
            if valid_path.exists():
                frame = pd.read_csv(valid_path)
                frame.insert(0, "profile", profile)
                frame.insert(1, "fold_id", fold_id)
                frame.insert(2, "test_model", fold_id)
                valid_rows.append(frame)
            if calibration_path.exists():
                frame = pd.read_csv(calibration_path)
                frame.insert(0, "profile", profile)
                frame.insert(1, "fold_id", fold_id)
                frame.insert(2, "test_model", fold_id)
                calibration_rows.append(frame)

    summary_dir = ensure_dir(run_dir / "summary")
    test = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame()
    valid = pd.concat(valid_rows, ignore_index=True) if valid_rows else pd.DataFrame()
    calibration = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    write_table(test, summary_dir / "per_fold_test_selected.csv")
    write_table(valid, summary_dir / "per_fold_valid_selected.csv")
    write_table(calibration, summary_dir / "per_fold_calibration.csv")

    aggregate_rows = []
    if not test.empty:
        for keys, part in test.groupby(["profile", "prefix_model", "score_mode"], sort=True):
            profile, predictor, score_mode = keys
            aggregate = _aggregate_selected_policy(part)
            aggregate_rows.append(
                {
                    "profile": profile,
                    "predictor": predictor,
                    "score_mode": score_mode,
                    **aggregate,
                }
            )
    aggregate_frame = pd.DataFrame(aggregate_rows)
    write_table(aggregate_frame, summary_dir / "aggregate_by_profile_predictor.csv")

    display = []
    if not aggregate_frame.empty:
        for row in aggregate_frame.sort_values(["profile", "score_mode", "step_save_pct"], ascending=[True, True, False]).to_dict("records"):
            display.append(
                {
                    "profile": row["profile"],
                    "predictor": row["predictor"],
                    "score": row["score_mode"],
                    "folds": int(row["folds"]),
                    "save_pct": _fmt_float(row["step_save_pct"]),
                    "acc_pct": _fmt_float(row["decision_accuracy_pct"]),
                    "coverage_pct": _fmt_float(row["coverage_pct"]),
                    "resolve_change_pp": f"{float(row['resolve_rate_change_pp']):+.2f}",
                }
            )
    lines = [
        "# SWEVerify LightGBM Ablation",
        "",
        f"- run_dir: `{run_dir}`",
        f"- completed selected rows: `{len(test)}`",
        "",
        "## Outputs",
        "",
        "- `per_fold_test_selected.csv`: validation-selected policy applied to held-out test for each fold/profile/predictor.",
        "- `per_fold_valid_selected.csv`: selected validation policy rows.",
        "- `aggregate_by_profile_predictor.csv`: count-weighted aggregate across completed folds.",
        "- Each fold keeps `valid_predictions_safe_stop.parquet` and `test_predictions_safe_stop.parquet` for later analysis.",
        "",
    ]
    if display:
        lines.extend(["## Aggregate", ""])
        lines.extend(_markdown_table(display, ["profile", "predictor", "score", "folds", "save_pct", "acc_pct", "coverage_pct", "resolve_change_pp"]))
        lines.append("")
    (summary_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "ok": True,
        "run_dir": str(run_dir),
        "summary_dir": str(summary_dir),
        "completed_profiles": sorted(test["profile"].unique().tolist()) if not test.empty else [],
        "completed_folds": int(test[["profile", "fold_id"]].drop_duplicates().shape[0]) if not test.empty else 0,
        "aggregate_rows": int(len(aggregate_frame)),
    }
    write_json(summary_dir / "summary_manifest.json", payload)
    return payload


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_rq_final_config(args.config)
    out = ensure_dir(args.output_dir or _default_output_dir(cfg, "rq_final_lightgbm_17"))
    run_dir = ensure_dir(out / "ablations" / "sweverify" / args.run_subdir)
    rows, commands = _build_plan(cfg, args, run_dir)
    write_table(__import__("pandas").DataFrame(rows), run_dir / "command_index.csv")
    write_json(
        run_dir / "run_manifest.json",
        {
            "ok": True,
            "config": str(cfg.path),
            "run_dir": str(run_dir),
            "execute": bool(args.execute),
            "force": bool(args.force),
            "profiles": list(args.profiles),
            "profile_specs": {profile: PROFILE_SPECS[profile] for profile in args.profiles},
            "folds": len(_selected_folds(cfg, args)),
            "commands": len(rows),
            "max_folds": args.max_folds,
            "sample_folds": args.sample_folds,
            "sample_seed": int(args.sample_seed),
            "selected_test_models": [str(row["test_model"]) for row in _selected_folds(cfg, args)],
            "test_models": args.test_models,
            "smoke_trajectories_per_split": int(args.smoke_trajectories_per_split),
            "serial": True,
            "heavy_training": True,
        },
    )
    if not args.execute:
        return {
            "ok": True,
            "execute": False,
            "run_dir": str(run_dir),
            "commands": len(rows),
            "command_index": str(run_dir / "command_index.csv"),
            "note": "Dry-run only. Add --execute to run ablation folds serially.",
        }

    skipped = 0
    pending: list[tuple[dict[str, Any], list[str]]] = []
    for row, command in zip(rows, commands):
        fold_dir = Path(row["output_dir"])
        marker = fold_dir / "safe_stop_test_selected.csv"
        if marker.exists() and not args.force:
            skipped += 1
            if not (fold_dir / "_SUCCESS").exists():
                (fold_dir / "_SUCCESS").write_text("completed from existing marker\n", encoding="utf-8")
            continue
        pending.append((row, command))

    completed = 0
    failed: list[dict[str, Any]] = []

    def _run_one(item: tuple[dict[str, Any], list[str]]) -> tuple[dict[str, Any], int]:
        row, command = item
        fold_dir = Path(row["output_dir"])
        log_path = Path(row["log"])
        fold_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write("COMMAND:\n")
            log.write(" ".join(command) + "\n\n")
            log.flush()
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode == 0:
            (fold_dir / "_SUCCESS").write_text("completed\n", encoding="utf-8")
        return row, int(proc.returncode)

    def _record(row: dict[str, Any], rc: int) -> None:
        nonlocal completed
        if rc == 0:
            completed += 1
            return
        failed.append(
            {
                "profile": row["profile"],
                "fold_id": row["fold_id"],
                "test_model": row["test_model"],
                "returncode": rc,
                "log": str(row["log"]),
            }
        )
        print(
            f"[ablation] {row['profile']}/{row['fold_id']} failed (rc={rc}); "
            f"continuing. Log: {row['log']}",
            flush=True,
        )

    parallelism = max(1, int(args.max_parallel_folds))
    if parallelism <= 1:
        for item in pending:
            row_back, rc = _run_one(item)
            _record(row_back, rc)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(
            f"[ablation] running {len(pending)} fold(s) with parallelism={parallelism}.",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = [executor.submit(_run_one, item) for item in pending]
            for fut in as_completed(futures):
                row_back, rc = fut.result()
                _record(row_back, rc)
    summary = _summarize(run_dir)
    payload = {
        "ok": not failed,
        "execute": True,
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "max_parallel_folds": parallelism,
        "run_dir": str(run_dir),
        "command_index": str(run_dir / "command_index.csv"),
        "summary": summary,
    }
    write_json(run_dir / "execution_summary.json", payload)
    return payload


def main() -> int:
    args = parse_args()
    result = run_ablation(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
