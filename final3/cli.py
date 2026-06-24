from __future__ import annotations

import argparse
import json
from pathlib import Path

from final3.benchmarks.normalize import normalize_file
from final3.checks.preflight import run_preflight
from final3.experiments.paper_bundle import materialize_paper_inputs
from final3.experiments.registry import list_experiments
from final3.experiments.rq_final import run_rq_final_stage
from final3.legacy.wrappers import explain_legacy_entry
from final3.models.dual_head_lightgbm import DualHeadRunSpec, run_dual_head
from final3.models.heavy import describe_heavy_experiments
from final3.pipelines.current_safe_stop import run_current_safe_stop
from final3.policies.apply import apply_policy_to_file
from final3.reports.paper_tables import refresh_paper_tables


def _print_result(payload) -> None:
    """Print command results as stable JSON for scripts and logs."""

    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level final3 command-line parser."""

    parser = argparse.ArgumentParser(
        prog="python -m final3.cli",
        description="Final3 maintainable source entrypoints for SWE-bench safe-stop experiments.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pipeline = sub.add_parser("pipeline", help="Run composed workflows.")
    pipeline_sub = pipeline.add_subparsers(dest="pipeline_name", required=True)
    current = pipeline_sub.add_parser("current-safe-stop", help="Run/apply the current main safe-stop strategy.")
    current.add_argument("--mode", choices=("smoke", "main", "full"), default="smoke")
    current.add_argument("--predictions", type=Path, default=None)
    current.add_argument("--output-dir", type=Path, default=None)
    current.add_argument("--preset", default="current_safe_stop")

    data = sub.add_parser("data", help="Data normalization and audits.")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    normalize = data_sub.add_parser("normalize", help="Normalize raw benchmark records into final3 trajectory contract.")
    normalize.add_argument("--benchmark", required=True, choices=("swebench", "terminalbench", "toolathlon", "generic"))
    normalize.add_argument("--input", required=True, type=Path)
    normalize.add_argument("--output-dir", required=True, type=Path)

    policy = sub.add_parser("policy", help="Apply or inspect safe-stop policies.")
    policy_sub = policy.add_subparsers(dest="policy_command", required=True)
    apply = policy_sub.add_parser("apply", help="Apply a policy preset to a prediction table.")
    apply.add_argument("--preset", default="current_safe_stop")
    apply.add_argument("--preset-config", type=Path, default=None)
    apply.add_argument("--predictions", required=True, type=Path)
    apply.add_argument("--output-dir", required=True, type=Path)

    check = sub.add_parser("check", help="Run preflight checks before experiments.")
    check_sub = check.add_subparsers(dest="check_command", required=True)
    preflight = check_sub.add_parser("preflight", help="Check data, code, config, and Python dependencies.")
    preflight.add_argument("--paths-config", type=Path, default=Path("configs/paths.yaml"))
    preflight.add_argument("--experiment", choices=("all", "main", "paper", "heavy"), default="all")
    preflight.add_argument("--output-dir", type=Path, default=None)

    experiment = sub.add_parser("experiment", help="List or prepare experiment sets owned by final3.")
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)
    exp_list = experiment_sub.add_parser("list", help="List experiment sets and their entrypoints.")
    exp_list.add_argument("--registry", type=Path, default=Path("configs/experiment_registry.yaml"))
    materialize = experiment_sub.add_parser("materialize-paper", help="Put paper inputs under final3/paper/data.")
    materialize.add_argument("--paths-config", type=Path, default=Path("configs/paths.yaml"))
    materialize.add_argument("--mode", choices=("link", "copy", "manifest"), default="link")
    materialize.add_argument("--output-dir", type=Path, default=None)
    materialize.add_argument("--no-raw-inventory", action="store_true")
    rq_final = experiment_sub.add_parser("rq-final", help="Plan or smoke-test the final paper RQ experiment set.")
    rq_final.add_argument(
        "--stage",
        choices=(
            "plan",
            "audit-prefix",
            "make-splits",
            "smoke",
            "lightgbm-main",
            "lightgbm-summary",
            "lightgbm-policy-sweep",
        ),
        default="smoke",
    )
    rq_final.add_argument("--config", type=Path, default=Path("configs/rq_final.yaml"))
    rq_final.add_argument("--output-dir", type=Path, default=None)
    rq_final.add_argument("--datasets", nargs="*", default=None)
    rq_final.add_argument("--max-folds", type=int, default=None)
    rq_final.add_argument("--execute", action="store_true", help="Run heavy stages instead of writing a dry-run plan.")
    rq_final.add_argument("--force", action="store_true", help="Re-run folds even when output markers already exist.")
    rq_final.add_argument(
        "--max-parallel-folds",
        type=int,
        default=1,
        help=(
            "Run up to this many LightGBM folds concurrently in the lightgbm-main stage. "
            "Each parallel fold spawns its own subprocess; you should reduce per-fold "
            "--max-cpu-threads accordingly so total CPU usage stays sane."
        ),
    )

    train = sub.add_parser("train", help="Training entrypoints. Heavy jobs are dry-run unless --execute is given.")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    dual = train_sub.add_parser("dual-head", help="Build or execute the current LightGBM dual-head command.")
    dual.add_argument("--run-name", default="model_holdout_answer_calibrated_full")
    dual.add_argument("--prefix-table", type=Path, default=None)
    dual.add_argument("--verified-jsonl", type=Path, default=None)
    dual.add_argument("--feature-engineer-path", type=Path, default=None)
    dual.add_argument("--holdout-models", default="auto_mid3")
    dual.add_argument("--max-instances", type=int, default=500)
    dual.add_argument("--output-subdir", default="final3_current_safe_stop_dual_head")
    dual.add_argument("--max-cpu-threads", type=int, default=8)
    dual.add_argument("--no-low-memory", action="store_true")
    dual.add_argument("--execute", action="store_true")
    train_sub.add_parser("list-heavy", help="List heavy opt-in experiments.")

    report = sub.add_parser("report", help="Paper and diagnostic report entrypoints.")
    report_sub = report.add_subparsers(dest="report_command", required=True)
    paper_tables = report_sub.add_parser("paper-tables", help="Refresh current ICSE draft paper tables.")
    paper_tables.add_argument("--output-dir", type=Path, default=None)

    legacy = sub.add_parser("legacy", help="Show migration help for old entrypoints.")
    legacy_sub = legacy.add_subparsers(dest="legacy_command", required=True)
    explain = legacy_sub.add_parser("explain", help="Explain replacement command for an old script.")
    explain.add_argument("name")
    return parser


def _dispatch(args: argparse.Namespace) -> int:
    """Dispatch parsed CLI arguments to the corresponding implementation."""

    if args.command == "pipeline" and args.pipeline_name == "current-safe-stop":
        _print_result(
            run_current_safe_stop(
                mode=args.mode,
                predictions=args.predictions,
                output_dir=args.output_dir,
                preset=args.preset,
            )
        )
        return 0
    if args.command == "data" and args.data_command == "normalize":
        _print_result(normalize_file(benchmark=args.benchmark, input_path=args.input, output_dir=args.output_dir))
        return 0
    if args.command == "policy" and args.policy_command == "apply":
        _print_result(
            apply_policy_to_file(
                predictions=args.predictions,
                output_dir=args.output_dir,
                preset=args.preset,
                preset_config=args.preset_config,
            )
        )
        return 0
    if args.command == "check" and args.check_command == "preflight":
        _print_result(
            run_preflight(
                paths_config=args.paths_config,
                experiment=args.experiment,
                output_dir=args.output_dir,
            )
        )
        return 0
    if args.command == "experiment" and args.experiment_command == "list":
        _print_result(list_experiments(args.registry))
        return 0
    if args.command == "experiment" and args.experiment_command == "materialize-paper":
        _print_result(
            materialize_paper_inputs(
                paths_config=args.paths_config,
                mode=args.mode,
                include_raw_inventory=not args.no_raw_inventory,
                output_dir=args.output_dir,
            )
        )
        return 0
    if args.command == "experiment" and args.experiment_command == "rq-final":
        _print_result(
            run_rq_final_stage(
                stage=args.stage,
                config=args.config,
                output_dir=args.output_dir,
                datasets=args.datasets,
                max_folds=args.max_folds,
                execute=args.execute,
                force=args.force,
                max_parallel_folds=args.max_parallel_folds,
            )
        )
        return 0
    if args.command == "train" and args.train_command == "dual-head":
        spec = DualHeadRunSpec(
            run_name=args.run_name,
            prefix_table=args.prefix_table,
            verified_jsonl=args.verified_jsonl,
            feature_engineer_path=args.feature_engineer_path,
            holdout_models=args.holdout_models,
            max_instances=args.max_instances,
            output_subdir=args.output_subdir,
            max_cpu_threads=args.max_cpu_threads,
            low_memory=not args.no_low_memory,
        )
        _print_result(run_dual_head(spec, execute=args.execute))
        return 0
    if args.command == "train" and args.train_command == "list-heavy":
        _print_result(describe_heavy_experiments())
        return 0
    if args.command == "report" and args.report_command == "paper-tables":
        _print_result(refresh_paper_tables(output_dir=args.output_dir))
        return 0
    if args.command == "legacy" and args.legacy_command == "explain":
        print(explain_legacy_entry(args.name))
        return 0

    raise RuntimeError("Unhandled command")


def main(argv: list[str] | None = None) -> int:
    """Run the final3 CLI and convert expected user errors to one-line messages."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
