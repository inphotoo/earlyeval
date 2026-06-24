from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from earlyeval.core.io import ensure_dir, write_json, write_table
from earlyeval.core.paths import ProjectPaths, load_paths, package_root, resolve_repo_path


@dataclass(frozen=True)
class CheckRow:
    area: str
    name: str
    status: str
    required: bool
    path: str = ""
    detail: str = ""


def _path_row(area: str, name: str, path: Path, *, required: bool = True) -> CheckRow:
    exists = path.exists()
    if exists:
        status = "ok"
        detail = "exists"
    else:
        status = "missing" if required else "warning"
        detail = "not found"
    return CheckRow(area=area, name=name, status=status, required=required, path=str(path), detail=detail)


def _module_row(name: str, *, required: bool = True, python_executable: Path | None = None) -> CheckRow:
    if python_executable is not None and python_executable.exists():
        proc = subprocess.run(
            [str(python_executable), "-c", f"import {name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        found = proc.returncode == 0
        detail = f"importable via {python_executable}" if found else f"not importable via {python_executable}"
    else:
        found = importlib.util.find_spec(name) is not None
        detail = f"importable via {sys.executable}" if found else f"not importable via {sys.executable}"
    return CheckRow(
        area="python",
        name=name,
        status="ok" if found else ("missing" if required else "warning"),
        required=required,
        detail=detail,
    )


CURRENT_PAPER_TABLES = [
    "table_ablation_balanced4_aggregate.csv",
    "table_ablation_balanced4_min_step_k.csv",
    "table_ablation_balanced4_per_fold.csv",
    "table_ablation_balanced4_policy_mode.csv",
    "table_ablation_balanced4_score_mode.csv",
    "table_ablation_default_reg_sweverify.csv",
    "table_ablation_fine_grained_sweverify.csv",
    "table_architecture_compare_sweverify.csv",
    "table_architecture_run_status.csv",
    "table_llm_logit_lomo_representative_sweverify.csv",
    "table_llm_logit_lomo_threshold_summary_sweverify.csv",
    "table_lr_tfidf_sweverify.csv",
    "table_main_aggregate.csv",
    "table_main_per_fold.csv",
    "table_main_valid_accuracy_frontier.csv",
    "table_min_step_k_sweverify.csv",
    "table_policy_mode_sweverify.csv",
    "table_prefix_audit_summary.csv",
    "table_rank_change_sweverify.csv",
    "table_robustness.csv",
    "table_robustness_loo.csv",
    "table_score_mode_sweverify.csv",
    "table_split_check_counts.csv",
    "table_stop_signal_decision_lift_by_prefix_sweverify.csv",
    "table_stop_signal_decision_lift_by_trajectory_sweverify.csv",
    "table_stop_signal_stop_composition_sweverify.csv",
    "table_success_examples_sweverify.csv",
    "table_token_by_decision_round_sweverify.csv",
    "table_token_summary_sweverify.csv",
]

CURRENT_PAPER_FIGURES = [
    "lr_tfidf_selected_sweverify_aggregate.png",
    "lr_tfidf_valid_accuracy_frontier.png",
    "main_sweverify_selected_per_model.png",
    "main_sweverify_valid_accuracy_frontier.png",
    "main_sweverify_valid_accuracy_per_model_heatmap.png",
    "opposite_complement_075_095_aggregate_test_frontier.png",
    "opposite_complement_075_095_per_model_test_resolve_change_by_threshold.png",
    "opposite_complement_075_095_per_model_test_resolve_change_heatmap.png",
    "robustness_process_valid_accuracy_frontier.png",
    "robustness_rich_af_gold_valid_accuracy_frontier.png",
    "robustness_selected_test_metrics.png",
    "split_prefix_audit_summary.png",
]


def _current_paper_paths() -> list[tuple[str, Path, bool]]:
    root = package_root()
    draft = root / "paper" / "icse_submission_draft"
    data = draft / "data"
    figures = draft / "figures"
    out: list[tuple[str, Path, bool]] = [
        ("paper_source.paper_md", draft / "paper.md", True),
        ("paper_source.paper_tex", draft / "paper.tex", True),
        ("paper_source.data_lineage", data / "DATA_LINEAGE.md", True),
        ("paper_source.data_index", data / "PAPER_DATA_INDEX.md", True),
        ("paper_source.refresh_tables", data / "refresh_tables.py", True),
        (
            "experiment.completion_status",
            root / "paper" / "experiments" / "earlyeval_lightgbm" / "reporting_detail"
            / "experiment_completion_status.csv",
            True,
        ),
    ]
    out.extend((f"paper_table.{name}", data / name, True) for name in CURRENT_PAPER_TABLES)
    out.extend((f"paper_figure.{name}", figures / name, True) for name in CURRENT_PAPER_FIGURES)

    earlyeval = root / "configs" / "earlyeval.yaml"
    out.append(("config.earlyeval", earlyeval, True))
    if earlyeval.exists():
        cfg = yaml.safe_load(earlyeval.read_text(encoding="utf-8")) or {}
        for dataset_name, dataset in (cfg.get("datasets", {}) or {}).items():
            prefix_table = dataset.get("prefix_table")
            if prefix_table:
                out.append((f"dataset.{dataset_name}.prefix_table", resolve_repo_path(prefix_table), True))
            verified_jsonl = dataset.get("verified_jsonl")
            if verified_jsonl:
                out.append((f"dataset.{dataset_name}.verified_jsonl", resolve_repo_path(verified_jsonl), True))
    return out


def _core_path_rows(paths: ProjectPaths) -> list[CheckRow]:
    return [
        _path_row("root", "repo_root", paths.repo_root),
        _path_row("root", "data_root", paths.data_root),
        _path_row("root", "paper_root", paths.paper_root, required=False),
        _path_row(
            "python",
            "runtime_python",
            paths.python_executable or Path(sys.executable),
            required=paths.python_executable is not None,
        ),
        _path_row("data", "shared_answer_root", paths.shared_answer_root),
        _path_row("data", "prefix_table", paths.prefix_table),
        _path_row("data", "prefix_table_filtered", paths.prefix_table_filtered),
        _path_row("data", "prefix_table_answer_enriched", paths.prefix_table_answer_enriched),
        _path_row("data", "step_table", paths.step_table),
        _path_row(
            "code",
            "current_paper_table_refresher",
            package_root() / "paper" / "icse_submission_draft" / "data" / "refresh_tables.py",
        ),
        _path_row("code", "vendored_answer_module", paths.vendor_answer_module_root),
        _path_row("code", "legacy_answer_module_reference", paths.answer_module_root, required=False),
    ]


def run_preflight(
    *,
    paths_config: str | Path | None = None,
    experiment: str = "all",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    import pandas as pd

    paths = load_paths(paths_config)
    paths.ensure_work_dirs()
    out_dir = ensure_dir(output_dir or paths.check_root / "preflight")

    rows: list[CheckRow] = []
    rows.extend(_core_path_rows(paths))
    rows.append(_path_row("config", "paths_config", Path(paths_config) if paths_config else package_root() / "configs" / "paths.yaml"))
    for module_name, required in (
        ("yaml", True),
        ("pandas", True),
        ("numpy", True),
        ("pyarrow", True),
        ("sklearn", True),
        ("lightgbm", experiment in {"all", "main", "heavy"}),
        ("matplotlib", experiment in {"all", "paper"}),
    ):
        rows.append(_module_row(module_name, required=required, python_executable=paths.python_executable))

    if experiment in {"all", "paper"}:
        for name, path, required in _current_paper_paths():
            rows.append(_path_row("paper_current", name, path, required=required))

    frame = pd.DataFrame([asdict(row) for row in rows])
    failed = frame[(frame["required"]) & (frame["status"] != "ok")]
    warnings = frame[(~frame["required"]) & (frame["status"] != "ok")]
    summary = {
        "ok": bool(failed.empty),
        "experiment": experiment,
        "checks": int(len(frame)),
        "failed": int(len(failed)),
        "warnings": int(len(warnings)),
        "output_dir": str(out_dir),
    }
    write_table(frame, out_dir / "preflight_checks.csv")
    write_json(out_dir / "preflight_summary.json", summary)
    _write_markdown(out_dir / "preflight_report.md", summary, frame)
    return summary


def _write_markdown(path: Path, summary: dict[str, Any], frame) -> None:
    lines = [
        "# EarlyEval Preflight Report",
        "",
        f"- ok: `{summary['ok']}`",
        f"- experiment: `{summary['experiment']}`",
        f"- checks: `{summary['checks']}`",
        f"- failed: `{summary['failed']}`",
        f"- warnings: `{summary['warnings']}`",
        "",
        "| Area | Name | Status | Required | Path | Detail |",
        "|:--|:--|:--|:--:|:--|:--|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['area']} | {row['name']} | {row['status']} | {bool(row['required'])} | "
            f"`{row['path']}` | {row['detail']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
