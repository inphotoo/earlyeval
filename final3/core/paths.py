from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return package_root().parent


def _resolve(value: str | Path, *, base: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base or package_root()) / path


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    data_root: Path
    paper_root: Path
    paper_raw_root: Path
    paper_artifact_root: Path
    paper_results_root: Path
    check_root: Path
    output_root: Path
    python_executable: Path | None
    legacy_final_root: Path
    legacy_final2_root: Path
    answer_module_root: Path
    vendor_answer_module_root: Path
    shared_answer_root: Path
    prefix_table: Path
    prefix_table_filtered: Path
    prefix_table_answer_enriched: Path
    step_table: Path
    feature_engineer_with_model: Path

    def ensure_work_dirs(self) -> None:
        for path in (
            self.paper_root,
            self.paper_raw_root,
            self.paper_artifact_root,
            self.paper_results_root,
            self.check_root,
            self.output_root,
        ):
            path.mkdir(parents=True, exist_ok=True)


def _paths_config_path(config: str | Path | None = None) -> Path:
    cfg_path = _resolve(config or "configs/paths.yaml")
    if cfg_path.exists():
        return cfg_path
    if config is None:
        example = _resolve("configs/paths.example.yaml")
        if example.exists():
            return example
    raise FileNotFoundError(f"paths config not found: {cfg_path}")


def load_paths(config: str | Path | None = None) -> ProjectPaths:
    cfg_path = _paths_config_path(config)
    payload: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    root = _resolve(payload.get("repo_root", ".."))
    legacy = payload.get("legacy", {}) or {}
    vendor = payload.get("vendor", {}) or {}
    shared = payload.get("shared_data", {}) or {}
    runtime = payload.get("runtime", {}) or {}

    def p(key: str, default: str) -> Path:
        return _resolve(payload.get(key, default))

    def lp(key: str, default: str | None) -> Path:
        value = legacy.get(key, default)
        if value is None:
            return Path()
        return _resolve(value)

    def vp(key: str, default: str) -> Path:
        return _resolve(vendor.get(key, default))

    def sp(key: str, default: str) -> Path:
        return _resolve(shared.get(key, default))

    return ProjectPaths(
        repo_root=root,
        data_root=p("data_root", "../data"),
        paper_root=p("paper_root", "paper"),
        paper_raw_root=p("paper_raw_root", "paper/data/raw"),
        paper_artifact_root=p("paper_artifact_root", "paper/data/artifacts"),
        paper_results_root=p("paper_results_root", "paper/results"),
        check_root=p("check_root", "paper/checks"),
        output_root=p("output_root", "outputs"),
        python_executable=(
            _resolve(runtime["python_executable"]) if runtime.get("python_executable") else None
        ),
        legacy_final_root=lp("final_root", None),
        legacy_final2_root=lp("final2_root", None),
        answer_module_root=lp(
            "answer_module_root",
            None,
        ),
        vendor_answer_module_root=vp(
            "answer_module_root",
            "final3/vendor/prefix_predict_model_holdout_answer",
        ),
        shared_answer_root=sp(
            "answer_root",
            "../data/prefix_predict_model_holdout_answer/model_holdout_answer_shared",
        ),
        prefix_table=sp(
            "prefix_table",
            "../data/prefix_predict_model_holdout_answer/model_holdout_answer_shared/prefix_table.parquet",
        ),
        prefix_table_filtered=sp(
            "prefix_table_filtered",
            "../data/prefix_predict_model_holdout_answer/model_holdout_answer_shared/prefix_table_filtered.parquet",
        ),
        prefix_table_answer_enriched=sp(
            "prefix_table_answer_enriched",
            "../data/prefix_predict_model_holdout_answer/model_holdout_answer_shared/prefix_table_answer_enriched.parquet",
        ),
        step_table=sp(
            "step_table",
            "../data/prefix_predict_model_holdout_answer/model_holdout_answer_shared/step_table.parquet",
        ),
        feature_engineer_with_model=sp(
            "feature_engineer_with_model",
            "../artifacts/model_holdout_answer_calibrated_full/models/feature_engineer_with_model.pkl",
        ),
    )


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return package_root() / path
