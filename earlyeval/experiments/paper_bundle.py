from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from earlyeval.checks.preflight import CURRENT_PAPER_FIGURES, CURRENT_PAPER_TABLES
from earlyeval.core.io import ensure_dir, write_json, write_table
from earlyeval.core.paths import load_paths, package_root, resolve_repo_path


@dataclass(frozen=True)
class MaterializedInput:
    kind: str
    name: str
    required: bool
    source: str
    destination: str
    mode: str
    status: str
    size_bytes: int


def _safe_name(index: int, name: str, source: Path) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    suffix = source.suffix if source.is_file() else ""
    if suffix and not cleaned.endswith(suffix):
        cleaned = f"{cleaned}{suffix}"
    return f"{index:04d}_{cleaned}"


def _size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for root, _, files in os.walk(path):
        for file_name in files:
            try:
                total += int((Path(root) / file_name).stat().st_size)
            except OSError:
                pass
    return total


def _materialize(source: Path, destination: Path, mode: str, *, required: bool) -> str:
    if not source.exists():
        return "missing_source" if required else "missing_optional_source"
    if mode == "manifest":
        return "manifest_only"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return "already_linked"
        return "destination_exists"
    if mode == "link":
        destination.symlink_to(source, target_is_directory=source.is_dir())
        return "linked"
    if mode == "copy":
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        return "copied"
    raise ValueError(f"Unsupported materialize mode: {mode}")


def _current_dataset_sources() -> list[tuple[str, Path]]:
    cfg_path = package_root() / "configs" / "earlyeval.yaml"
    if not cfg_path.exists():
        return []
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    rows: list[tuple[str, Path]] = []
    for dataset_name, dataset in (cfg.get("datasets", {}) or {}).items():
        prefix_table = dataset.get("prefix_table")
        if prefix_table:
            rows.append((f"dataset:{dataset_name}:prefix_table", resolve_repo_path(prefix_table)))
        verified_jsonl = dataset.get("verified_jsonl")
        if verified_jsonl:
            rows.append((f"dataset:{dataset_name}:verified_jsonl", resolve_repo_path(verified_jsonl)))
    return rows


def _prefer_data_root(source: Path, *, data_root: Path, repo_root: Path) -> Path:
    if source.exists():
        return source
    try:
        relative = source.relative_to(repo_root)
    except ValueError:
        relative = None
    if relative is not None:
        candidate = data_root / relative
        if candidate.exists():
            return candidate
    if source.as_posix().endswith("swebench_verified/test.jsonl"):
        candidate = data_root / "swe_verify_500" / "offical_answer" / "test.jsonl"
        if candidate.exists():
            return candidate
    return source


def materialize_paper_inputs(
    *,
    paths_config: str | Path | None = None,
    mode: str = "link",
    include_raw_inventory: bool = True,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    if mode not in {"link", "copy", "manifest"}:
        raise ValueError("--mode must be one of: link, copy, manifest")

    paths = load_paths(paths_config)
    paths.ensure_work_dirs()
    manifest_dir = ensure_dir(output_dir or paths.paper_root / "data")

    entries: list[MaterializedInput] = []
    draft = package_root() / "paper" / "icse_submission_draft"
    in_package_sources: list[tuple[str, str, Path]] = [
        ("paper_source", "paper.md", draft / "paper.md"),
        ("paper_source", "paper.tex", draft / "paper.tex"),
        ("paper_source", "DATA_LINEAGE.md", draft / "data" / "DATA_LINEAGE.md"),
        ("paper_source", "PAPER_DATA_INDEX.md", draft / "data" / "PAPER_DATA_INDEX.md"),
        ("paper_source", "refresh_tables.py", draft / "data" / "refresh_tables.py"),
        (
            "experiment_artifacts",
            "earlyeval_lightgbm",
            package_root() / "paper" / "experiments" / "earlyeval_lightgbm",
        ),
        ("config", "earlyeval.yaml", package_root() / "configs" / "earlyeval.yaml"),
    ]
    in_package_sources.extend(
        ("paper_table", name, draft / "data" / name) for name in CURRENT_PAPER_TABLES
    )
    in_package_sources.extend(
        ("paper_figure", name, draft / "figures" / name) for name in CURRENT_PAPER_FIGURES
    )
    for index, (kind, name, source) in enumerate(in_package_sources, start=1):
        status = "in_package" if source.exists() else "missing_source"
        entries.append(
            MaterializedInput(
                kind=kind,
                name=name,
                required=True,
                source=str(source),
                destination=str(source),
                mode="in_package",
                status=status,
                size_bytes=_size(source),
            )
        )

    if include_raw_inventory:
        offset = len(entries) + 1
        for index, (name, source) in enumerate(_current_dataset_sources(), start=offset):
            source = _prefer_data_root(source, data_root=paths.data_root, repo_root=paths.repo_root)
            destination = paths.paper_raw_root / _safe_name(index, name, source)
            status = _materialize(source, destination, mode, required=True)
            entries.append(
                MaterializedInput(
                    kind="dataset_source",
                    name=name,
                    required=True,
                    source=str(source),
                    destination=str(destination),
                    mode=mode,
                    status=status,
                    size_bytes=_size(source),
                )
            )

    frame = pd.DataFrame([asdict(item) for item in entries])
    manifest_path = write_table(frame, manifest_dir / "input_manifest.csv")
    summary = {
        "ok": bool(frame.empty or not frame["status"].eq("missing_source").any()),
        "mode": mode,
        "entries": int(len(frame)),
        "missing_sources": int(frame["status"].eq("missing_source").sum()) if not frame.empty else 0,
        "missing_optional_sources": int(frame["status"].eq("missing_optional_source").sum()) if not frame.empty else 0,
        "manifest": str(manifest_path),
    }
    write_json(manifest_dir / "input_manifest_summary.json", summary)
    _write_readme(manifest_dir / "README.md", summary)
    return summary


def _write_readme(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Paper Data Inputs",
        "",
        "This directory is managed by `python -m earlyeval.cli experiment materialize-paper`.",
        "It tracks the current ICSE draft data package; old RQ-page inputs are not part of this manifest.",
        "",
        f"- mode: `{summary['mode']}`",
        f"- entries: `{summary['entries']}`",
        f"- missing sources: `{summary['missing_sources']}`",
        f"- missing optional sources: `{summary['missing_optional_sources']}`",
        "",
        "`input_manifest.csv` records every paper-facing artifact or raw source, its original path, and its local paper data destination.",
        "",
        "Use `--mode link` for reproducible local paper bundles without duplicating large parquet/model files. Use `--mode copy` only when you intentionally want a self-contained archive.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
