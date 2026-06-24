from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from earlyeval.core.io import ensure_dir, write_json, write_table
from earlyeval.core.paths import load_paths, package_root


# Runtime source code is expected to live inside earlyeval/ and earlyeval/vendor/.
# Historical artifact paths are intentionally not treated as code dependencies.
EXTERNAL_CODE_PATTERNS: tuple[str, ...] = ()

TEXT_SUFFIXES = {
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".md",
    ".txt",
    ".toml",
    ".json",
}


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(package_root().resolve()))
    except ValueError:
        return str(path)


def _scan_external_code_refs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skip_parts = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        "manifests",
        "paper",
        "outputs",
    }
    for path in package_root().rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_parts for part in path.relative_to(package_root()).parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in EXTERNAL_CODE_PATTERNS:
            if pattern not in text:
                continue
            line_numbers = [
                index
                for index, line in enumerate(text.splitlines(), start=1)
                if pattern in line
            ]
            rel_path = _rel(path)
            if rel_path in {"earlyeval/checks/portability_audit.py"}:
                continue
            status = "historical_optional" if rel_path.startswith("scripts/run_earlyeval_09_") else "review"
            if rel_path.startswith("configs/") and "experiment_registry" in rel_path:
                status = "historical_result_reference"
            if rel_path in {"configs/paths.yaml", "configs/paths.example.yaml", "earlyeval/core/paths.py"}:
                status = "external_artifact_reference"
            rows.append(
                {
                    "path": rel_path,
                    "pattern": pattern,
                    "lines": ",".join(str(item) for item in line_numbers),
                    "status": status,
                }
            )
    return rows


def _scan_symlinks(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*")):
        if not path.is_symlink():
            continue
        target_raw = os.readlink(path)
        resolved = path.resolve(strict=False)
        rows.append(
            {
                "path": _rel(path),
                "target": target_raw,
                "resolved": str(resolved),
                "exists": resolved.exists(),
                "size_bytes": int(resolved.stat().st_size) if resolved.exists() and resolved.is_file() else "",
            }
        )
    return rows


def _external_artifact_rows(config_path: Path, *, hash_large: bool, hash_limit_mb: int) -> list[dict[str, Any]]:
    paths = load_paths(config_path)
    candidates = [
        ("shared_prefix_table", paths.prefix_table),
        ("shared_prefix_table_filtered", paths.prefix_table_filtered),
        ("shared_prefix_table_answer_enriched", paths.prefix_table_answer_enriched),
        ("shared_step_table", paths.step_table),
        ("shared_feature_engineer_with_model", paths.feature_engineer_with_model),
    ]
    rows = []
    hash_limit = int(hash_limit_mb) * 1024 * 1024
    for name, path in candidates:
        exists = path.exists()
        size = int(path.stat().st_size) if exists and path.is_file() else None
        should_hash = bool(exists and path.is_file() and (hash_large or (size or 0) <= hash_limit))
        rows.append(
            {
                "name": name,
                "path": str(path),
                "exists": bool(exists),
                "size_bytes": size if size is not None else "",
                "sha256": _sha256(path) if should_hash else "",
                "sha256_status": "computed" if should_hash else ("skipped_large" if exists else "missing"),
            }
        )
    return rows


def _write_readme(
    out_dir: Path,
    *,
    external_refs: list[dict[str, Any]],
    symlinks: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> None:
    missing_artifacts = [row for row in artifacts if not row["exists"]]
    lines = [
        "# earlyeval Portability Audit",
        "",
        "Scope: mainline code should execute from `earlyeval/`; historical results are frozen; large data/model files stay external and are checked by manifest.",
        "",
        "## Summary",
        "",
        f"- external code references needing review: {len(external_refs)}",
        f"- symlinks under `paper/data`: {len(symlinks)}",
        f"- missing external artifacts: {len(missing_artifacts)}",
        "",
        "## Files",
        "",
        "- `external_code_references.csv`: source/config/script references to old executable code roots.",
        "- `paper_data_symlinks.csv`: symlinks that should be frozen to real files for a portable paper bundle.",
        "- `external_artifacts.csv`: heavyweight data/model files kept outside earlyeval, with size and optional sha256.",
        "- `manifest.json`: machine-readable summary.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit earlyeval portability boundaries.")
    parser.add_argument("--config", type=Path, default=Path("configs/paths.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/checks/portability_audit"))
    parser.add_argument(
        "--hash-large",
        action="store_true",
        help="Compute sha256 for all registered external artifacts, including multi-GB parquet files.",
    )
    parser.add_argument(
        "--hash-limit-mb",
        type=int,
        default=256,
        help="Hash files up to this size unless --hash-large is set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    external_refs = _scan_external_code_refs()
    symlinks = _scan_symlinks(package_root() / "paper" / "data")
    artifacts = _external_artifact_rows(
        args.config,
        hash_large=bool(args.hash_large),
        hash_limit_mb=int(args.hash_limit_mb),
    )

    import pandas as pd

    write_table(pd.DataFrame(external_refs), out_dir / "external_code_references.csv")
    write_table(pd.DataFrame(symlinks), out_dir / "paper_data_symlinks.csv")
    write_table(pd.DataFrame(artifacts), out_dir / "external_artifacts.csv")
    manifest = {
        "ok": not any(not row["exists"] for row in artifacts),
        "external_code_reference_count": len(external_refs),
        "paper_data_symlink_count": len(symlinks),
        "external_artifacts": artifacts,
    }
    write_json(out_dir / "manifest.json", manifest)
    _write_readme(out_dir, external_refs=external_refs, symlinks=symlinks, artifacts=artifacts)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
