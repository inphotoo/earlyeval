from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from earlyeval.core.io import ensure_dir, write_json, write_table
from earlyeval.core.paths import package_root


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(package_root().resolve()))
    except ValueError:
        return str(path)


def _symlink_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_symlink():
            continue
        target_raw = os.readlink(path)
        resolved = path.resolve(strict=False)
        rows.append(
            {
                "path": str(path),
                "path_rel": _rel(path),
                "target": target_raw,
                "resolved": str(resolved),
                "exists": resolved.exists(),
                "is_file": resolved.is_file(),
                "size_bytes": int(resolved.stat().st_size) if resolved.exists() and resolved.is_file() else "",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace small paper/data symlinks with real file snapshots."
    )
    parser.add_argument("--root", type=Path, default=Path("paper/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/checks/freeze_paper_data_symlinks"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--max-file-mb",
        type=int,
        default=64,
        help="Refuse to copy symlink targets larger than this size.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = package_root() / args.root if not args.root.is_absolute() else args.root
    out_dir = ensure_dir(args.output_dir)
    rows = _symlink_rows(root)
    max_bytes = int(args.max_file_mb) * 1024 * 1024
    planned: list[dict[str, Any]] = []
    for row in rows:
        status = "ready"
        size = row["size_bytes"] if isinstance(row["size_bytes"], int) else None
        if not row["exists"]:
            status = "missing_target"
        elif not row["is_file"]:
            status = "target_not_file"
        elif size is not None and size > max_bytes:
            status = "too_large"
        planned.append({**row, "status": status})

    if args.execute:
        for row in planned:
            if row["status"] != "ready":
                continue
            link_path = Path(row["path"])
            target_path = Path(row["resolved"])
            tmp_path = link_path.with_name(link_path.name + ".freeze_tmp")
            shutil.copy2(target_path, tmp_path)
            link_path.unlink()
            tmp_path.rename(link_path)
            row["status"] = "frozen"

    write_table(pd.DataFrame(planned), out_dir / "freeze_plan.csv")
    manifest = {
        "ok": not any(row["status"] in {"missing_target", "target_not_file", "too_large"} for row in planned),
        "execute": bool(args.execute),
        "root": str(root),
        "symlink_count": len(rows),
        "frozen_count": sum(1 for row in planned if row["status"] == "frozen"),
        "ready_count": sum(1 for row in planned if row["status"] == "ready"),
        "blocked_count": sum(1 for row in planned if row["status"] not in {"ready", "frozen"}),
        "max_file_mb": int(args.max_file_mb),
    }
    write_json(out_dir / "manifest.json", manifest)
    lines = [
        "# Freeze paper/data symlinks",
        "",
        f"- execute: {bool(args.execute)}",
        f"- symlinks scanned: {len(rows)}",
        f"- frozen: {manifest['frozen_count']}",
        f"- ready: {manifest['ready_count']}",
        f"- blocked: {manifest['blocked_count']}",
        "",
        "Details: `freeze_plan.csv`.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
