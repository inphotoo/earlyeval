from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

from final3.core.paths import package_root


def refresh_paper_tables(*, output_dir: str | Path | None = None) -> dict[str, Any]:
    """Refresh the current ICSE draft CSV tables.

    The canonical refresh logic lives next to the paper data so table lineage
    remains easy to inspect. This wrapper only exposes it through the final3
    CLI.
    """

    data_dir = package_root() / "paper" / "icse_submission_draft" / "data"
    script = data_dir / "refresh_tables.py"
    if not script.exists():
        raise FileNotFoundError(f"paper table refresh script not found: {script}")
    runpy.run_path(str(script), run_name="__main__")
    table_dir = Path(output_dir) if output_dir else data_dir
    tables = sorted(data_dir.glob("table_*.csv"))
    return {
        "ok": True,
        "script": str(script),
        "table_dir": str(table_dir),
        "tables": len(tables),
    }
