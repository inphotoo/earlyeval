from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_table(path: str | Path):
    """Read a CSV/TSV/JSON/JSONL/parquet table into a DataFrame."""

    import pandas as pd

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame([data])
    raise ValueError(f"Unsupported table format: {path}")


def write_table(frame, path: str | Path) -> Path:
    """Write a DataFrame to a CSV/TSV/JSON/JSONL/parquet path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        frame.to_csv(path, index=False, sep=sep)
    elif suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for row in frame.to_dict("records"):
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif suffix == ".json":
        path.write_text(json.dumps(frame.to_dict("records"), ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported table format: {path}")
    return path


def write_json(path: str | Path, payload: Any) -> Path:
    """Write a JSON payload with stable UTF-8 formatting."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    """Read a JSON payload from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))
