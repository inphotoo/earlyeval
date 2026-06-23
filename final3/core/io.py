from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在并返回 `Path`。

    所有入口统一用这个 helper 创建输出目录，避免每个 CLI 子命令自己写
    `mkdir(parents=True, exist_ok=True)`。这也让后续集中加入权限检查或
    dry-run 输出策略更容易。
    """

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_table(path: str | Path):
    """按文件后缀读取表格数据。

    支持格式：
    - `.parquet`: 大型 prediction/prefix 表的主格式。
    - `.csv` / `.tsv`: 小型报告和 smoke 数据。
    - `.jsonl`: normalized trajectories 或轻量记录列表。
    - `.json`: 单对象或对象数组。

    返回值是 pandas DataFrame。这里把依赖延迟 import，是为了让 `--help`
    这类轻量 CLI 不必立即加载 pandas。
    """

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
    """按目标后缀写出 DataFrame。

    这个函数刻意保持简单：不做 schema 推断，也不做压缩策略选择。schema
    和数据语义应由调用方负责，IO 层只保证格式稳定、目录存在。
    """

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
    """写出 UTF-8 JSON 文件。

    用于 run metadata、dry-run 计划和 manifest 小文件。大表格不要走这个
    函数，避免把 DataFrame 误写成巨型 JSON。
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    """读取 UTF-8 JSON 文件并返回 Python 对象。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))
