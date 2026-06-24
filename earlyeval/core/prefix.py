from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

from earlyeval.core.legacy_paths import answer_module_root, require_path


def _load_legacy_module(module_name: str):
    root = require_path(answer_module_root(), "answer-aware source module root")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(module_name)


def build_prefix_table_from_parquet(parquet_input: str | Path, **kwargs: Any):
    """Delegate prefix-table construction to the current answer-aware source module.

    This function is intentionally explicit and does not run in default smoke
    paths. It is the earlyeval bridge until the full prefix builder is migrated.
    """
    module = _load_legacy_module("prefix_builder")
    return module.build_prefix_table(str(parquet_input), **kwargs)


def build_step_table_from_parquet(parquet_input: str | Path, **kwargs: Any):
    module = _load_legacy_module("step_builder")
    return module.build_step_table(str(parquet_input), **kwargs)
