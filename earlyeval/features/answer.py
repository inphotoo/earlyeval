from __future__ import annotations

import importlib
import sys
from pathlib import Path

from earlyeval.core.legacy_paths import answer_module_root, require_path


def enrich_with_gold_answers(prefix_frame, verified_jsonl: str | Path):
    """Add SWE-bench gold-answer features using the migrated source module."""
    root = require_path(answer_module_root(), "answer-aware source module root")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("answer_features")
    return module.enrich_prefix_with_answer_features(prefix_frame, str(verified_jsonl))
