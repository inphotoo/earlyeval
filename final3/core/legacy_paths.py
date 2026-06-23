from __future__ import annotations

from pathlib import Path

from final3.core.paths import load_paths


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return package_root().parent


def final2_root() -> Path:
    return load_paths().legacy_final2_root


def legacy_answer_module_root() -> Path:
    """Historical artifact location only.

    Runtime code should use `answer_module_root()`, which resolves to the
    vendored final3-owned source tree. This helper remains for audit reports
    that need to mention where old artifacts came from.
    """

    return load_paths().answer_module_root


def vendor_answer_module_root() -> Path:
    return load_paths().vendor_answer_module_root


def answer_module_root() -> Path:
    """Return the final3-owned answer-aware module root.

    The GitHub release is code-self-contained: the active answer-aware trainer,
    feature code, and posthoc scripts live under `final3/vendor/`. Do not fall
    back to an external legacy package at runtime.
    """

    return vendor_answer_module_root()


def shared_answer_data_root() -> Path:
    return load_paths().shared_answer_root


def shared_prefix_table_filtered_path() -> Path:
    return load_paths().prefix_table_filtered


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path
