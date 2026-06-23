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
    return load_paths().answer_module_root


def vendor_answer_module_root() -> Path:
    return load_paths().vendor_answer_module_root


def answer_module_root() -> Path:
    """Return the final3-owned answer-aware module root.

    The vendored copy is the editable source for future final3 work. The
    legacy package remains available through `legacy_answer_module_root()` for
    old-run reproduction and artifact lineage.
    """

    vendor = vendor_answer_module_root()
    if vendor.exists():
        return vendor
    return legacy_answer_module_root()


def shared_answer_data_root() -> Path:
    return load_paths().shared_answer_root


def shared_prefix_table_filtered_path() -> Path:
    return load_paths().prefix_table_filtered


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path
