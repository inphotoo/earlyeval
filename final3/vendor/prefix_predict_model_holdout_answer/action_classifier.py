'Public-release English note.'
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import config


def classify_action(action_text: Optional[str]) -> Tuple[str, List[str], str]:
    'Public-release English note.'
    if not action_text:
        return "none", [], "none"

    s = action_text.strip()
    if not s:
        return "none", [], "none"

    # ── submit ──
    if s == "submit" or s.endswith("\nsubmit") or s.startswith("submit"):
        return "submit", ["submit"], "submit"

    # ── str_replace_editor ──
    if s.startswith("str_replace_editor "):
        subtypes = _classify_editor(s)
        primary = subtypes[0] if subtypes else "str_replace_editor_other"
        major = config.MAJOR_TYPE_MAP.get(primary, "edit")
        return major, subtypes, primary

    # Public-release English note.
    lower = s.lower()
    subtypes = _classify_bash(s, lower)
    primary = subtypes[0] if subtypes else "run_cli"
    major = config.MAJOR_TYPE_MAP.get(primary, "execute")
    return major, subtypes, primary


def _classify_editor(s: str) -> list[str]:
    'Public-release English note.'
    padded = f" {s} "
    subtypes = []
    if " view " in padded or s.startswith("str_replace_editor view "):
        subtypes.append("read_view")
    if " create " in padded or s.startswith("str_replace_editor create "):
        subtypes.append("edit_create")
    if " str_replace " in padded or s.startswith("str_replace_editor str_replace "):
        subtypes.append("edit_replace")
    if " insert " in padded or s.startswith("str_replace_editor insert "):
        subtypes.append("edit_insert")
    if " undo_edit " in padded or s.startswith("str_replace_editor undo_edit "):
        subtypes.append("edit_undo")
    return subtypes if subtypes else ["read_view"]  # fallback


def _classify_bash(s: str, lower: str) -> list[str]:
    'Public-release English note.'
    subtypes = []

    # Public-release English note.
    if _is_test_command(lower):
        subtypes.append("test")
        return subtypes  # Public-release English note.

    # ── run_python ──
    if _is_python_command(lower):
        subtypes.append("run_python")
        return subtypes

    # ── git ──
    if lower.startswith("git "):
        subtypes.append("git")
        return subtypes

    # ── cleanup ──
    if lower.startswith("rm ") or lower.startswith("unlink "):
        subtypes.append("cleanup")
        return subtypes

    # ── read_search ──
    if _is_read_search(lower):
        subtypes.append("read_search")
        return subtypes

    # Public-release English note.
    subtypes.append("run_cli")
    return subtypes


# ═══════════════════════════════════════════════════
# Public-release English note.
# ═══════════════════════════════════════════════════

_TEST_PATTERNS = [
    re.compile(r"(?:^|\s)pytest(?:\s|$)"),
    re.compile(r"python\s+(?:-\w\s+)*-m\s+pytest"),
    re.compile(r"python\s+(?:-\w\s+)*-m\s+unittest"),
    re.compile(r"(?:^|\s)py\.test(?:\s|$)"),
    re.compile(r"(?:^|\s)tox(?:\s|$)"),
    re.compile(r"manage\.py\s+test"),
    re.compile(r"(?:^|\s)nosetests(?:\s|$)"),
    re.compile(r"(?:^|\s)python\s+.*test.*\.py(?:\s|$)"),  # python test_xxx.py
]


def _is_test_command(lower: str) -> bool:
    for pat in _TEST_PATTERNS:
        if pat.search(lower):
            return True
    return False


def _is_python_command(lower: str) -> bool:
    if lower.startswith("python ") or lower.startswith("python3 "):
        return True
    if " python " in lower or " python3 " in lower:
        return True
    if lower.startswith("ipython ") or lower.startswith("ipython3 "):
        return True
    if "python -m " in lower or "python3 -m " in lower:
        return True
    return False


_READ_SEARCH_PREFIXES = ("grep ", "find ", "ls ", "cat ", "sed -n", "head ", "tail ", "awk ", "wc ")


def _is_read_search(lower: str) -> bool:
    for prefix in _READ_SEARCH_PREFIXES:
        if lower.startswith(prefix):
            return True
    return False
