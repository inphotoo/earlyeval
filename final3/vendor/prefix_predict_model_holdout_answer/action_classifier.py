"""
动作分类器。

实现文档 §6 的 taxonomy，严格按照：
  先判 test → 再判 run_python → 再判其余
避免 "python -m pytest" 被误归为 run_python。
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import config


def classify_action(action_text: Optional[str]) -> Tuple[str, List[str], str]:
    """
    对一条 action 文本进行分类。

    Returns:
        (major_type, subtypes_list, primary_subtype)
    """
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

    # ── bash 命令 ──
    lower = s.lower()
    subtypes = _classify_bash(s, lower)
    primary = subtypes[0] if subtypes else "run_cli"
    major = config.MAJOR_TYPE_MAP.get(primary, "execute")
    return major, subtypes, primary


def _classify_editor(s: str) -> list[str]:
    """str_replace_editor 子类型判断。"""
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
    """
    Bash 命令子类型判断。
    关键：test 在 run_python 之前判定。
    """
    subtypes = []

    # ── test（优先级最高）──
    if _is_test_command(lower):
        subtypes.append("test")
        return subtypes  # test 直接返回，不混分

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

    # ── 兜底 ──
    subtypes.append("run_cli")
    return subtypes


# ═══════════════════════════════════════════════════
# 细粒度匹配规则
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
