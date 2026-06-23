"""Gold-answer feature enrichment for SWE-bench prefix prediction."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from utils import get_logger

logger = get_logger("answer_features")

ANSWER_NUMERIC_FEATURES = [
    "gold_has_answer",
    "gold_problem_statement_chars",
    "gold_hints_chars",
    "gold_patch_chars",
    "gold_test_patch_chars",
    "gold_patch_hunks",
    "gold_test_patch_hunks",
    "gold_patch_files_count",
    "gold_test_files_count",
    "gold_patch_added_lines",
    "gold_patch_deleted_lines",
    "gold_test_added_lines",
    "gold_test_deleted_lines",
    "gold_fail_to_pass_count",
    "gold_pass_to_pass_count",
    "gold_patch_api_token_count",
    "gold_patch_import_token_count",
    "gold_patch_exception_keyword_count",
    "gold_patch_test_keyword_count",
    "gold_patch_config_keyword_count",
    "gold_patch_max_dir_depth",
    "gold_prefix_action_file_jaccard",
    "gold_prefix_feedback_file_jaccard",
    "gold_prefix_thought_file_jaccard",
    "gold_last_action_file_jaccard",
    "gold_last_feedback_file_jaccard",
    "gold_last_thought_file_jaccard",
    "gold_prefix_action_api_jaccard",
    "gold_prefix_feedback_api_jaccard",
    "gold_prefix_thought_api_jaccard",
    "gold_last_action_api_jaccard",
    "gold_last_feedback_api_jaccard",
    "gold_last_thought_api_jaccard",
    "gold_prefix_action_test_jaccard",
    "gold_prefix_feedback_test_jaccard",
    "gold_prefix_thought_test_jaccard",
    "gold_last_action_test_jaccard",
    "gold_last_feedback_test_jaccard",
    "gold_last_thought_test_jaccard",
    "gold_prefix_action_file_hits",
    "gold_prefix_feedback_file_hits",
    "gold_prefix_thought_file_hits",
    "gold_last_action_file_hits",
    "gold_last_feedback_file_hits",
    "gold_last_thought_file_hits",
    "gold_prefix_action_api_hits",
    "gold_prefix_feedback_api_hits",
    "gold_prefix_thought_api_hits",
    "gold_last_action_api_hits",
    "gold_last_feedback_api_hits",
    "gold_last_thought_api_hits",
    "gold_prefix_action_test_hits",
    "gold_prefix_feedback_test_hits",
    "gold_prefix_thought_test_hits",
    "gold_last_action_test_hits",
    "gold_last_feedback_test_hits",
    "gold_last_thought_test_hits",
]

ANSWER_BOOL_FEATURES = [
    "gold_has_hints",
    "gold_has_test_patch",
    "gold_prefix_action_file_hit_any",
    "gold_prefix_feedback_file_hit_any",
    "gold_prefix_thought_file_hit_any",
    "gold_last_action_file_hit_any",
    "gold_last_feedback_file_hit_any",
    "gold_last_thought_file_hit_any",
    "gold_prefix_action_api_hit_any",
    "gold_prefix_feedback_api_hit_any",
    "gold_prefix_thought_api_hit_any",
    "gold_last_action_api_hit_any",
    "gold_last_feedback_api_hit_any",
    "gold_last_thought_api_hit_any",
    "gold_prefix_action_test_hit_any",
    "gold_prefix_feedback_test_hit_any",
    "gold_prefix_thought_test_hit_any",
    "gold_last_action_test_hit_any",
    "gold_last_feedback_test_hit_any",
    "gold_last_thought_test_hit_any",
]

ANSWER_CATEGORICAL_FEATURES = [
    "gold_repo",
    "gold_difficulty",
    "gold_version",
    "gold_primary_patch_ext",
    "gold_primary_patch_dir",
]

ANSWER_TEXT_COLUMNS = {
    "tfidf_gold_patch": "gold_patch_text",
    "tfidf_gold_test_patch": "gold_test_patch_text",
    "tfidf_gold_fail_to_pass": "gold_fail_to_pass_text",
    "tfidf_gold_answer_summary": "gold_answer_summary_text",
}

_TEXT_MATCH_COLUMNS = {
    "prefix_action": "prefix_action_text",
    "prefix_feedback": "prefix_feedback_text",
    "prefix_thought": "prefix_thought_text",
    "last_action": "last_action_text",
    "last_feedback": "last_feedback_text",
    "last_thought": "last_thought_text",
}

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_DIFF_FILE_RE = re.compile(r"^diff --git a/(.*?) b/(.*?)$", re.MULTILINE)
_HUNK_RE = re.compile(r"^@@", re.MULTILINE)
_DEF_RE = re.compile(r"^[+\- ]\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_IMPORT_RE = re.compile(r"^[+\- ]\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*))", re.MULTILINE)
_LINE_NUM_RE = re.compile(r"(?:^|[^\w])(?:line|L)\s*#?\s*(\d{1,6})(?:[^\w]|$)", re.IGNORECASE)

_EXCEPTION_WORDS = ("error", "exception", "traceback", "assert", "raise", "warning")
_TEST_WORDS = ("test", "pytest", "unittest", "assert", "fail_to_pass", "pass_to_pass")
_CONFIG_WORDS = ("config", "setting", "option", "env", "version", "deprecat")


def _parse_json_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    return [text]


def _diff_paths(diff_text: str) -> list[str]:
    paths = []
    for _a, b_path in _DIFF_FILE_RE.findall(diff_text or ""):
        if b_path and b_path != "/dev/null":
            paths.append(b_path)
    return sorted(set(paths))


def _changed_line_counts(diff_text: str) -> tuple[int, int]:
    added = deleted = 0
    for line in (diff_text or "").splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            deleted += 1
    return added, deleted


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _IDENTIFIER_RE.finditer(text or "")}


def _basename_tokens(paths: Iterable[str]) -> set[str]:
    out = set()
    for path in paths:
        p = Path(path)
        out.add(p.name.lower())
        out.add(p.stem.lower())
    return {x for x in out if x}


def _path_dir(path: str) -> str:
    parent = str(Path(path).parent)
    return "" if parent == "." else parent


def _safe_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return float(len(left & right) / len(left | right))


def _keyword_count(text: str, words: Iterable[str]) -> int:
    low = (text or "").lower()
    return int(sum(low.count(w) for w in words))


def _make_answer_rows(verified_jsonl: str | Path) -> pd.DataFrame:
    rows = []
    path = Path(verified_jsonl)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            instance_id = str(obj.get("instance_id", ""))
            patch = str(obj.get("patch", "") or "")
            test_patch = str(obj.get("test_patch", "") or "")
            problem = str(obj.get("problem_statement", "") or "")
            hints = str(obj.get("hints_text", "") or "")
            fail_to_pass = _parse_json_list(obj.get("FAIL_TO_PASS"))
            pass_to_pass = _parse_json_list(obj.get("PASS_TO_PASS"))
            patch_paths = _diff_paths(patch)
            test_paths = _diff_paths(test_patch)
            patch_added, patch_deleted = _changed_line_counts(patch)
            test_added, test_deleted = _changed_line_counts(test_patch)
            api_tokens = set(x.lower() for x in _DEF_RE.findall(patch))
            api_tokens |= _tokenize("\n".join(patch_paths))
            import_tokens = set()
            for left, right in _IMPORT_RE.findall(patch):
                import_tokens |= _tokenize(left or right)
            test_tokens = _tokenize("\n".join(fail_to_pass + pass_to_pass + test_paths))
            primary_path = patch_paths[0] if patch_paths else ""
            primary_ext = Path(primary_path).suffix.lower() if primary_path else "__MISSING__"
            primary_dir = _path_dir(primary_path) if primary_path else "__MISSING__"
            max_depth = max((len(Path(p).parts) for p in patch_paths), default=0)
            summary = "\n".join([
                problem[:4000], hints[:2000],
                "PATCH_FILES: " + " ".join(patch_paths),
                "TEST_FILES: " + " ".join(test_paths),
                "FAIL_TO_PASS: " + " ".join(fail_to_pass),
                "API_TOKENS: " + " ".join(sorted(api_tokens)),
            ])
            rows.append({
                "instance_id": instance_id,
                "gold_has_answer": 1,
                "gold_repo": str(obj.get("repo", "__MISSING__") or "__MISSING__"),
                "gold_difficulty": str(obj.get("difficulty", "__MISSING__") or "__MISSING__"),
                "gold_version": str(obj.get("version", "__MISSING__") or "__MISSING__"),
                "gold_primary_patch_ext": primary_ext or "__MISSING__",
                "gold_primary_patch_dir": primary_dir or "__MISSING__",
                "gold_problem_statement_chars": len(problem),
                "gold_hints_chars": len(hints),
                "gold_has_hints": bool(hints),
                "gold_has_test_patch": bool(test_patch),
                "gold_patch_chars": len(patch),
                "gold_test_patch_chars": len(test_patch),
                "gold_patch_hunks": len(_HUNK_RE.findall(patch)),
                "gold_test_patch_hunks": len(_HUNK_RE.findall(test_patch)),
                "gold_patch_files_count": len(patch_paths),
                "gold_test_files_count": len(test_paths),
                "gold_patch_added_lines": patch_added,
                "gold_patch_deleted_lines": patch_deleted,
                "gold_test_added_lines": test_added,
                "gold_test_deleted_lines": test_deleted,
                "gold_fail_to_pass_count": len(fail_to_pass),
                "gold_pass_to_pass_count": len(pass_to_pass),
                "gold_patch_api_token_count": len(api_tokens),
                "gold_patch_import_token_count": len(import_tokens),
                "gold_patch_exception_keyword_count": _keyword_count(patch + test_patch + problem, _EXCEPTION_WORDS),
                "gold_patch_test_keyword_count": _keyword_count(patch + test_patch + problem, _TEST_WORDS),
                "gold_patch_config_keyword_count": _keyword_count(patch + test_patch + problem, _CONFIG_WORDS),
                "gold_patch_max_dir_depth": max_depth,
                "gold_patch_text": patch,
                "gold_test_patch_text": test_patch,
                "gold_fail_to_pass_text": "\n".join(fail_to_pass),
                "gold_answer_summary_text": summary,
                "_gold_file_tokens": sorted(_basename_tokens(patch_paths + test_paths)),
                "_gold_api_tokens": sorted(api_tokens | import_tokens),
                "_gold_test_tokens": sorted(test_tokens),
            })
    ans = pd.DataFrame(rows)
    if ans.empty:
        raise ValueError(f"No rows loaded from {path}")
    logger.info(f"Loaded gold answers: {len(ans)} instances from {path}")
    return ans


def _fill_missing_answer_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ANSWER_NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in ANSWER_BOOL_FEATURES:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)
    for col in ANSWER_CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "__MISSING__"
        df[col] = df[col].fillna("__MISSING__").astype(str).replace("", "__MISSING__")
    for col in ANSWER_TEXT_COLUMNS.values():
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    for col in ["_gold_file_tokens", "_gold_api_tokens", "_gold_test_tokens"]:
        if col not in df.columns:
            df[col] = [[] for _ in range(len(df))]
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
    return df


def enrich_prefix_with_answer_features(prefix_df: pd.DataFrame, verified_jsonl: str | Path) -> tuple[pd.DataFrame, dict]:
    """Join SWE-bench verified gold-answer metadata and add match features."""
    answer_df = _make_answer_rows(verified_jsonl)
    before_instances = set(prefix_df["instance_id"].astype(str).unique())
    out = prefix_df.merge(answer_df, on="instance_id", how="left")
    out = _fill_missing_answer_columns(out)

    unmatched_instances = sorted(before_instances - set(answer_df["instance_id"].astype(str).unique()))
    for match_name, text_col in _TEXT_MATCH_COLUMNS.items():
        text_tokens = out[text_col].fillna("").astype(str).map(_tokenize)
        for target_name, token_col in [
            ("file", "_gold_file_tokens"),
            ("api", "_gold_api_tokens"),
            ("test", "_gold_test_tokens"),
        ]:
            gold_tokens = out[token_col].map(lambda xs: set(str(x).lower() for x in (xs or [])))
            hits = [len(a & b) for a, b in zip(text_tokens, gold_tokens)]
            jacc = [_safe_jaccard(a, b) for a, b in zip(text_tokens, gold_tokens)]
            hit_col = f"gold_{match_name}_{target_name}_hits"
            jac_col = f"gold_{match_name}_{target_name}_jaccard"
            any_col = f"gold_{match_name}_{target_name}_hit_any"
            out[hit_col] = np.asarray(hits, dtype=np.float32)
            out[jac_col] = np.asarray(jacc, dtype=np.float32)
            out[any_col] = np.asarray(hits, dtype=np.int32) > 0

    out = _fill_missing_answer_columns(out)
    summary = {
        "verified_instances": int(answer_df["instance_id"].nunique()),
        "prefix_instances": int(len(before_instances)),
        "matched_instances": int(len(before_instances) - len(unmatched_instances)),
        "unmatched_instances": int(len(unmatched_instances)),
        "unmatched_instance_examples": unmatched_instances[:20],
    }
    logger.info(f"Gold answer enrichment summary: {summary}")
    return out, summary
