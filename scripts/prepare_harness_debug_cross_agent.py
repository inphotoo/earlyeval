#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import base64
import concurrent.futures
import csv
import ctypes
import gc
import html
import json
import math
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DATA_ROOT_DEFAULT = WORKSPACE_ROOT / "data" / "harness_debug_cross_agent_20260625"
PREP_ROOT_DEFAULT = (
    PROJECT_ROOT
    / "paper"
    / "icse_submission_draft"
    / "rq_tables_reorg_20260623"
    / "cross_agent_model_prep_20260624"
)
SWE_VERIFIED_JSONL_DEFAULT = WORKSPACE_ROOT / "data" / "swe_verify_500" / "offical_answer" / "test.jsonl"
TB_PREFIX_DEFAULT = (
    WORKSPACE_ROOT
    / "data"
    / "other_bench_prefix_tables"
    / "terminalbench"
    / "holdout15_p99_v3_audited_20260512"
    / "prefix_table_terminalbench.parquet"
)
S3_BASE = "https://swe-bench-submissions.s3.amazonaws.com"
GITHUB_CONTENTS_BASE = "https://api.github.com/repos/SWE-bench/experiments/contents/evaluation/verified"


def _vendor_imports() -> dict[str, Any]:
    vendor = PROJECT_ROOT / "earlyeval" / "vendor" / "prefix_predict_model_holdout_answer"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    from action_classifier import classify_action  # type: ignore
    from observation_parser import parse_observation  # type: ignore
    from prefix_builder import _build_prefix_features  # type: ignore

    return {
        "classify_action": classify_action,
        "parse_observation": parse_observation,
        "_build_prefix_features": _build_prefix_features,
    }


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _safe_unit_part(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "unit"


def _swe_unit_id(model_group: str, heldout_unit: str) -> str:
    return f"swe__{_safe_unit_part(model_group)}__scaffold__{_safe_unit_part(heldout_unit)}"


def _tb_unit_id(model_group: str, heldout_unit: str) -> str:
    return f"tb__{_safe_unit_part(model_group)}__agent__{_safe_unit_part(heldout_unit)}"


def _request_get(url: str, *, params: dict[str, str] | None = None, timeout: int = 60, retries: int = 4) -> requests.Response:
    last_exc: Exception | None = None
    request_timeout = (10, int(timeout))
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=request_timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == retries:
                break
            time.sleep(min(2 * attempt, 10))
    assert last_exc is not None
    raise last_exc


def list_s3_keys(prefix: str) -> list[dict[str, Any]]:
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    rows: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        resp = _request_get(S3_BASE + "/", params=params, timeout=60)
        root = ET.fromstring(resp.text)
        for item in root.findall("s:Contents", ns):
            key = item.findtext("s:Key", namespaces=ns) or ""
            size = int(item.findtext("s:Size", namespaces=ns) or "0")
            rows.append({"key": key, "size": size})
        next_token = root.findtext("s:NextContinuationToken", namespaces=ns)
        if not next_token:
            break
        token = next_token
    return rows


def is_relevant_swe_key(key: str) -> bool:
    name = Path(key).name
    lower = name.lower()
    if not lower:
        return False
    if lower in {"preds.json", "predictions.json"}:
        return False
    skip_suffixes = (".patch", ".pred", ".diff", ".config.yaml", ".yaml", ".yml")
    if lower.endswith(skip_suffixes):
        return False
    if "regression_test_result" in lower or lower == "selected_patch.json":
        return False
    return lower.endswith((".traj", ".json", ".txt", ".log"))


def instance_from_s3_key(folder: str, key: str) -> str:
    prefix = f"verified/{folder}/trajs/"
    rel = key[len(prefix) :] if key.startswith(prefix) else key
    parts = Path(rel).parts
    if parts and "__" in parts[0]:
        inst = parts[0]
    else:
        inst = Path(rel).stem
    inst = re.sub(r"\.(traj|json|txt|log)$", "", inst)
    inst = re.sub(r"(_trail|_trial)_\d+$", "", inst)
    inst = re.sub(r"_voting$", "", inst)
    return inst


def download_swe(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    prep_root = Path(args.prep_root)
    raw_root = _mkdir(data_root / "raw_sweverify_s3")
    manifest_dir = _mkdir(data_root / "manifests")
    swe_manifest = _read_csv(prep_root / "sweverify_harness_debug_unit_set_core18.csv")
    all_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for _, row in swe_manifest.iterrows():
        folder = str(row["submission_folder"])
        prefix = f"verified/{folder}/trajs/"
        print(f"[swe-download] listing {folder}", flush=True)
        for item in list_s3_keys(prefix):
            key = str(item["key"])
            selected = is_relevant_swe_key(key)
            rel = key[len(prefix) :] if key.startswith(prefix) else key
            local_path = raw_root / folder / rel
            rec = {
                "submission_folder": folder,
                "model_group": row["model_group"],
                "heldout_unit": row["heldout_unit"],
                "unit_id": _swe_unit_id(str(row["model_group"]), str(row["heldout_unit"])),
                "key": key,
                "size": int(item["size"]),
                "selected": bool(selected),
                "instance_id": instance_from_s3_key(folder, key),
                "local_path": str(local_path),
            }
            all_rows.append(rec)
            if selected:
                selected_rows.append(rec)
    pd.DataFrame(all_rows).to_csv(manifest_dir / "swe_s3_all_keys.csv", index=False)
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(manifest_dir / "swe_s3_selected_keys.csv", index=False)
    size_gb = selected_df["size"].sum() / (1024**3) if not selected_df.empty else 0.0
    print(f"[swe-download] selected files={len(selected_df)} estimated_size_gib={size_gb:.2f}", flush=True)
    if args.list_only:
        return
    def fetch_one(payload: tuple[int, dict[str, Any]]) -> tuple[int, str, str]:
        i, rec = payload
        local_path = Path(str(rec["local_path"]))
        if local_path.exists() and local_path.stat().st_size == int(rec["size"]):
            return i, "cached", str(local_path)
        _mkdir(local_path.parent)
        url = f"{S3_BASE}/{rec['key']}"
        resp = _request_get(url, timeout=120, retries=5)
        tmp = local_path.with_suffix(local_path.suffix + ".tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(local_path)
        return i, "downloaded", str(local_path)

    workers = max(1, int(args.download_workers))
    if workers == 1:
        iterator = map(fetch_one, enumerate(selected_rows, start=1))
    else:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        iterator = pool.map(fetch_one, enumerate(selected_rows, start=1))
    try:
        for i, status, path in iterator:
            if i == 1 or i % 200 == 0:
                print(f"[swe-download] {i}/{len(selected_rows)} status={status} path={path}", flush=True)
    finally:
        if workers != 1:
            pool.shutdown(wait=True, cancel_futures=False)
    print("[swe-download] done", flush=True)


def download_swe_results(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    prep_root = Path(args.prep_root)
    results_root = _mkdir(data_root / "sweverify_results")
    manifest_dir = _mkdir(data_root / "manifests")
    swe_manifest = _read_csv(prep_root / "sweverify_harness_debug_unit_set_core18.csv")
    rows: list[dict[str, Any]] = []
    for _, row in swe_manifest.iterrows():
        folder = str(row["submission_folder"])
        out = results_root / folder / "results.json"
        _mkdir(out.parent)
        if not out.exists() or out.stat().st_size == 0:
            print(f"[swe-results] download {folder}", flush=True)
            api_url = f"{GITHUB_CONTENTS_BASE}/{folder}/results/results.json"
            resp = _request_get(api_url, params={"ref": "main"}, timeout=30, retries=5)
            payload = resp.json()
            content = base64.b64decode(str(payload["content"]).encode("utf-8"))
            out.write_bytes(content)
        data = json.loads(out.read_text(encoding="utf-8"))
        resolved = set(data.get("resolved") or [])
        unresolved = set(data.get("unresolved") or [])
        no_generation = set(data.get("no_generation") or [])
        no_logs = set(data.get("no_logs") or [])
        all_ids = sorted(resolved | unresolved | no_generation | no_logs)
        for instance_id in all_ids:
            rows.append(
                {
                    "submission_folder": folder,
                    "instance_id": instance_id,
                    "resolved": instance_id in resolved,
                    "status": (
                        "resolved"
                        if instance_id in resolved
                        else "no_generation"
                        if instance_id in no_generation
                        else "no_logs"
                        if instance_id in no_logs
                        else "unresolved"
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(manifest_dir / "swe_results_by_instance.csv", index=False)
    print(f"[swe-results] rows={len(rows)}", flush=True)


def _load_verified_prompts(path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("problem_statement") or "")
            hints = str(row.get("hints_text") or "")
            if hints.strip():
                text = f"{text}\n\nHints:\n{hints}"
            prompts[str(row["instance_id"])] = text
    return prompts


def _clean_text(value: Any, *, limit: int) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item.get("text") or ""))
                elif "content" in item:
                    parts.append(_clean_text(item.get("content"), limit=limit))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        text = "\n".join(parts)
    elif isinstance(value, dict):
        if "text" in value:
            text = str(value.get("text") or "")
        elif "content" in value:
            text = _clean_text(value.get("content"), limit=limit)
        else:
            text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    text = html.unescape(text).replace("\x00", "")
    if len(text) > limit:
        return text[:limit] + "\n[TRUNCATED]"
    return text


def _json_or_literal(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:  # noqa: BLE001
            pass
    return text


def _tool_name_to_action(name: str, args: Any, *, text_limit: int) -> str:
    parsed = _json_or_literal(args)
    lower_name = name.lower().strip()
    if isinstance(parsed, dict):
        command = str(parsed.get("command") or parsed.get("cmd") or parsed.get("input") or "").strip()
        path = str(parsed.get("path") or parsed.get("file_path") or parsed.get("directory") or "").strip()
        if "bash" in lower_name or "shell" in lower_name or lower_name in {"execute", "execute_command", "sendShellInput".lower()}:
            return command or str(parsed)[:text_limit]
        if "grep" in lower_name:
            pattern = str(parsed.get("pattern") or parsed.get("searchTerm") or parsed.get("query") or "").strip()
            include = str(parsed.get("include") or "").strip()
            return f"grep -R {pattern} {include}".strip()
        if "list" in lower_name and path:
            return f"str_replace_editor view {path}"
        if "read" in lower_name and path:
            return f"str_replace_editor view {path}"
        if "editor" in lower_name or "replace" in lower_name or "edit" in lower_name or "string" in lower_name:
            cmd = command or str(parsed.get("action") or "view")
            bits = ["str_replace_editor", cmd]
            if path:
                bits.append(path)
            return " ".join(bits)
        if command:
            return command
        return f"{name} {json.dumps(parsed, ensure_ascii=False)[:text_limit]}".strip()
    text = _clean_text(parsed, limit=text_limit)
    if lower_name in {"bash", "execute_bash", "execute_command", "shell"}:
        return text
    return f"{name} {text}".strip()


def _overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _nearly_equal(a: str, b: str) -> bool:
    if not a.strip() and not b.strip():
        return True
    if not a.strip() or not b.strip():
        return False
    return _overlap(a, b) >= 0.9


def _append_step(
    steps: list[dict[str, Any]],
    *,
    vendor: dict[str, Any],
    traj_id: str,
    instance_id: str,
    resolved: bool,
    model: str,
    thought_text: str,
    action_text: str,
    feedback_text: str,
    assistant_content_text: str,
    tool_name: str,
    text_limit: int,
) -> None:
    action_text = _clean_text(action_text, limit=text_limit)
    feedback_text = _clean_text(feedback_text, limit=text_limit)
    thought_text = _clean_text(thought_text, limit=text_limit)
    assistant_content_text = _clean_text(assistant_content_text, limit=text_limit)
    if not action_text.strip() and not feedback_text.strip() and not thought_text.strip():
        return
    major_type, subtypes, primary_subtype = vendor["classify_action"](action_text)
    sig = vendor["parse_observation"](feedback_text)
    step_idx = len(steps) + 1
    steps.append(
        {
            "traj_id": traj_id,
            "instance_id": instance_id,
            "group_id": traj_id,
            "resolved": int(bool(resolved)),
            "model": model,
            "step_idx": step_idx,
            "message_start_idx": step_idx,
            "message_end_idx": step_idx + 1,
            "thought_text": thought_text,
            "action_text": action_text,
            "assistant_content_text": assistant_content_text,
            "tool_names": [tool_name] if tool_name else [],
            "tool_count": 1 if tool_name else 0,
            "tool_args_text": action_text,
            "tool_output_text": feedback_text,
            "observation_text": feedback_text,
            "combined_feedback_text": feedback_text,
            "action_major_type": major_type,
            "action_subtypes": subtypes,
            "action_primary_subtype": primary_subtype,
            "has_tool_output": bool(feedback_text.strip()),
            "has_observation": bool(feedback_text.strip()),
            "tool_error_seen_this_step": sig.tool_error,
            "traceback_seen_this_step": sig.traceback_seen,
            "test_fail_seen_this_step": sig.test_fail_seen,
            "test_pass_seen_this_step": sig.test_pass_seen,
            "last_fail_count_this_step": sig.fail_count,
            "action_char_len": len(action_text),
            "feedback_char_len": len(feedback_text),
            "thought_char_len": len(thought_text),
            "assistant_content_char_len": len(assistant_content_text),
            "has_thought_text": bool(thought_text.strip()),
            "has_assistant_content_text": bool(assistant_content_text.strip()),
            "thought_equals_content": _nearly_equal(thought_text, assistant_content_text),
            "thought_action_overlap_ratio": _overlap(thought_text, action_text),
            "content_action_overlap_ratio": _overlap(assistant_content_text, action_text),
        }
    )


def _extract_thinking_from_message(msg: dict[str, Any], *, text_limit: int) -> str:
    parts = [_clean_text(msg.get("reasoning_content"), limit=text_limit)]
    for block in msg.get("thinking_blocks") or []:
        if isinstance(block, dict):
            parts.append(_clean_text(block.get("thinking") or block.get("text"), limit=text_limit))
    content = _clean_text(msg.get("content"), limit=text_limit)
    thinking_tags = re.findall(r"<thinking>(.*?)</thinking>", content, flags=re.S | re.I)
    parts.extend(thinking_tags)
    if not any(part.strip() for part in parts):
        parts.append(content)
    return _clean_text("\n".join(part for part in parts if part), limit=text_limit)


def _conversation_steps(
    messages: list[Any],
    *,
    vendor: dict[str, Any],
    traj_id: str,
    instance_id: str,
    resolved: bool,
    model: str,
    text_limit: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    idx = 0
    while idx < len(messages) and len(steps) < max_steps:
        msg = messages[idx]
        if not isinstance(msg, dict):
            idx += 1
            continue
        if msg.get("role") != "assistant":
            idx += 1
            continue
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            text = _clean_text(msg.get("content"), limit=text_limit)
            for action in _extract_actions_from_text(text, max_actions=4):
                _append_step(
                    steps,
                    vendor=vendor,
                    traj_id=traj_id,
                    instance_id=instance_id,
                    resolved=resolved,
                    model=model,
                    thought_text=text,
                    action_text=action,
                    feedback_text="",
                    assistant_content_text=text,
                    tool_name="text",
                    text_limit=text_limit,
                )
                if len(steps) >= max_steps:
                    break
            idx += 1
            continue
        following: list[dict[str, Any]] = []
        j = idx + 1
        while j < len(messages):
            nxt = messages[j]
            if isinstance(nxt, dict) and nxt.get("role") == "tool":
                following.append(nxt)
                j += 1
                continue
            break
        outputs_by_id = {str(m.get("tool_call_id")): _clean_text(m.get("content"), limit=text_limit) for m in following}
        outputs_by_name: dict[str, list[str]] = defaultdict(list)
        for m in following:
            outputs_by_name[str(m.get("name") or "")].append(_clean_text(m.get("content"), limit=text_limit))
        thought = _extract_thinking_from_message(msg, text_limit=text_limit)
        content = _clean_text(msg.get("content"), limit=text_limit)
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            func = call.get("function") or {}
            name = str(func.get("name") or call.get("name") or "tool")
            if name.lower() in {"think", "thinking"}:
                thought = (thought + "\n" + _clean_text(func.get("arguments"), limit=text_limit)).strip()
                continue
            action = _tool_name_to_action(name, func.get("arguments"), text_limit=text_limit)
            call_id = str(call.get("id") or call.get("tool_call_id") or "")
            feedback = outputs_by_id.get(call_id) or "\n".join(outputs_by_name.get(name, []))
            _append_step(
                steps,
                vendor=vendor,
                traj_id=traj_id,
                instance_id=instance_id,
                resolved=resolved,
                model=model,
                thought_text=thought,
                action_text=action,
                feedback_text=feedback,
                assistant_content_text=content,
                tool_name=name,
                text_limit=text_limit,
            )
            if len(steps) >= max_steps:
                break
        idx = max(j, idx + 1)
    return steps[:max_steps]


def _event_map_steps(
    events: Iterable[Any],
    *,
    vendor: dict[str, Any],
    traj_id: str,
    instance_id: str,
    resolved: bool,
    model: str,
    text_limit: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    pending_thought: list[str] = []
    for ev in events:
        if len(steps) >= max_steps:
            break
        if not isinstance(ev, dict):
            continue
        author = str(ev.get("author_name") or ev.get("name") or ev.get("role") or "event")
        message = _clean_text(ev.get("message") or ev.get("content") or ev.get("output"), limit=text_limit)
        input_text = ev.get("input_text") if "input_text" in ev else ev.get("input")
        if "thought" in author.lower() and not input_text:
            pending_thought.append(message)
            continue
        action = _tool_name_to_action(author, input_text, text_limit=text_limit)
        if not action.strip():
            action = author
        thought = "\n".join(pending_thought)
        pending_thought = []
        _append_step(
            steps,
            vendor=vendor,
            traj_id=traj_id,
            instance_id=instance_id,
            resolved=resolved,
            model=model,
            thought_text=thought,
            action_text=action,
            feedback_text=message,
            assistant_content_text=(thought + "\n" + action).strip(),
            tool_name=author,
            text_limit=text_limit,
        )
    return steps[:max_steps]


def _moatless_steps(
    data: dict[str, Any],
    *,
    vendor: dict[str, Any],
    traj_id: str,
    instance_id: str,
    resolved: bool,
    model: str,
    text_limit: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for node in data.get("nodes") or []:
        if len(steps) >= max_steps or not isinstance(node, dict):
            break
        thoughts = node.get("thoughts") or {}
        thought = _clean_text(thoughts.get("text") if isinstance(thoughts, dict) else thoughts, limit=text_limit)
        assistant = _clean_text(node.get("assistant_message"), limit=text_limit)
        for action_step in node.get("action_steps") or []:
            if len(steps) >= max_steps or not isinstance(action_step, dict):
                break
            action_obj = action_step.get("action") or {}
            obs_obj = action_step.get("observation") or {}
            class_name = str(action_obj.get("action_args_class") or action_obj.get("name") or "moatless")
            feedback = _clean_text(obs_obj.get("message") if isinstance(obs_obj, dict) else obs_obj, limit=text_limit)
            action = _tool_name_to_action(class_name, action_obj, text_limit=text_limit)
            _append_step(
                steps,
                vendor=vendor,
                traj_id=traj_id,
                instance_id=instance_id,
                resolved=resolved,
                model=model,
                thought_text=thought,
                action_text=action,
                feedback_text=feedback,
                assistant_content_text=assistant or thought,
                tool_name=class_name.split(".")[-1],
                text_limit=text_limit,
            )
    return steps[:max_steps]


def _sweagent_steps(
    data: dict[str, Any],
    *,
    vendor: dict[str, Any],
    traj_id: str,
    instance_id: str,
    resolved: bool,
    model: str,
    text_limit: int,
    max_steps: int,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for item in data.get("trajectory") or []:
        if len(steps) >= max_steps or not isinstance(item, dict):
            break
        _append_step(
            steps,
            vendor=vendor,
            traj_id=traj_id,
            instance_id=instance_id,
            resolved=resolved,
            model=model,
            thought_text=_clean_text(item.get("response"), limit=text_limit),
            action_text=_clean_text(item.get("action"), limit=text_limit),
            feedback_text=_clean_text(item.get("observation"), limit=text_limit),
            assistant_content_text=_clean_text(item.get("response"), limit=text_limit),
            tool_name="sweagent",
            text_limit=text_limit,
        )
    return steps[:max_steps]


def _extract_actions_from_text(text: str, *, max_actions: int) -> list[str]:
    patterns = [
        r"```(?:bash|sh|shell)?\s*(.*?)```",
        r"<execute_bash>(.*?)</execute_bash>",
        r"<command>(.*?)</command>",
        r"<antml:parameter name=\"command\">(.*?)</antml:parameter>",
        r'"command"\s*:\s*"([^"]+)"',
        r"'command'\s*:\s*'([^']+)'",
        r"Action:\s*(.*?)(?:\n|$)",
    ]
    actions: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.S | re.I):
            action = html.unescape(match.group(1)).strip()
            action = re.sub(r"\s+", " ", action)
            if action and action not in actions:
                actions.append(action)
            if len(actions) >= max_actions:
                return actions
    return actions


def _text_log_steps(
    text: str,
    *,
    vendor: dict[str, Any],
    traj_id: str,
    instance_id: str,
    resolved: bool,
    model: str,
    text_limit: int,
    max_steps: int,
    chunk_chars: int,
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    actions = _extract_actions_from_text(text, max_actions=max_steps)
    if actions:
        cursor = 0
        for action in actions:
            if len(steps) >= max_steps:
                break
            pos = text.find(action, cursor)
            if pos < 0:
                pos = cursor
            next_cursor = pos + len(action)
            feedback = text[next_cursor : next_cursor + text_limit]
            thought = text[max(0, pos - min(text_limit, 4000)) : pos]
            _append_step(
                steps,
                vendor=vendor,
                traj_id=traj_id,
                instance_id=instance_id,
                resolved=resolved,
                model=model,
                thought_text=thought,
                action_text=action,
                feedback_text=feedback,
                assistant_content_text=thought,
                tool_name="log",
                text_limit=text_limit,
            )
            cursor = next_cursor
        return steps[:max_steps]
    n_chunks = min(max_steps, max(1, math.ceil(len(text) / max(1, chunk_chars))))
    for idx in range(n_chunks):
        start = idx * chunk_chars
        chunk = text[start : start + chunk_chars]
        _append_step(
            steps,
            vendor=vendor,
            traj_id=traj_id,
            instance_id=instance_id,
            resolved=resolved,
            model=model,
            thought_text="",
            action_text=f"log_chunk_{idx + 1}",
            feedback_text=chunk,
            assistant_content_text="",
            tool_name="log",
            text_limit=text_limit,
        )
    return steps[:max_steps]


def parse_file_to_steps(
    path: Path,
    *,
    vendor: dict[str, Any],
    traj_id: str,
    instance_id: str,
    resolved: bool,
    model: str,
    text_limit: int,
    max_steps: int,
    chunk_chars: int,
) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(text)
    except Exception:  # noqa: BLE001
        return _text_log_steps(
            text,
            vendor=vendor,
            traj_id=traj_id,
            instance_id=instance_id,
            resolved=resolved,
            model=model,
            text_limit=text_limit,
            max_steps=max_steps,
            chunk_chars=chunk_chars,
        )
    if isinstance(data, dict) and "trajectory" in data and isinstance(data.get("trajectory"), list):
        return _sweagent_steps(
            data,
            vendor=vendor,
            traj_id=traj_id,
            instance_id=instance_id,
            resolved=resolved,
            model=model,
            text_limit=text_limit,
            max_steps=max_steps,
        )
    if isinstance(data, dict) and "nodes" in data:
        return _moatless_steps(
            data,
            vendor=vendor,
            traj_id=traj_id,
            instance_id=instance_id,
            resolved=resolved,
            model=model,
            text_limit=text_limit,
            max_steps=max_steps,
        )
    if isinstance(data, dict) and "best_attempt" in data:
        attempt = data.get(str(data.get("best_attempt"))) or {}
        traj = attempt.get("traj") if isinstance(attempt, dict) else None
        if isinstance(traj, dict):
            return _event_map_steps(
                traj.values(),
                vendor=vendor,
                traj_id=traj_id,
                instance_id=instance_id,
                resolved=resolved,
                model=model,
                text_limit=text_limit,
                max_steps=max_steps,
            )
    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict) and any(
        isinstance(v, dict) and "author_name" in v for v in data[0].values()
    ):
        return _event_map_steps(
            data[0].values(),
            vendor=vendor,
            traj_id=traj_id,
            instance_id=instance_id,
            resolved=resolved,
            model=model,
            text_limit=text_limit,
            max_steps=max_steps,
        )
    if isinstance(data, list) and data and all(isinstance(item, dict) and "role" in item for item in data):
        return _conversation_steps(
            data,
            vendor=vendor,
            traj_id=traj_id,
            instance_id=instance_id,
            resolved=resolved,
            model=model,
            text_limit=text_limit,
            max_steps=max_steps,
        )
    if isinstance(data, list) and data and all(isinstance(item, str) for item in data):
        return _text_log_steps(
            "\n".join(data),
            vendor=vendor,
            traj_id=traj_id,
            instance_id=instance_id,
            resolved=resolved,
            model=model,
            text_limit=text_limit,
            max_steps=max_steps,
            chunk_chars=chunk_chars,
        )
    return _text_log_steps(
        text,
        vendor=vendor,
        traj_id=traj_id,
        instance_id=instance_id,
        resolved=resolved,
        model=model,
        text_limit=text_limit,
        max_steps=max_steps,
        chunk_chars=chunk_chars,
    )


def _write_parquet_parts(records: list[dict[str, Any]], out_dir: Path, stem: str, part_idx: int) -> Path:
    _mkdir(out_dir)
    path = out_dir / f"{stem}.part-{part_idx:04d}.parquet"
    frame = pd.DataFrame(records)
    frame.to_parquet(path, index=False)
    del frame
    _release_memory()
    return path


def _release_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


OPTIONAL_NUMERIC_COLUMNS = {
    "last_fail_count_this_step",
    "last_step_fail_count",
    "first_edit_step",
    "first_test_step",
    "first_run_python_step",
    "first_submit_step",
    "first_error_step",
    "first_traceback_step",
    "first_read_step",
    "steps_since_last_edit",
    "steps_since_last_test",
    "steps_since_last_submit",
    "steps_since_last_error",
    "steps_since_last_traceback",
    "steps_since_last_read",
    "last_fail_count",
    "best_fail_count_so_far",
    "fail_count_delta_from_prev_test",
}


def _normalise_nullable_feature_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in OPTIONAL_NUMERIC_COLUMNS.intersection(frame.columns):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame


def _normalise_nullable_arrow_columns(table: pa.Table) -> pa.Table:
    for column in OPTIONAL_NUMERIC_COLUMNS.intersection(table.column_names):
        idx = table.schema.get_field_index(column)
        table = table.set_column(idx, column, pc.cast(table[column], pa.float64()))
    return table


def _combine_parts(
    part_paths: list[Path],
    out_path: Path,
    *,
    exclude_instance_ids: set[str] | None = None,
    batch_size: int = 16,
) -> None:
    if not part_paths:
        pd.DataFrame().to_parquet(out_path, index=False)
        return
    writer: pq.ParquetWriter | None = None
    try:
        for part in part_paths:
            parquet_file = pq.ParquetFile(part)
            for batch in parquet_file.iter_batches(batch_size=batch_size):
                table = pa.Table.from_batches([batch])
                if exclude_instance_ids and "instance_id" in table.column_names:
                    values = pa.array(sorted(exclude_instance_ids), type=table.schema.field("instance_id").type)
                    keep = pc.invert(pc.is_in(table["instance_id"], value_set=values))
                    table = table.filter(keep)
                    if table.num_rows == 0:
                        del table
                        del batch
                        _release_memory()
                        continue
                table = _normalise_nullable_arrow_columns(table)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
                else:
                    table = table.cast(writer.schema)
                try:
                    writer.write_table(table)
                except pa.ArrowCapacityError:
                    if table.num_rows <= 1:
                        raise
                    for row_idx in range(table.num_rows):
                        writer.write_table(table.slice(row_idx, 1))
                del table
                del batch
                _release_memory()
    finally:
        if writer is not None:
            writer.close()


def _write_records_to_stream(
    records: list[dict[str, Any]],
    out_path: Path,
    writer: pq.ParquetWriter | None,
) -> pq.ParquetWriter | None:
    if not records:
        return writer
    if len(records) > 1:
        try:
            return _write_records_to_stream_no_split(records, out_path, writer)
        except pa.ArrowCapacityError:
            mid = max(1, len(records) // 2)
            writer = _write_records_to_stream(records[:mid], out_path, writer)
            return _write_records_to_stream(records[mid:], out_path, writer)
    return _write_records_to_stream_no_split(records, out_path, writer)


def _write_records_to_stream_no_split(
    records: list[dict[str, Any]],
    out_path: Path,
    writer: pq.ParquetWriter | None,
) -> pq.ParquetWriter | None:
    frame = pd.DataFrame(records)
    frame = _normalise_nullable_feature_columns(frame)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
    else:
        table = table.cast(writer.schema)
    writer.write_table(table)
    del table
    del frame
    _release_memory()
    return writer


def build_swe_prefix(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    prep_root = Path(args.prep_root)
    raw_root = data_root / "raw_sweverify_s3"
    prefix_dir = _mkdir(data_root / "prefix_tables")
    manifest_dir = _mkdir(data_root / "manifests")
    vendor = _vendor_imports()
    prompts = _load_verified_prompts(Path(args.verified_jsonl))
    swe_manifest = _read_csv(prep_root / "sweverify_harness_debug_unit_set_core18.csv")
    results_path = manifest_dir / "swe_results_by_instance.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}; run --download-swe-results first.")
    result_df = pd.read_csv(results_path)
    resolved_map = {
        (str(row.submission_folder), str(row.instance_id)): bool(row.resolved)
        for row in result_df.itertuples(index=False)
    }
    selected_path = manifest_dir / "swe_s3_selected_keys.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing {selected_path}; run --download-swe first.")
    selected_df = pd.read_csv(selected_path)
    selected_df["local_path"] = selected_df["local_path"].astype(str)
    selected_df = selected_df[selected_df["local_path"].map(lambda p: Path(p).exists())].copy()
    step_out = prefix_dir / "step_table_sweverify_harness_core18.parquet"
    prefix_out = prefix_dir / "prefix_table_sweverify_harness_core18.parquet"
    step_tmp = step_out.with_suffix(step_out.suffix + ".tmp")
    prefix_tmp = prefix_out.with_suffix(prefix_out.suffix + ".tmp")
    for path in [step_tmp, prefix_tmp]:
        if path.exists():
            path.unlink()
    step_writer: pq.ParquetWriter | None = None
    prefix_writer: pq.ParquetWriter | None = None
    step_buf: list[dict[str, Any]] = []
    prefix_buf: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    total_traj = 0
    total_steps = 0
    try:
        for _, unit in swe_manifest.iterrows():
            folder = str(unit["submission_folder"])
            unit_id = _swe_unit_id(str(unit["model_group"]), str(unit["heldout_unit"]))
            unit_files = selected_df[selected_df["submission_folder"].astype(str) == folder].copy()
            grouped: dict[str, list[Path]] = defaultdict(list)
            for rec in unit_files.itertuples(index=False):
                grouped[str(rec.instance_id)].append(Path(str(rec.local_path)))
            print(f"[swe-prefix] unit={unit_id} trajectories={len(grouped)}", flush=True)
            for local_idx, (instance_id, paths) in enumerate(sorted(grouped.items()), start=1):
                resolved = resolved_map.get((folder, instance_id), False)
                traj_id = f"{instance_id}__{unit_id}"
                task_prompt = prompts.get(instance_id, "")
                steps: list[dict[str, Any]] = []
                for path in sorted(paths):
                    remain = max(0, int(args.max_steps_per_traj) - len(steps))
                    if remain <= 0:
                        break
                    parsed = parse_file_to_steps(
                        path,
                        vendor=vendor,
                        traj_id=traj_id,
                        instance_id=instance_id,
                        resolved=resolved,
                        model=unit_id,
                        text_limit=int(args.max_step_text_chars),
                        max_steps=remain,
                        chunk_chars=int(args.fallback_log_chunk_chars),
                    )
                    for step in parsed:
                        step = dict(step)
                        step["step_idx"] = len(steps) + 1
                        steps.append(step)
                if not steps:
                    summary_rows.append(
                        {
                            "unit_id": unit_id,
                            "unit_shard_index": unit_shard_index,
                            "unit_shard_count": unit_shard_count,
                            "submission_folder": folder,
                            "instance_id": instance_id,
                            "parsed": False,
                            "steps": 0,
                            "files": len(paths),
                        }
                    )
                    continue
                total_traj += 1
                total_steps += len(steps)
                step_buf.extend(steps)
                for prefix_step_idx in range(0, len(steps) + 1):
                    prefix_buf.append(
                        vendor["_build_prefix_features"](
                            traj_id=traj_id,
                            instance_id=instance_id,
                            resolved=int(bool(resolved)),
                            model=unit_id,
                            task_prompt_text=task_prompt,
                            steps=steps[:prefix_step_idx],
                            prefix_step_idx=prefix_step_idx,
                            n_steps_total=len(steps),
                            sample_weight=1.0 / (len(steps) + 1),
                        )
                    )
                summary_rows.append(
                    {
                        "unit_id": unit_id,
                        "submission_folder": folder,
                        "instance_id": instance_id,
                        "parsed": True,
                        "steps": len(steps),
                        "files": len(paths),
                        "resolved": bool(resolved),
                    }
                )
                if len(step_buf) >= int(args.part_rows):
                    step_writer = _write_records_to_stream(step_buf, step_tmp, step_writer)
                    step_buf = []
                    _release_memory()
                if len(prefix_buf) >= int(args.part_rows):
                    prefix_writer = _write_records_to_stream(prefix_buf, prefix_tmp, prefix_writer)
                    prefix_buf = []
                    _release_memory()
                if local_idx == 1 or local_idx % 50 == 0 or local_idx == len(grouped_items):
                    print(
                        f"[swe-prefix] unit_progress={unit_id} {local_idx}/{len(grouped)} "
                        f"cumulative_traj={total_traj} cumulative_steps={total_steps}",
                        flush=True,
                    )
            print(f"[swe-prefix] unit={unit_id} cumulative_traj={total_traj} cumulative_steps={total_steps}", flush=True)
        if step_buf:
            step_writer = _write_records_to_stream(step_buf, step_tmp, step_writer)
            step_buf = []
            _release_memory()
        if prefix_buf:
            prefix_writer = _write_records_to_stream(prefix_buf, prefix_tmp, prefix_writer)
            prefix_buf = []
            _release_memory()
    finally:
        if step_writer is not None:
            step_writer.close()
        if prefix_writer is not None:
            prefix_writer.close()
    if not step_tmp.exists() or not prefix_tmp.exists():
        raise RuntimeError("SWE prefix build produced no parquet output.")
    step_tmp.replace(step_out)
    prefix_tmp.replace(prefix_out)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(manifest_dir / "swe_prefix_parse_summary.csv", index=False)
    print(
        f"[swe-prefix] wrote {prefix_out} trajectories={summary_df['parsed'].sum()} prefixes={pq.ParquetFile(prefix_out).metadata.num_rows}",
        flush=True,
    )


def _swe_manifest_with_unit_ids(args: argparse.Namespace) -> pd.DataFrame:
    prep_root = Path(args.prep_root)
    swe_manifest = _read_csv(prep_root / "sweverify_harness_debug_unit_set_core18.csv").copy()
    swe_manifest["unit_id"] = [
        _swe_unit_id(str(row.model_group), str(row.heldout_unit))
        for row in swe_manifest.itertuples(index=False)
    ]
    return swe_manifest


def _select_swe_manifest_units(args: argparse.Namespace, swe_manifest: pd.DataFrame) -> pd.DataFrame:
    selected = swe_manifest.copy()
    if str(args.swe_units).strip():
        wanted = {part.strip() for part in str(args.swe_units).split(",") if part.strip()}
        selected = selected[selected["unit_id"].astype(str).isin(wanted)].copy()
        missing = wanted.difference(set(selected["unit_id"].astype(str)))
        if missing:
            raise ValueError(f"Unknown SWE unit ids: {sorted(missing)}")
    shard_count = int(args.swe_shard_count)
    if shard_count > 1:
        shard_index = int(args.swe_shard_index)
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError("--swe-shard-index must be in [0, --swe-shard-count)")
        selected = selected.iloc[[i for i in range(len(selected)) if i % shard_count == shard_index]].copy()
    if selected.empty:
        raise ValueError("No SWE units selected.")
    return selected


def _validate_swe_unit_shard_args(args: argparse.Namespace) -> tuple[int, int]:
    shard_count = int(getattr(args, "swe_unit_shard_count", 1))
    shard_index = int(getattr(args, "swe_unit_shard_index", 0))
    if shard_count < 1:
        raise ValueError("--swe-unit-shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("--swe-unit-shard-index must be in [0, --swe-unit-shard-count)")
    return shard_index, shard_count


def _parse_csv_arg(value: str) -> set[str]:
    return {part.strip() for part in str(value).split(",") if part.strip()}


def _swe_unit_shard_stem(unit_id: str, shard_index: int, shard_count: int) -> str:
    if shard_count <= 1:
        return unit_id
    return f"{unit_id}.shard-{shard_index:03d}-of-{shard_count:03d}"


def list_swe_units(args: argparse.Namespace) -> None:
    for unit_id in _swe_manifest_with_unit_ids(args)["unit_id"].astype(str).tolist():
        print(unit_id, flush=True)


def build_swe_prefix_parts(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    raw_root = data_root / "raw_sweverify_s3"
    if not raw_root.exists():
        raise FileNotFoundError(raw_root)
    prefix_dir = _mkdir(data_root / "prefix_tables")
    manifest_dir = _mkdir(data_root / "manifests")
    parts_root = _mkdir(prefix_dir / "sweverify_harness_core18_unit_parts")
    step_parts_dir = _mkdir(parts_root / "step")
    prefix_parts_dir = _mkdir(parts_root / "prefix")
    summary_parts_dir = _mkdir(parts_root / "summary")
    success_dir = _mkdir(parts_root / "_SUCCESS")

    vendor = _vendor_imports()
    prompts = _load_verified_prompts(Path(args.verified_jsonl))
    swe_manifest = _select_swe_manifest_units(args, _swe_manifest_with_unit_ids(args))
    results_path = manifest_dir / "swe_results_by_instance.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}; run --download-swe-results first.")
    result_df = pd.read_csv(results_path)
    resolved_map = {
        (str(row.submission_folder), str(row.instance_id)): bool(row.resolved)
        for row in result_df.itertuples(index=False)
    }
    selected_path = manifest_dir / "swe_s3_selected_keys.csv"
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing {selected_path}; run --download-swe first.")
    selected_df = pd.read_csv(selected_path)
    selected_df["local_path"] = selected_df["local_path"].astype(str)
    selected_df = selected_df[selected_df["local_path"].map(lambda p: Path(p).exists())].copy()
    exclude_instance_ids = _parse_csv_arg(getattr(args, "swe_exclude_instance_ids", ""))
    unit_shard_index, unit_shard_count = _validate_swe_unit_shard_args(args)

    for _, unit in swe_manifest.iterrows():
        folder = str(unit["submission_folder"])
        unit_id = str(unit["unit_id"])
        part_stem = _swe_unit_shard_stem(unit_id, unit_shard_index, unit_shard_count)
        step_out = step_parts_dir / f"{part_stem}.parquet"
        prefix_out = prefix_parts_dir / f"{part_stem}.parquet"
        summary_out = summary_parts_dir / f"{part_stem}.csv"
        success_out = success_dir / f"{part_stem}.done"
        if (
            success_out.exists()
            and step_out.exists()
            and prefix_out.exists()
            and summary_out.exists()
            and not bool(args.force_swe_unit)
        ):
            print(f"[swe-prefix-part] skip completed unit={unit_id}", flush=True)
            continue
        for path in [step_out, prefix_out, summary_out, success_out]:
            if path.exists():
                path.unlink()
        step_tmp = step_out.with_suffix(step_out.suffix + ".tmp")
        prefix_tmp = prefix_out.with_suffix(prefix_out.suffix + ".tmp")
        for path in [step_tmp, prefix_tmp]:
            if path.exists():
                path.unlink()

        unit_files = selected_df[selected_df["submission_folder"].astype(str) == folder].copy()
        grouped: dict[str, list[Path]] = defaultdict(list)
        for rec in unit_files.itertuples(index=False):
            grouped[str(rec.instance_id)].append(Path(str(rec.local_path)))
        grouped_items = sorted(grouped.items())
        total_unit_trajectories = len(grouped_items)
        if unit_shard_count > 1:
            grouped_items = [
                item for idx, item in enumerate(grouped_items) if idx % unit_shard_count == unit_shard_index
            ]
        if exclude_instance_ids:
            grouped_items = [
                item for item in grouped_items if str(item[0]) not in exclude_instance_ids
            ]
        print(
            f"[swe-prefix-part] unit={unit_id} shard={unit_shard_index + 1}/{unit_shard_count} "
            f"trajectories={len(grouped_items)} total_unit_trajectories={total_unit_trajectories}",
            flush=True,
        )

        step_writer: pq.ParquetWriter | None = None
        prefix_writer: pq.ParquetWriter | None = None
        step_buf: list[dict[str, Any]] = []
        prefix_buf: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        unit_traj = 0
        unit_steps = 0
        try:
            for local_idx, (instance_id, paths) in enumerate(grouped_items, start=1):
                resolved = resolved_map.get((folder, instance_id), False)
                traj_id = f"{instance_id}__{unit_id}"
                task_prompt = prompts.get(instance_id, "")
                steps: list[dict[str, Any]] = []
                for path in sorted(paths):
                    remain = max(0, int(args.max_steps_per_traj) - len(steps))
                    if remain <= 0:
                        break
                    parsed = parse_file_to_steps(
                        path,
                        vendor=vendor,
                        traj_id=traj_id,
                        instance_id=instance_id,
                        resolved=resolved,
                        model=unit_id,
                        text_limit=int(args.max_step_text_chars),
                        max_steps=remain,
                        chunk_chars=int(args.fallback_log_chunk_chars),
                    )
                    for step in parsed:
                        step = dict(step)
                        step["step_idx"] = len(steps) + 1
                        steps.append(step)
                if not steps:
                    summary_rows.append(
                        {
                            "unit_id": unit_id,
                            "submission_folder": folder,
                            "instance_id": instance_id,
                            "parsed": False,
                            "steps": 0,
                            "files": len(paths),
                        }
                    )
                    continue
                unit_traj += 1
                unit_steps += len(steps)
                step_buf.extend(steps)
                for prefix_step_idx in range(0, len(steps) + 1):
                    prefix_buf.append(
                        vendor["_build_prefix_features"](
                            traj_id=traj_id,
                            instance_id=instance_id,
                            resolved=int(bool(resolved)),
                            model=unit_id,
                            task_prompt_text=task_prompt,
                            steps=steps[:prefix_step_idx],
                            prefix_step_idx=prefix_step_idx,
                            n_steps_total=len(steps),
                            sample_weight=1.0 / (len(steps) + 1),
                        )
                    )
                    if len(prefix_buf) >= int(args.part_rows):
                        prefix_writer = _write_records_to_stream(prefix_buf, prefix_tmp, prefix_writer)
                        prefix_buf = []
                        _release_memory()
                summary_rows.append(
                    {
                        "unit_id": unit_id,
                        "unit_shard_index": unit_shard_index,
                        "unit_shard_count": unit_shard_count,
                        "submission_folder": folder,
                        "instance_id": instance_id,
                        "parsed": True,
                        "steps": len(steps),
                        "files": len(paths),
                        "resolved": bool(resolved),
                    }
                )
                if len(step_buf) >= int(args.part_rows):
                    step_writer = _write_records_to_stream(step_buf, step_tmp, step_writer)
                    step_buf = []
                    _release_memory()
                if len(prefix_buf) >= int(args.part_rows):
                    prefix_writer = _write_records_to_stream(prefix_buf, prefix_tmp, prefix_writer)
                    prefix_buf = []
                    _release_memory()
                if local_idx == 1 or local_idx % 50 == 0 or local_idx == len(grouped):
                    print(
                        f"[swe-prefix-part] unit_progress={unit_id} shard={unit_shard_index + 1}/{unit_shard_count} "
                        f"{local_idx}/{len(grouped_items)} "
                        f"unit_traj={unit_traj} unit_steps={unit_steps}",
                        flush=True,
                    )
            if step_buf:
                step_writer = _write_records_to_stream(step_buf, step_tmp, step_writer)
                step_buf = []
                _release_memory()
            if prefix_buf:
                prefix_writer = _write_records_to_stream(prefix_buf, prefix_tmp, prefix_writer)
                prefix_buf = []
                _release_memory()
        finally:
            if step_writer is not None:
                step_writer.close()
            if prefix_writer is not None:
                prefix_writer.close()
        if not step_tmp.exists() or not prefix_tmp.exists():
            raise RuntimeError(f"SWE prefix part produced no parquet output: {unit_id}")
        step_tmp.replace(step_out)
        prefix_tmp.replace(prefix_out)
        pd.DataFrame(summary_rows).to_csv(summary_out, index=False)
        success_out.write_text("completed\n", encoding="utf-8")
        print(
            f"[swe-prefix-part] wrote unit={unit_id} shard={unit_shard_index + 1}/{unit_shard_count} "
            f"trajectories={unit_traj} "
            f"prefixes={pq.ParquetFile(prefix_out).metadata.num_rows}",
            flush=True,
        )


def combine_swe_unit_shards(args: argparse.Namespace) -> None:
    unit_shard_index, unit_shard_count = _validate_swe_unit_shard_args(args)
    if unit_shard_count <= 1:
        raise ValueError("--combine-swe-unit-shards requires --swe-unit-shard-count > 1")
    if unit_shard_index != 0:
        print(
            "[swe-prefix-shard-combine] --swe-unit-shard-index is ignored while combining shards",
            flush=True,
        )
    data_root = Path(args.data_root)
    prefix_dir = _mkdir(data_root / "prefix_tables")
    parts_root = prefix_dir / "sweverify_harness_core18_unit_parts"
    step_parts_dir = parts_root / "step"
    prefix_parts_dir = parts_root / "prefix"
    summary_parts_dir = parts_root / "summary"
    success_dir = parts_root / "_SUCCESS"
    swe_manifest = _select_swe_manifest_units(args, _swe_manifest_with_unit_ids(args))
    exclude_instance_ids = _parse_csv_arg(getattr(args, "swe_exclude_instance_ids", ""))

    for _, unit in swe_manifest.iterrows():
        unit_id = str(unit["unit_id"])
        shard_stems = [_swe_unit_shard_stem(unit_id, idx, unit_shard_count) for idx in range(unit_shard_count)]
        missing = []
        for stem in shard_stems:
            if not (success_dir / f"{stem}.done").exists():
                missing.append(str(success_dir / f"{stem}.done"))
            if not (step_parts_dir / f"{stem}.parquet").exists():
                missing.append(str(step_parts_dir / f"{stem}.parquet"))
            if not (prefix_parts_dir / f"{stem}.parquet").exists():
                missing.append(str(prefix_parts_dir / f"{stem}.parquet"))
            if not (summary_parts_dir / f"{stem}.csv").exists():
                missing.append(str(summary_parts_dir / f"{stem}.csv"))
        if missing:
            raise FileNotFoundError(f"Missing SWE unit shard outputs for {unit_id}: {missing}")

        step_out = step_parts_dir / f"{unit_id}.parquet"
        prefix_out = prefix_parts_dir / f"{unit_id}.parquet"
        summary_out = summary_parts_dir / f"{unit_id}.csv"
        success_out = success_dir / f"{unit_id}.done"
        step_tmp = step_out.with_suffix(step_out.suffix + ".shard_combine_tmp")
        prefix_tmp = prefix_out.with_suffix(prefix_out.suffix + ".shard_combine_tmp")
        for path in [step_tmp, prefix_tmp]:
            if path.exists():
                path.unlink()
        _combine_parts(
            [step_parts_dir / f"{stem}.parquet" for stem in shard_stems],
            step_tmp,
            exclude_instance_ids=exclude_instance_ids,
        )
        _combine_parts(
            [prefix_parts_dir / f"{stem}.parquet" for stem in shard_stems],
            prefix_tmp,
            exclude_instance_ids=exclude_instance_ids,
        )
        step_tmp.replace(step_out)
        prefix_tmp.replace(prefix_out)
        summary = pd.concat([pd.read_csv(summary_parts_dir / f"{stem}.csv") for stem in shard_stems], ignore_index=True)
        if exclude_instance_ids and "instance_id" in summary.columns:
            summary = summary[~summary["instance_id"].astype(str).isin(exclude_instance_ids)].copy()
        summary.to_csv(summary_out, index=False)
        success_out.write_text("completed\n", encoding="utf-8")
        print(
            f"[swe-prefix-shard-combine] wrote unit={unit_id} shards={unit_shard_count} "
            f"trajectories={int(summary['parsed'].sum())} prefixes={pq.ParquetFile(prefix_out).metadata.num_rows}",
            flush=True,
        )


def combine_swe_prefix_parts(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    prefix_dir = _mkdir(data_root / "prefix_tables")
    manifest_dir = _mkdir(data_root / "manifests")
    parts_root = prefix_dir / "sweverify_harness_core18_unit_parts"
    step_parts_dir = parts_root / "step"
    prefix_parts_dir = parts_root / "prefix"
    summary_parts_dir = parts_root / "summary"
    success_dir = parts_root / "_SUCCESS"
    swe_manifest = _swe_manifest_with_unit_ids(args)
    unit_ids = swe_manifest["unit_id"].astype(str).tolist()
    missing = [
        unit_id
        for unit_id in unit_ids
        if not (success_dir / f"{unit_id}.done").exists()
        or not (step_parts_dir / f"{unit_id}.parquet").exists()
        or not (prefix_parts_dir / f"{unit_id}.parquet").exists()
        or not (summary_parts_dir / f"{unit_id}.csv").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing completed SWE prefix parts: {missing}")
    step_out = prefix_dir / "step_table_sweverify_harness_core18.parquet"
    prefix_out = prefix_dir / "prefix_table_sweverify_harness_core18.parquet"
    step_tmp = step_out.with_suffix(step_out.suffix + ".combine_tmp")
    prefix_tmp = prefix_out.with_suffix(prefix_out.suffix + ".combine_tmp")
    for path in [step_tmp, prefix_tmp]:
        if path.exists():
            path.unlink()
    _combine_parts([step_parts_dir / f"{unit_id}.parquet" for unit_id in unit_ids], step_tmp)
    _combine_parts([prefix_parts_dir / f"{unit_id}.parquet" for unit_id in unit_ids], prefix_tmp)
    step_tmp.replace(step_out)
    prefix_tmp.replace(prefix_out)
    summary = pd.concat([pd.read_csv(summary_parts_dir / f"{unit_id}.csv") for unit_id in unit_ids], ignore_index=True)
    summary.to_csv(manifest_dir / "swe_prefix_parse_summary.csv", index=False)
    print(
        f"[swe-prefix-combine] wrote {prefix_out} trajectories={int(summary['parsed'].sum())} "
        f"prefixes={pq.ParquetFile(prefix_out).metadata.num_rows}",
        flush=True,
    )


def build_terminalbench_prefix(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    prep_root = Path(args.prep_root)
    prefix_dir = _mkdir(data_root / "prefix_tables")
    manifest_dir = _mkdir(data_root / "manifests")
    tb_manifest = _read_csv(prep_root / "terminalbench_harness_debug_unit_set_core16.csv")
    combo_to_unit = {
        (str(row.model_group), str(row.heldout_unit)): _tb_unit_id(str(row.model_group), str(row.heldout_unit))
        for row in tb_manifest.itertuples(index=False)
    }
    source = Path(args.terminalbench_prefix)
    pf = pq.ParquetFile(source)
    out_path = prefix_dir / "prefix_table_terminalbench_harness_core16.parquet"
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    total_traj: set[str] = set()
    summary: dict[str, dict[str, Any]] = defaultdict(lambda: {"prefixes": 0, "trajectories": set(), "instances": set()})
    try:
        for batch_idx, batch in enumerate(pf.iter_batches(batch_size=int(args.parquet_batch_size))):
            df = batch.to_pandas()
            keys = list(zip(df["model_id"].astype(str), df["terminalbench_agent"].astype(str)))
            unit_ids = [combo_to_unit.get(key) for key in keys]
            mask = pd.Series([unit is not None for unit in unit_ids], index=df.index)
            if not bool(mask.any()):
                continue
            out = df.loc[mask].copy()
            out_units = [unit for unit in np.array(unit_ids, dtype=object)[mask.to_numpy()] if unit is not None]
            original_model = out["model_id"].astype(str).to_numpy()
            original_traj = out["traj_id"].astype(str).to_numpy()
            out["harness_original_model_id"] = original_model
            out["harness_agent"] = out["terminalbench_agent"].astype(str)
            out["harness_model_group"] = original_model
            out["harness_heldout_unit"] = out["terminalbench_agent"].astype(str)
            out["model_id"] = out_units
            if "model" in out.columns:
                out["model"] = out_units
            out["traj_id"] = [f"{traj}__{unit}" for traj, unit in zip(original_traj, out_units)]
            if "group_id" in out.columns:
                out["group_id"] = out["traj_id"]
            if "prefix_id" in out.columns:
                out["prefix_id"] = out["traj_id"].astype(str) + "::p" + out["prefix_step_idx"].astype(str)
            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
            writer.write_table(table)
            total_rows += len(out)
            total_traj.update(out["traj_id"].astype(str).unique().tolist())
            for unit_id, part in out.groupby("model_id", sort=False):
                summary[str(unit_id)]["prefixes"] += int(len(part))
                summary[str(unit_id)]["trajectories"].update(part["traj_id"].astype(str).unique().tolist())
                summary[str(unit_id)]["instances"].update(part["instance_id"].astype(str).unique().tolist())
            if batch_idx == 0 or batch_idx % 10 == 0:
                print(f"[tb-prefix] batch={batch_idx} rows={total_rows}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    rows = []
    for unit_id, info in sorted(summary.items()):
        rows.append(
            {
                "unit_id": unit_id,
                "prefixes": info["prefixes"],
                "trajectories": len(info["trajectories"]),
                "instances": len(info["instances"]),
            }
        )
    pd.DataFrame(rows).to_csv(manifest_dir / "terminalbench_prefix_summary_core16.csv", index=False)
    print(f"[tb-prefix] wrote {out_path} prefixes={total_rows} trajectories={len(total_traj)}", flush=True)


def write_config(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    config_path = PROJECT_ROOT / "configs" / "harness_debug_cross_agent.yaml"
    rel_swe = Path("../data/harness_debug_cross_agent_20260625/prefix_tables/prefix_table_sweverify_harness_core18.parquet")
    rel_tb = Path("../data/harness_debug_cross_agent_20260625/prefix_tables/prefix_table_terminalbench_harness_core16.parquet")
    text = f"""run:
  name: "harness_debug_cross_agent"
  default_run_id: "harness_debug_cross_agent"
  output_root: "paper/experiments"
  results_root: "paper/results"
  seed: 42

runtime:
  python_executable: null

resources:
  execution: "serial"
  max_cpu_threads: 1
  lightgbm_num_threads: 1
  low_memory_audit: true
  write_success_markers: true

datasets:
  sweverify_harness_core18:
    role: "robustness"
    prefix_table: "{rel_swe.as_posix()}"
    split_enabled: true
    ablations_enabled: false
    answer_features_available: false
    model_col: "model_id"
    traj_col: "traj_id"
    instance_col: "instance_id"
    step_col: "prefix_step_idx"
    label_col: "label"

  terminalbench_harness_core16:
    role: "robustness"
    prefix_table: "{rel_tb.as_posix()}"
    split_enabled: true
    ablations_enabled: false
    answer_features_available: false
    model_col: "model_id"
    traj_col: "traj_id"
    instance_col: "instance_id"
    step_col: "prefix_step_idx"
    label_col: "label"

split:
  strategy: "leave_one_model_known_task"
  valid_ratio: 0.15
  train_valid_filter:
    allow_short_trajectory_filter: true
    min_steps: 2
    apply_to_test: false
"""
    config_path.write_text(text, encoding="utf-8")
    print(f"[config] wrote {config_path}", flush=True)
    print(f"[config] data_root={data_root}", flush=True)


def write_counts(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    manifest_dir = _mkdir(data_root / "manifests")
    rows = []
    for name, path in [
        ("sweverify_harness_core18", data_root / "prefix_tables" / "prefix_table_sweverify_harness_core18.parquet"),
        ("terminalbench_harness_core16", data_root / "prefix_tables" / "prefix_table_terminalbench_harness_core16.parquet"),
    ]:
        if not path.exists():
            continue
        pf = pq.ParquetFile(path)
        cols = ["traj_id", "model_id", "prefix_step_idx"]
        df = pq.read_table(path, columns=cols).to_pandas()
        rows.append(
            {
                "dataset": name,
                "trajectory_count": int(df["traj_id"].nunique()),
                "trajectory_prefix_count": int(len(df)),
                "heldout_units": int(df["model_id"].nunique()),
                "path": str(path),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(manifest_dir / "trajectory_prefix_counts_harness_debug.csv", index=False)
    print(out.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cross-agent harness debugging datasets.")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--prep-root", type=Path, default=PREP_ROOT_DEFAULT)
    parser.add_argument("--verified-jsonl", type=Path, default=SWE_VERIFIED_JSONL_DEFAULT)
    parser.add_argument("--terminalbench-prefix", type=Path, default=TB_PREFIX_DEFAULT)
    parser.add_argument("--parquet-batch-size", type=int, default=32768)
    parser.add_argument("--part-rows", type=int, default=50000)
    parser.add_argument("--max-steps-per-traj", type=int, default=200)
    parser.add_argument("--max-step-text-chars", type=int, default=50000)
    parser.add_argument("--fallback-log-chunk-chars", type=int, default=8000)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--download-swe", action="store_true")
    parser.add_argument("--download-swe-results", action="store_true")
    parser.add_argument("--build-swe-prefix", action="store_true")
    parser.add_argument("--build-swe-prefix-parts", action="store_true")
    parser.add_argument("--combine-swe-prefix-parts", action="store_true")
    parser.add_argument("--combine-swe-unit-shards", action="store_true")
    parser.add_argument("--list-swe-units", action="store_true")
    parser.add_argument("--swe-units", default="")
    parser.add_argument("--swe-shard-index", type=int, default=0)
    parser.add_argument("--swe-shard-count", type=int, default=1)
    parser.add_argument("--swe-unit-shard-index", type=int, default=0)
    parser.add_argument("--swe-unit-shard-count", type=int, default=1)
    parser.add_argument("--swe-exclude-instance-ids", default="")
    parser.add_argument("--force-swe-unit", action="store_true")
    parser.add_argument("--build-terminalbench-prefix", action="store_true")
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--write-counts", action="store_true")
    parser.add_argument("--all", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _mkdir(Path(args.data_root))
    if args.all:
        args.download_swe = True
        args.download_swe_results = True
        args.build_swe_prefix = True
        args.build_terminalbench_prefix = True
        args.write_config = True
        args.write_counts = True
    if args.list_swe_units:
        list_swe_units(args)
    if args.download_swe:
        download_swe(args)
    if args.download_swe_results:
        download_swe_results(args)
    if args.build_terminalbench_prefix:
        build_terminalbench_prefix(args)
    if args.build_swe_prefix:
        build_swe_prefix(args)
    if args.build_swe_prefix_parts:
        build_swe_prefix_parts(args)
    if args.combine_swe_unit_shards:
        combine_swe_unit_shards(args)
    if args.combine_swe_prefix_parts:
        combine_swe_prefix_parts(args)
    if args.write_config:
        write_config(args)
    if args.write_counts:
        write_counts(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
