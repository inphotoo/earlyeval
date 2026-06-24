#!/usr/bin/env python3
"""Zero/few-shot local LLM yes/no logit baseline for safe-stop.

This baseline consumes the compact text cache built by the BERT baseline
cache builder.  It does not train a classifier: it asks a local causal LM
whether a prefix is safely successful or safely failed, then converts the
next-token Yes/No logits into probabilities.  Valid split is used only for
calibration and policy selection; test stays locked for evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BERT_BASELINE_DIR = SCRIPT_DIR.parent / "bert_baselines"
sys.path.insert(0, str(BERT_BASELINE_DIR))

from bert_baseline_common import (  # noqa: E402
    DEFAULT_RESULTS_ROOT,
    binary_metric_row,
    resolve_device,
    safe_stop_outputs,
    set_threads,
    write_json,
)
from model_holdout_shadow_valid_retrain import _prediction_frame  # noqa: E402
from safe_stop_dual_head_retrain import _evaluate_policies, _head_column, _policy_grid  # noqa: E402


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
DEFAULT_CACHE = DEFAULT_RESULTS_ROOT / "llm_logit_smoke_cache"
DEFAULT_OUTPUT = DEFAULT_RESULTS_ROOT / "llm_logit_qwen25_coder_3b_smoke"

SYSTEM_PROMPT = (
    "You are a careful SWE-bench trajectory judge. "
    "You will be given the task, a gold answer summary, and the current prefix. "
    "Answer with exactly Yes or No."
)

SUCCESS_QUESTION = (
    "Question: Based only on the task, gold answer summary, and current prefix, "
    "is this prefix safe to stop as SUCCESS because it is already clearly on track "
    "to finish correctly?\nAnswer exactly Yes or No."
)

FAILURE_QUESTION = (
    "Question: Based only on the task, gold answer summary, and current prefix, "
    "is this prefix safe to stop as FAILURE because it is already clearly unlikely "
    "to finish correctly?\nAnswer exactly Yes or No."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--predictor-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--auto-batch", action="store_true", default=True)
    parser.add_argument("--no-auto-batch", dest="auto_batch", action="store_false")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--truncation-side", choices=("left", "right"), default="left")
    parser.add_argument("--prompt-mode", choices=("dual", "single_complement"), default="dual")
    parser.add_argument("--splits", nargs="+", choices=("valid", "test"), default=["valid", "test"])
    parser.add_argument("--trajectory-sample-frac", type=float, default=0.0)
    parser.add_argument("--trajectory-sample-seed", type=int, default=42)
    parser.add_argument("--trajectory-shard-count", type=int, default=1)
    parser.add_argument("--trajectory-shard-index", type=int, default=0)
    parser.add_argument("--progress-every-batches", type=int, default=10)
    parser.add_argument("--use-chat-template", action="store_true", default=True)
    parser.add_argument("--no-chat-template", dest="use_chat_template", action="store_false")
    parser.add_argument("--yes-candidates", nargs="+", default=["Yes", " yes", "YES", " yes."])
    parser.add_argument("--no-candidates", nargs="+", default=["No", " no", "NO", " no."])
    parser.add_argument("--score-modes", nargs="+", default=["raw", "calibrated"], choices=["raw", "calibrated"])
    parser.add_argument("--success-thresholds", nargs="+", type=float, default=[0.70, 0.80, 0.90, 0.95])
    parser.add_argument("--failure-thresholds", nargs="+", type=float, default=[0.70, 0.80, 0.90, 0.95])
    parser.add_argument("--policy-min-steps", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--consecutive", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--max-valid-abs-drop-pp", type=float, default=3.0)
    parser.add_argument("--min-valid-decision-acc", type=float, default=0.0)
    parser.add_argument("--fallback-min-save-pct", type=float, default=0.0)
    parser.add_argument("--limit-rows-per-split", type=int, default=0)
    parser.add_argument("--max-cpu-threads", type=int, default=16)
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return cleaned or "LocalLLMLogit"


def dtype_from_arg(dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(dtype)


def _sample_trajectories(frame: pd.DataFrame, frac: float, seed: int, split_name: str) -> pd.DataFrame:
    if frac <= 0.0 or frac >= 1.0:
        return frame
    traj_ids = np.array(sorted(frame["traj_id"].astype(str).unique()))
    n_keep = max(1, int(round(len(traj_ids) * float(frac))))
    rng = np.random.default_rng(int(seed))
    keep = set(rng.choice(traj_ids, size=n_keep, replace=False).tolist())
    sampled = frame[frame["traj_id"].astype(str).isin(keep)].copy()
    if sampled.empty:
        raise RuntimeError(f"Trajectory sampling produced no rows for split={split_name}.")
    return sampled.reset_index(drop=True)


def _shard_trajectories(frame: pd.DataFrame, shard_count: int, shard_index: int, split_name: str) -> pd.DataFrame:
    shard_count = int(shard_count)
    shard_index = int(shard_index)
    if shard_count <= 1:
        return frame
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"Invalid shard index {shard_index} for shard_count={shard_count}.")
    traj_ids = np.array(sorted(frame["traj_id"].astype(str).unique()))
    keep = set(traj_ids[shard_index::shard_count].tolist())
    sharded = frame[frame["traj_id"].astype(str).isin(keep)].copy()
    if sharded.empty:
        raise RuntimeError(f"Trajectory sharding produced no rows for split={split_name}.")
    return sharded.reset_index(drop=True)


def load_text_cache(
    cache_dir: Path,
    limit_rows_per_split: int,
    splits: list[str],
    trajectory_sample_frac: float,
    trajectory_sample_seed: int,
    trajectory_shard_count: int,
    trajectory_shard_index: int,
) -> dict[str, pd.DataFrame]:
    path = cache_dir / "bert_text_cache.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing text cache: {path}")
    frame = pd.read_parquet(path)
    out: dict[str, pd.DataFrame] = {}
    for split_name in splits:
        split = frame[frame["split"] == split_name].sort_values("row_idx").reset_index(drop=True)
        split = _sample_trajectories(
            split,
            frac=trajectory_sample_frac,
            seed=trajectory_sample_seed,
            split_name=split_name,
        )
        split = _shard_trajectories(
            split,
            shard_count=trajectory_shard_count,
            shard_index=trajectory_shard_index,
            split_name=split_name,
        )
        if limit_rows_per_split > 0:
            split = split.head(int(limit_rows_per_split)).copy()
        if split.empty:
            raise RuntimeError(f"Text cache has no rows for split={split_name}")
        out[split_name] = split
    return out


def raw_test_only_outputs(
    *,
    output_dir: Path,
    run_label: str,
    predictor_name: str,
    test_frame: pd.DataFrame,
    test_success_raw: np.ndarray,
    test_failure_raw: np.ndarray,
    success_thresholds: list[float],
    failure_thresholds: list[float],
    policy_min_steps: list[int],
    consecutive: list[int],
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    test_pred = _prediction_frame(test_frame)
    test_pred[_head_column("success", "raw", predictor_name)] = test_success_raw.astype(np.float32)
    test_pred[_head_column("failure", "raw", predictor_name)] = test_failure_raw.astype(np.float32)
    test_pred.to_parquet(output_dir / "test_predictions_safe_stop.parquet", index=False)

    metric_rows = [
        binary_metric_row(
            f"{predictor_name}__safe_success",
            "test_raw",
            test_frame["safe_success_label"].to_numpy(dtype=np.int8),
            test_success_raw,
        ),
        binary_metric_row(
            f"{predictor_name}__safe_failure",
            "test_raw",
            test_frame["safe_failure_label"].to_numpy(dtype=np.int8),
            test_failure_raw,
        ),
    ]
    pd.DataFrame(metric_rows).to_csv(output_dir / "head_metrics.csv", index=False)

    policies = _policy_grid(
        success_thresholds=success_thresholds,
        failure_thresholds=failure_thresholds,
        min_steps=policy_min_steps,
        consecutive_values=consecutive,
    )
    test_grid, test_per_agent = _evaluate_policies(
        test_pred,
        run_label=run_label,
        predictors=[predictor_name],
        score_modes=["raw"],
        policies=policies,
    )
    test_grid["drop_pp"] = test_grid["resolve_rate_drop"] * 100.0
    test_grid["abs_drop_pp"] = test_grid["drop_pp"].abs()
    test_grid.to_csv(output_dir / "safe_stop_test_policy_grid.csv", index=False)
    test_per_agent.to_csv(output_dir / "safe_stop_test_policy_per_agent.csv", index=False)

    display = test_grid.sort_values(
        ["abs_drop_pp", "pct_steps_saved", "decision_accuracy"],
        ascending=[True, False, False],
    ).head(20)
    lines = [
        "# Local LLM Logit Test-Only Report",
        "",
        "This run evaluates fixed raw Yes/No logit policies on test only.",
        "No training, calibration, or validation-threshold selection is performed.",
        "Do not treat rows selected from this table as valid-selected locked policies.",
        "",
        "| Model | Mode | S_thr | F_thr | Min | K | Test Save | Test Drop pp | Test Acc | Test FN | Test FP |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in display.iterrows():
        failure = "-" if np.isinf(float(row["failure_thr"])) else f"{float(row['failure_thr']):.2f}"
        acc = "-" if pd.isna(row["decision_accuracy"]) else f"{float(row['decision_accuracy']) * 100.0:.1f}%"
        lines.append(
            f"| {row['prefix_model']} | {row['policy_mode']} | {float(row['success_thr']):.2f} | "
            f"{failure} | {int(row['min_step'])} | {int(row['consecutive'])} | "
            f"{float(row['pct_steps_saved']):.1f}% | {float(row['drop_pp']):.1f} | "
            f"{acc} | {int(row['false_negatives'])} | {int(row['false_positives'])} |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `test_predictions_safe_stop.parquet`",
            "- `safe_stop_test_policy_grid.csv`",
            "- `safe_stop_test_policy_per_agent.csv`",
            "- `head_metrics.csv`",
        ]
    )
    (output_dir / "safe_stop_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return test_grid


def context_from_row(row: pd.Series) -> str:
    text_a = str(row.get("text_a", "") or "").strip()
    text_b = str(row.get("text_b", "") or "").strip()
    if text_a and text_b:
        return f"{text_a}\n\n{text_b}"
    return text_a or text_b


def plain_prompt(user_prompt: str) -> str:
    return f"{SYSTEM_PROMPT}\n\n{user_prompt}\n\nAnswer:"


def format_prompt(tokenizer: Any, user_prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return plain_prompt(user_prompt)


def build_user_prompt(row: pd.Series, question: str) -> str:
    return (
        "We are evaluating whether an agent trajectory can be stopped early.\n\n"
        f"{context_from_row(row)}\n\n"
        f"{question}"
    )


def label_token_ids(tokenizer: Any, candidates: list[str], label: str) -> list[int]:
    ids: set[int] = set()
    for candidate in candidates:
        token_ids = tokenizer.encode(candidate, add_special_tokens=False)
        for token_id in token_ids:
            decoded = tokenizer.decode([int(token_id)], skip_special_tokens=True).strip().lower()
            if decoded == label:
                ids.add(int(token_id))
    if not ids:
        fallback_ids = tokenizer.encode(label.capitalize(), add_special_tokens=False)
        for token_id in fallback_ids:
            decoded = tokenizer.decode([int(token_id)], skip_special_tokens=True).strip().lower()
            if decoded == label:
                ids.add(int(token_id))
    return sorted(ids)


def load_causal_lm(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.import_utils import is_accelerate_available

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(
            "[llm-logit] CUDA_VISIBLE_DEVICES="
            f"{os.environ.get('CUDA_VISIBLE_DEVICES', '')}; "
            f"torch_device_count={torch.cuda.device_count()}; "
            f"cuda:0={props.name}; total_mem_gib={props.total_memory / 1024**3:.2f}",
            file=sys.stderr,
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=not args.allow_download,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = args.truncation_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    low_cpu_mem_usage = bool(is_accelerate_available())
    if not low_cpu_mem_usage:
        print(
            "[llm-logit] accelerate is too old for low_cpu_mem_usage=True; "
            "falling back to standard loading.",
            file=sys.stderr,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        local_files_only=not args.allow_download,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype_from_arg(args.dtype),
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    model.to(device)
    return tokenizer, model


def yes_probability_from_logits(logits: torch.Tensor, yes_ids: list[int], no_ids: list[int]) -> np.ndarray:
    yes = torch.logsumexp(logits[:, yes_ids], dim=-1)
    no = torch.logsumexp(logits[:, no_ids], dim=-1)
    probs = torch.sigmoid(yes - no)
    return probs.detach().float().cpu().numpy().astype(np.float32)


def _is_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in text or "cublas_status_alloc_failed" in text


def _clear_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "?"
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _log_progress(
    *,
    split_name: str,
    head_name: str,
    done_rows: int,
    total_rows: int,
    done_batches: int,
    total_batches: int,
    elapsed: float,
    batch_size: int,
    effective_batch_size: int,
) -> None:
    rows_per_second = done_rows / elapsed if elapsed > 0 else float("nan")
    remaining_rows = max(total_rows - done_rows, 0)
    eta = remaining_rows / rows_per_second if rows_per_second > 0 else float("nan")
    print(
        "[llm-logit]"
        f"[{split_name}:{head_name}] "
        f"{done_rows}/{total_rows} rows ({done_rows * 100.0 / total_rows:.1f}%), "
        f"batch {done_batches}/{total_batches}, "
        f"req_batch={batch_size}, eff_batch={effective_batch_size}, "
        f"elapsed={_format_seconds(elapsed)}, eta={_format_seconds(eta)}",
        file=sys.stderr,
        flush=True,
    )


def _next_token_logits(
    *,
    model: torch.nn.Module,
    encoded: dict[str, torch.Tensor],
) -> torch.Tensor:
    base_model = getattr(model, "model", None)
    output_embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if base_model is not None and output_embeddings is not None:
        base_inputs = {
            key: value
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask", "position_ids"}
        }
        outputs = base_model(**base_inputs, return_dict=True, use_cache=False)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        return output_embeddings(last_hidden)

    output = model(**encoded, return_dict=True, use_cache=False)
    return output.logits[:, -1, :]


def _score_prompt_batch(
    *,
    prompts: list[str],
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    yes_ids: list[int],
    no_ids: list[int],
    args: argparse.Namespace,
    effective_batch_sizes: list[int],
) -> tuple[np.ndarray, list[int]]:
    try:
        encoded = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
            return_tensors="pt",
        )
        token_counts = encoded["attention_mask"].sum(dim=1).cpu().numpy().astype(int).tolist()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            next_logits = _next_token_logits(model=model, encoded=encoded)
            batch_probs = yes_probability_from_logits(next_logits, yes_ids, no_ids)
        effective_batch_sizes.append(len(prompts))
        del encoded, next_logits
        _clear_cuda(device)
        return batch_probs, token_counts
    except RuntimeError as exc:
        _clear_cuda(device)
        if args.auto_batch and len(prompts) > 1 and _is_oom_error(exc):
            next_size = max(1, len(prompts) // 2)
            print(
                f"[llm-logit] CUDA OOM at batch={len(prompts)}; retrying as {next_size}+{len(prompts) - next_size}.",
                file=sys.stderr,
                flush=True,
            )
            left_probs, left_counts = _score_prompt_batch(
                prompts=prompts[:next_size],
                tokenizer=tokenizer,
                model=model,
                device=device,
                yes_ids=yes_ids,
                no_ids=no_ids,
                args=args,
                effective_batch_sizes=effective_batch_sizes,
            )
            right_probs, right_counts = _score_prompt_batch(
                prompts=prompts[next_size:],
                tokenizer=tokenizer,
                model=model,
                device=device,
                yes_ids=yes_ids,
                no_ids=no_ids,
                args=args,
                effective_batch_sizes=effective_batch_sizes,
            )
            return np.concatenate([left_probs, right_probs]).astype(np.float32), left_counts + right_counts
        if args.auto_batch and len(prompts) == 1 and _is_oom_error(exc):
            print(
                "[llm-logit] CUDA OOM even at batch=1. Reduce MAX_INPUT_TOKENS or use a larger GPU.",
                file=sys.stderr,
                flush=True,
            )
        raise


def infer_yes_probs(
    *,
    split_name: str,
    head_name: str,
    frame: pd.DataFrame,
    question: str,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    yes_ids: list[int],
    no_ids: list[int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    probs: list[np.ndarray] = []
    token_counts: list[int] = []
    effective_batch_sizes: list[int] = []
    start_time = time.time()
    total_rows = len(frame)
    prompts = [
        format_prompt(tokenizer, build_user_prompt(row, question), args.use_chat_template)
        for _, row in frame.iterrows()
    ]
    total_batches = max(1, math.ceil(len(prompts) / max(int(args.batch_size), 1)))
    for batch_index, start in enumerate(range(0, len(prompts), args.batch_size), start=1):
        batch_prompts = prompts[start : start + args.batch_size]
        batch_probs, batch_token_counts = _score_prompt_batch(
            prompts=batch_prompts,
            tokenizer=tokenizer,
            model=model,
            device=device,
            yes_ids=yes_ids,
            no_ids=no_ids,
            args=args,
            effective_batch_sizes=effective_batch_sizes,
        )
        token_counts.extend(batch_token_counts)
        probs.append(batch_probs)
        if (
            args.progress_every_batches > 0
            and (batch_index % args.progress_every_batches == 0 or batch_index == total_batches)
        ):
            done_rows = min(start + len(batch_prompts), total_rows)
            elapsed = time.time() - start_time
            _log_progress(
                split_name=split_name,
                head_name=head_name,
                done_rows=done_rows,
                total_rows=total_rows,
                done_batches=batch_index,
                total_batches=total_batches,
                elapsed=elapsed,
                batch_size=int(args.batch_size),
                effective_batch_size=int(effective_batch_sizes[-1]) if effective_batch_sizes else len(batch_prompts),
            )
    elapsed = time.time() - start_time
    arr = np.concatenate(probs).astype(np.float32)
    summary = {
        "rows": int(len(frame)),
        "elapsed_seconds": float(elapsed),
        "rows_per_second": float(len(frame) / elapsed) if elapsed > 0 else float("nan"),
        "mean_input_tokens": float(np.mean(token_counts)) if token_counts else float("nan"),
        "p95_input_tokens": float(np.percentile(token_counts, 95)) if token_counts else float("nan"),
        "max_input_tokens_seen": int(max(token_counts)) if token_counts else 0,
        "requested_batch_size": int(args.batch_size),
        "min_effective_batch_size": int(min(effective_batch_sizes)) if effective_batch_sizes else 0,
        "max_effective_batch_size": int(max(effective_batch_sizes)) if effective_batch_sizes else 0,
    }
    return arr, summary


def main() -> int:
    args = parse_args()
    set_threads(args.max_cpu_threads)
    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = list(dict.fromkeys(args.splits))
    frames = load_text_cache(
        args.cache_dir,
        args.limit_rows_per_split,
        splits,
        args.trajectory_sample_frac,
        args.trajectory_sample_seed,
        args.trajectory_shard_count,
        args.trajectory_shard_index,
    )
    tokenizer, model = load_causal_lm(args, device)
    yes_ids = label_token_ids(tokenizer, args.yes_candidates, "yes")
    no_ids = label_token_ids(tokenizer, args.no_candidates, "no")
    if not yes_ids or not no_ids:
        raise RuntimeError(f"Could not identify Yes/No token ids. yes_ids={yes_ids}, no_ids={no_ids}")

    predictor_name = args.predictor_name or f"{sanitize_name(args.model_name)}_LogitJudge"
    inference_rows: list[dict[str, Any]] = []
    split_probs: dict[str, dict[str, np.ndarray]] = {}
    for split_name, frame in frames.items():
        success_prob, success_summary = infer_yes_probs(
            split_name=split_name,
            head_name="safe_success",
            frame=frame,
            question=SUCCESS_QUESTION,
            tokenizer=tokenizer,
            model=model,
            device=device,
            yes_ids=yes_ids,
            no_ids=no_ids,
            args=args,
        )
        success_summary.update({"split": split_name, "head": "safe_success"})
        inference_rows.append(success_summary)
        if args.prompt_mode == "dual":
            failure_prob, failure_summary = infer_yes_probs(
                split_name=split_name,
                head_name="safe_failure",
                frame=frame,
                question=FAILURE_QUESTION,
                tokenizer=tokenizer,
                model=model,
                device=device,
                yes_ids=yes_ids,
                no_ids=no_ids,
                args=args,
            )
            failure_summary.update({"split": split_name, "head": "safe_failure"})
            inference_rows.append(failure_summary)
        else:
            failure_prob = (1.0 - success_prob).astype(np.float32)
            inference_rows.append(
                {
                    "split": split_name,
                    "head": "safe_failure",
                    "rows": int(len(frame)),
                    "elapsed_seconds": 0.0,
                    "rows_per_second": float("inf"),
                    "mean_input_tokens": float("nan"),
                    "p95_input_tokens": float("nan"),
                    "max_input_tokens_seen": 0,
                    "derived_from": "1 - safe_success",
                }
            )
        split_probs[split_name] = {
            "success": success_prob,
            "failure": failure_prob,
        }

    pd.DataFrame(inference_rows).to_csv(output_dir / "logit_inference_summary.csv", index=False)
    write_json(
        output_dir / "logit_label_token_ids.json",
        {
            "yes_ids": yes_ids,
            "no_ids": no_ids,
            "yes_decoded": [tokenizer.decode([idx], skip_special_tokens=True) for idx in yes_ids],
            "no_decoded": [tokenizer.decode([idx], skip_special_tokens=True) for idx in no_ids],
            "yes_candidates": args.yes_candidates,
            "no_candidates": args.no_candidates,
        },
    )
    write_json(
        output_dir / "llm_logit_config.json",
        {
            "model_name": args.model_name,
            "predictor_name": predictor_name,
            "cache_dir": str(args.cache_dir),
            "device": str(device),
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "max_input_tokens": args.max_input_tokens,
            "truncation_side": args.truncation_side,
            "prompt_mode": args.prompt_mode,
            "use_chat_template": bool(args.use_chat_template),
            "score_modes": args.score_modes,
            "splits": splits,
            "trajectory_sample_frac": args.trajectory_sample_frac,
            "trajectory_sample_seed": args.trajectory_sample_seed,
            "trajectory_shard_count": args.trajectory_shard_count,
            "trajectory_shard_index": args.trajectory_shard_index,
            "progress_every_batches": args.progress_every_batches,
            "success_thresholds": args.success_thresholds,
            "failure_thresholds": args.failure_thresholds,
            "policy_min_steps": args.policy_min_steps,
            "consecutive": args.consecutive,
            "max_valid_abs_drop_pp": args.max_valid_abs_drop_pp,
            "limit_rows_per_split": args.limit_rows_per_split,
        },
    )

    if "valid" in frames and "test" in frames:
        safe_stop_outputs(
            output_dir=output_dir,
            run_label="llm_logit_final",
            predictor_name=predictor_name,
            valid_frame=frames["valid"],
            test_frame=frames["test"],
            valid_success_raw=split_probs["valid"]["success"],
            valid_failure_raw=split_probs["valid"]["failure"],
            test_success_raw=split_probs["test"]["success"],
            test_failure_raw=split_probs["test"]["failure"],
            score_modes=args.score_modes,
            success_thresholds=args.success_thresholds,
            failure_thresholds=args.failure_thresholds,
            policy_min_steps=args.policy_min_steps,
            consecutive=args.consecutive,
            max_valid_abs_drop_pp=args.max_valid_abs_drop_pp,
            min_valid_decision_acc=args.min_valid_decision_acc,
            fallback_min_save_pct=args.fallback_min_save_pct,
        )
    elif splits == ["test"]:
        raw_test_only_outputs(
            output_dir=output_dir,
            run_label="llm_logit_final_test_only",
            predictor_name=predictor_name,
            test_frame=frames["test"],
            test_success_raw=split_probs["test"]["success"],
            test_failure_raw=split_probs["test"]["failure"],
            success_thresholds=args.success_thresholds,
            failure_thresholds=args.failure_thresholds,
            policy_min_steps=args.policy_min_steps,
            consecutive=args.consecutive,
        )
    else:
        raise RuntimeError("Need either both valid+test for valid-selected evaluation, or test only.")
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
