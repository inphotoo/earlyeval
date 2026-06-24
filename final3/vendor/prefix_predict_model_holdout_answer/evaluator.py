'Public-release English note.'
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve

import config
from utils import get_logger, save_json

logger = get_logger("evaluator")

DEFAULT_DECISION_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

# Public-release English note.
DGKN_MODEL_KEYS = (
    "D_Dense_Full_LR",
    "G_TfIdf_Full_LR",
    "K_LightGBM_Dense_Full",
    "N_LightGBM_TfIdf_Full",
)


def per_step_accuracy_dgkn(
    df: pd.DataFrame,
    y_true: np.ndarray | None = None,
    *,
    step_col: str = "prefix_step_idx",
    label_col: str = "label",
    prob_prefix: str = "prob__",
    threshold: float = 0.5,
    models: tuple[str, ...] = DGKN_MODEL_KEYS,
) -> pd.DataFrame:
    'Public-release English note.'
    if step_col not in df.columns:
        raise KeyError(f"missing column {step_col}")
    if y_true is None:
        if label_col not in df.columns:
            raise KeyError(f"y_true not passed and missing {label_col}")
        y = np.asarray(df[label_col].values, dtype=np.float64)
    else:
        y = np.asarray(y_true, dtype=np.float64)
        if len(y) != len(df):
            raise ValueError("y_true length mismatch with df")

    rows: list[dict] = []
    idx_order = sorted(df[step_col].unique())

    for step in idx_order:
        m = df[step_col].values == step
        if not m.any():
            continue
        g = y[m]
        n = int(m.sum())
        n_pos = int(np.sum(g))
        row: dict = {
            "prefix_step_idx": int(step) if not isinstance(step, (np.floating, float)) else float(step),
            "n_samples": n,
            "n_positive_labels": n_pos,
        }
        sub = df.loc[m].reset_index(drop=True)
        y_sub = y[m]
        for model in models:
            pcol = f"{prob_prefix}{model}"
            if pcol not in df.columns:
                row[f"mean_prob__{model}"] = np.nan
                row[f"accuracy__{model}"] = np.nan
                continue
            p = np.asarray(sub[pcol].values, dtype=np.float64)
            row[f"mean_prob__{model}"] = float(np.mean(p))
            pred = (p >= threshold).astype(np.float64)
            row[f"accuracy__{model}"] = float(np.mean(pred == y_sub))
        row["decision_threshold"] = float(threshold)
        rows.append(row)

    return pd.DataFrame(rows)


# Public-release English note.
DGKN_SHORT_TO_MODEL = (
    ("D", "D_Dense_Full_LR"),
    ("G", "G_TfIdf_Full_LR"),
    ("K", "K_LightGBM_Dense_Full"),
    ("N", "N_LightGBM_TfIdf_Full"),
)


def _step_content_from_prefix_row(row: pd.Series, max_chars: int = 12000) -> str:
    'Public-release English note.'
    idx = int(row.get("prefix_step_idx", 0) or 0)
    task = str(row.get("task_prompt_text", "") or "").strip()
    thought = str(row.get("last_thought_text", "") or "").strip()
    action = str(row.get("last_action_text", "") or "").strip()
    feedback = str(row.get("last_feedback_text", "") or "").strip()

    if idx == 0:
        parts = ['Public-release English note.']
        if task:
            parts.append('Public-release English note.')
    else:
        parts = []
        if thought:
            parts.append(f"[last thought]\n{thought}")
        if action:
            parts.append(f"[last action]\n{action}")
        if feedback:
            parts.append(f"[last feedback]\n{feedback}")
        if not parts:
            parts.append('Public-release English note.')
    text = "\n\n".join(parts)
    if len(text) > max_chars:
        text = text[: max_chars - 24].rstrip() + "\n...(truncated)"
    return text


def per_step_dgkn_convert_export(
    df: pd.DataFrame,
    *,
    prob_prefix: str = "prob__",
    step_col: str = "prefix_step_idx",
    max_content_chars: int = 12000,
    drop_traj_if_single: bool = True,
) -> pd.DataFrame:
    'Public-release English note.'
    if step_col not in df.columns:
        raise KeyError(f"missing {step_col}")

    sort_cols = ["traj_id", step_col] if "traj_id" in df.columns else [step_col]
    out_rows: list[dict] = []
    for _, row in df.sort_values(sort_cols).iterrows():
        rec: dict = {"idx": int(row[step_col])}
        if "traj_id" in df.columns:
            rec["traj_id"] = row["traj_id"]
        for short, full in DGKN_SHORT_TO_MODEL:
            pcol = f"{prob_prefix}{full}"
            if pcol in df.columns and pd.notna(row.get(pcol)):
                rec[f"p_{short}"] = float(row[pcol])
            else:
                rec[f"p_{short}"] = np.nan
        rec["step_content"] = _step_content_from_prefix_row(row, max_chars=max_content_chars)
        out_rows.append(rec)

    exp = pd.DataFrame(out_rows)
    if drop_traj_if_single and "traj_id" in exp.columns and exp["traj_id"].nunique() <= 1:
        exp = exp.drop(columns=["traj_id"])

    col_order: list[str] = []
    if "traj_id" in exp.columns:
        col_order.append("traj_id")
    col_order.append("idx")
    col_order.extend(f"p_{s}" for s, _ in DGKN_SHORT_TO_MODEL)
    col_order.append("step_content")
    col_order = [c for c in col_order if c in exp.columns]
    return exp[col_order]


def instance_id_to_repo_key(instance_id: str) -> str:
    'Public-release English note.'
    s = (instance_id or "").strip()
    if not s:
        return "_empty"
    if "__" in s:
        head = s.split("__", 1)[0].strip()
        return head if head else "_empty"
    return s


def compute_stratified_metrics_by_repo(
    df: pd.DataFrame,
    all_predictions: dict[str, np.ndarray],
    *,
    min_prefix_rows: int = 50,
    min_pos: int = 5,
    min_neg: int = 5,
    all_rank_for_auc: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    'Public-release English note.'
    if "instance_id" not in df.columns or "label" not in df.columns:
        logger.warning('Public-release English note.')
        return pd.DataFrame()
    n_rows = len(df)
    if n_rows == 0:
        return pd.DataFrame()

    y_true = np.asarray(df["label"].values, dtype=np.float64)
    keys = df["instance_id"].map(instance_id_to_repo_key)
    rows: list[dict] = []

    for repo in sorted(keys.unique()):
        mask = (keys == repo).values
        n = int(mask.sum())
        if n < min_prefix_rows:
            continue
        yt = y_true[mask]
        n_pos = int(yt.sum())
        n_neg = int(n - n_pos)
        if n_pos < min_pos or n_neg < min_neg:
            continue
        n_traj = (
            int(df.loc[mask, "traj_id"].nunique()) if "traj_id" in df.columns else -1
        )
        for model_name, y_prob in all_predictions.items():
            yp = np.asarray(y_prob, dtype=np.float64)[mask]
            yr = None
            if all_rank_for_auc is not None and model_name in all_rank_for_auc:
                yr = np.asarray(all_rank_for_auc[model_name], dtype=np.float64)[mask]
            try:
                m = compute_metrics(yt, yp, y_rank_for_auc=yr)
            except Exception as e:
                logger.debug("stratified metrics skip repo=%s model=%s: %s", repo, model_name, e)
                continue
            rows.append(
                {
                    "repo_key": repo,
                    "model_name": model_name,
                    "n_traj": n_traj,
                    "roc_auc": m.get("roc_auc"),
                    "pr_auc": m.get("pr_auc"),
                    "log_loss": m.get("log_loss"),
                    "brier_score": m.get("brier_score"),
                    "n_samples": m.get("n_samples"),
                    "n_positive": m.get("n_positive"),
                    "n_negative": m.get("n_negative"),
                    "pos_rate": m.get("pos_rate"),
                }
            )
    return pd.DataFrame(rows)


def build_stratified_repo_report_lines(
    df_long: pd.DataFrame,
    *,
    highlight_models: tuple[str, ...] = ("I_LightGBM_Dense_AF", "A_Dense_LR"),
    top_repos: int = 30,
) -> list[str]:
    'Public-release English note.'
    if df_long is None or df_long.empty:
        return []
    lines: list[str] = [
        "## Stratified by SWE repo (instance_id prefix)",
        "",
        'Public-release English note.',
        'Public-release English note.',
        "",
    ]
    for hm in highlight_models:
        sub = df_long[df_long["model_name"] == hm].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("n_samples", ascending=False).head(int(top_repos))
        lines.append(f"### {hm} - by repo (top {len(sub)} by N)")
        lines.append("")
        hdr = (
            f"{'repo_key':<30} {'ROC-AUC':>8} {'PR-AUC':>8} {'N':>8} "
            f"{'traj':>6} {'+':>5} {'-':>5}"
        )
        lines.append(hdr)
        lines.append("-" * len(hdr))
        for _, r in sub.iterrows():
            ra = r.get("roc_auc")
            pa = r.get("pr_auc")
            ra_s = (
                f"{float(ra):>8.4f}"
                if ra is not None and pd.notna(ra)
                else f"{'N/A':>8}"
            )
            pa_s = (
                f"{float(pa):>8.4f}"
                if pa is not None and pd.notna(pa)
                else f"{'N/A':>8}"
            )
            lines.append(
                f"{str(r['repo_key']):<30} {ra_s} {pa_s} "
                f"{int(r['n_samples']):>8d} {int(r['n_traj']):>6d} "
                f"{int(r['n_positive']):>5d} {int(r['n_negative']):>5d}"
            )
        lines.append("")
    return lines


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    y_rank_for_auc: np.ndarray | None = None,
) -> dict:
    'Public-release English note.'
    metrics = {}
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    scores = (
        np.asarray(y_rank_for_auc, dtype=np.float64)
        if y_rank_for_auc is not None
        else y_prob
    )
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        metrics["roc_auc"] = None
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, scores))
    except ValueError:
        metrics["pr_auc"] = None
    # Public-release English note.
    try:
        if len(np.unique(y_true)) < 2:
            metrics["log_loss"] = None
        else:
            yp = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
            metrics["log_loss"] = float(log_loss(y_true, yp, labels=[0, 1]))
    except Exception:
        metrics["log_loss"] = None
    try:
        metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))
    except Exception:
        metrics["brier_score"] = None
    metrics["n_samples"] = int(len(y_true))
    metrics["n_positive"] = int(y_true.sum())
    metrics["n_negative"] = int(len(y_true) - y_true.sum())
    metrics["pos_rate"] = float(y_true.mean())
    return metrics


def compute_trajectory_level_savings(
    df_test: pd.DataFrame,
    y_prob: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict]:
    'Public-release English note.'
    thresholds = thresholds or DEFAULT_DECISION_THRESHOLDS

    df = df_test.copy()
    df["_y_prob"] = np.asarray(y_prob, dtype=np.float64)

    traj_groups = df.groupby("traj_id")

    results = []
    for thr in thresholds:
        total_trajs = 0
        decided_trajs = 0
        undecided_trajs = 0
        success_decided = 0
        failure_decided = 0
        correct_success = 0
        correct_failure = 0
        total_steps_all = 0
        total_saved_steps = 0
        saved_steps_list = []
        all_step_counts = []

        for traj_id, grp in traj_groups:
            grp_sorted = grp.sort_values("prefix_step_idx")
            n_total = int(grp_sorted["n_steps_total_for_weighting"].iloc[0])
            label = int(grp_sorted["label"].iloc[0])
            total_trajs += 1
            total_steps_all += n_total
            all_step_counts.append(n_total)

            decision_step = None
            decision_type = None

            for _, row in grp_sorted.iterrows():
                step_idx = int(row["prefix_step_idx"])
                prob = float(row["_y_prob"])

                if prob >= thr:
                    decision_step = step_idx
                    decision_type = "success"
                    break
                elif prob <= (1.0 - thr):
                    decision_step = step_idx
                    decision_type = "failure"
                    break

            if decision_step is not None:
                decided_trajs += 1
                saved = max(n_total - decision_step, 0)
                total_saved_steps += saved
                saved_steps_list.append(saved)

                if decision_type == "success":
                    success_decided += 1
                    if label == 1:
                        correct_success += 1
                else:
                    failure_decided += 1
                    if label == 0:
                        correct_failure += 1
            else:
                undecided_trajs += 1

        precision_success = correct_success / success_decided if success_decided > 0 else None
        precision_failure = correct_failure / failure_decided if failure_decided > 0 else None
        overall_correct = correct_success + correct_failure
        overall_decided = decided_trajs
        overall_accuracy = overall_correct / overall_decided if overall_decided > 0 else None

        avg_steps_all = np.mean(all_step_counts) if all_step_counts else 0.0
        median_steps_all = np.median(all_step_counts) if all_step_counts else 0.0
        avg_saved = np.mean(saved_steps_list) if saved_steps_list else 0.0
        median_saved = np.median(saved_steps_list) if saved_steps_list else 0.0
        total_possible_steps = total_steps_all
        saving_ratio = total_saved_steps / total_possible_steps if total_possible_steps > 0 else 0.0

        results.append({
            "threshold": float(thr),
            "total_trajs": total_trajs,
            "decided_trajs": decided_trajs,
            "undecided_trajs": undecided_trajs,
            "decided_ratio": decided_trajs / total_trajs if total_trajs > 0 else 0.0,
            "success_decided": success_decided,
            "failure_decided": failure_decided,
            "correct_success": correct_success,
            "correct_failure": correct_failure,
            "precision_success": precision_success,
            "precision_failure": precision_failure,
            "overall_accuracy": overall_accuracy,
            "total_steps_all_trajs": total_steps_all,
            "total_saved_steps": total_saved_steps,
            "saving_ratio": saving_ratio,
            "avg_saved_per_decided_traj": avg_saved,
            "median_saved_per_decided_traj": median_saved,
            "avg_saved_per_all_traj": total_saved_steps / total_trajs if total_trajs > 0 else 0.0,
            "avg_steps_all_trajs": avg_steps_all,
            "median_steps_all_trajs": median_steps_all,
        })
    return results


def compute_trajectory_savings_at_precision_levels(
    df_test: pd.DataFrame,
    y_prob: np.ndarray,
    precision_levels: list[float] | None = None,
    search_thresholds: np.ndarray | None = None,
) -> dict:
    'Public-release English note.'
    precision_levels = precision_levels or [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    if search_thresholds is None:
        lo = float(getattr(config, "PRECISION_SAVINGS_THR_GRID_START", 0.05))
        hi = float(getattr(config, "PRECISION_SAVINGS_THR_GRID_STOP", 1.0))
        stp = float(getattr(config, "PRECISION_SAVINGS_THR_GRID_STEP", 0.005))
        search_thresholds = np.arange(lo, hi, stp)

    # Public-release English note.
    traj_ids = df_test["traj_id"].to_numpy()
    steps = df_test["prefix_step_idx"].to_numpy(dtype=np.int64)
    labels = df_test["label"].to_numpy(dtype=np.int64)
    n_totals = df_test["n_steps_total_for_weighting"].to_numpy(dtype=np.int64)
    prob = np.asarray(y_prob, dtype=np.float64)
    order = np.lexsort((steps, traj_ids))
    tid_o = traj_ids[order]
    step_o = steps[order]
    prob_o = prob[order]
    ntot_o = n_totals[order]
    lab_o = labels[order]
    bounds = np.r_[np.flatnonzero(np.r_[True, tid_o[1:] != tid_o[:-1]]), len(order)]
    traj_data: list[dict] = []
    for bi in range(len(bounds) - 1):
        s, e = int(bounds[bi]), int(bounds[bi + 1])
        traj_data.append({
            "step": step_o[s:e],
            "prob": prob_o[s:e],
            "n_total": int(ntot_o[s]),
            "label": int(lab_o[s]),
        })

    anchor_fail = float(getattr(config, "PRECISION_SAVINGS_ANCHOR_P_FAIL", 0.25))
    anchor_succ = float(getattr(config, "PRECISION_SAVINGS_ANCHOR_THR_SUCCESS", 0.55))
    fp_lo = float(getattr(config, "PRECISION_SAVINGS_FAILURE_P_GRID_START", 0.02))
    fp_hi = float(getattr(config, "PRECISION_SAVINGS_FAILURE_P_GRID_STOP", 0.80))
    fp_st = float(getattr(config, "PRECISION_SAVINGS_FAILURE_P_GRID_STEP", 0.005))
    failure_p_grid = np.arange(fp_lo, fp_hi, fp_st)

    def _eval_asymmetric(thr_success: float, p_fail_max: float) -> dict:
        total_trajs = 0
        decided = 0
        succ_dec = 0
        fail_dec = 0
        corr_succ = 0
        corr_fail = 0
        total_steps = 0
        saved_steps = 0
        saved_list: list[int] = []
        saved_steps_succ_branch = 0
        saved_steps_fail_branch = 0
        saved_steps_on_success_label_trajs = 0
        saved_steps_on_failure_label_trajs = 0
        n_trajs_s = 0
        n_trajs_f = 0
        total_steps_label_success = 0
        total_steps_label_failure = 0

        for td in traj_data:
            n_total = td["n_total"]
            label = td["label"]
            total_trajs += 1
            total_steps += n_total
            if label == 1:
                n_trajs_s += 1
                total_steps_label_success += n_total
            else:
                n_trajs_f += 1
                total_steps_label_failure += n_total

            dec_step = None
            dec_type = None
            st = td["step"]
            pr = td["prob"]
            for i in range(pr.shape[0]):
                p = float(pr[i])
                if p >= thr_success:
                    dec_step = int(st[i])
                    dec_type = "success"
                    break
                if p <= p_fail_max:
                    dec_step = int(st[i])
                    dec_type = "failure"
                    break

            if dec_step is not None:
                decided += 1
                s = max(n_total - dec_step, 0)
                saved_steps += s
                saved_list.append(s)
                if label == 1:
                    saved_steps_on_success_label_trajs += s
                else:
                    saved_steps_on_failure_label_trajs += s
                if dec_type == "success":
                    succ_dec += 1
                    saved_steps_succ_branch += s
                    if label == 1:
                        corr_succ += 1
                else:
                    fail_dec += 1
                    saved_steps_fail_branch += s
                    if label == 0:
                        corr_fail += 1

        ps = corr_succ / succ_dec if succ_dec > 0 else None
        pf = corr_fail / fail_dec if fail_dec > 0 else None
        oa = (corr_succ + corr_fail) / decided if decided > 0 else None
        s_hit_rate = corr_succ / n_trajs_s if n_trajs_s > 0 else None
        f_hit_rate = corr_fail / n_trajs_f if n_trajs_f > 0 else None
        share_s = saved_steps_succ_branch / total_steps if total_steps > 0 else 0.0
        share_f = saved_steps_fail_branch / total_steps if total_steps > 0 else 0.0
        # Public-release English note.
        share_succ_vs_ts = (
            saved_steps_on_success_label_trajs / total_steps_label_success
            if total_steps_label_success > 0
            else 0.0
        )
        share_fail_vs_tf = (
            saved_steps_on_failure_label_trajs / total_steps_label_failure
            if total_steps_label_failure > 0
            else 0.0
        )
        undecided = total_trajs - decided
        traj_hit = (corr_succ + corr_fail) / total_trajs if total_trajs > 0 else 0.0
        return {
            "threshold_s": float(thr_success),
            "failure_p_max": float(p_fail_max),
            "threshold_f": float(p_fail_max),
            "total_trajs": total_trajs,
            "decided_trajs": decided,
            "undecided_trajs": int(undecided),
            "decided_ratio": decided / total_trajs if total_trajs > 0 else 0.0,
            "success_decided": succ_dec,
            "failure_decided": fail_dec,
            "corr_success_decisions": corr_succ,
            "corr_failure_decisions": corr_fail,
            "precision_success": ps,
            "precision_failure": pf,
            "overall_accuracy": oa,
            "trajectory_hit_rate": traj_hit,
            "count_trajs_label_success": n_trajs_s,
            "count_trajs_label_failure": n_trajs_f,
            "label_success_hit_rate": s_hit_rate,
            "label_failure_hit_rate": f_hit_rate,
            "saved_steps_via_success_branch": int(saved_steps_succ_branch),
            "saved_steps_via_failure_branch": int(saved_steps_fail_branch),
            "saved_steps_on_success_label_trajs": int(saved_steps_on_success_label_trajs),
            "saved_steps_on_failure_label_trajs": int(saved_steps_on_failure_label_trajs),
            "share_steps_saved_via_success_decisions": share_s,
            "share_steps_saved_via_failure_decisions": share_f,
            "total_steps_label_success": int(total_steps_label_success),
            "total_steps_label_failure": int(total_steps_label_failure),
            "share_succ_saved_vs_label_success_steps": share_succ_vs_ts,
            "share_fail_saved_vs_label_failure_steps": share_fail_vs_tf,
            "total_steps": total_steps,
            "total_saved_steps": saved_steps,
            "saving_ratio": saved_steps / total_steps if total_steps > 0 else 0.0,
            "avg_saved_per_decided": float(np.mean(saved_list)) if saved_list else 0.0,
            "median_saved_per_decided": float(np.median(saved_list)) if saved_list else 0.0,
            "avg_saved_per_all": saved_steps / total_trajs if total_trajs > 0 else 0.0,
        }

    def _eval_at_threshold_symmetric(thr: float) -> dict:
        'Public-release English note.'
        total_trajs = 0
        decided = 0
        succ_dec = 0
        fail_dec = 0
        corr_succ = 0
        corr_fail = 0
        total_steps = 0
        saved_steps = 0
        saved_list: list[int] = []
        low = 1.0 - thr

        saved_steps_succ_branch = 0
        saved_steps_fail_branch = 0
        n_trajs_s = 0
        n_trajs_f = 0

        for td in traj_data:
            n_total = td["n_total"]
            label = td["label"]
            total_trajs += 1
            total_steps += n_total
            if label == 1:
                n_trajs_s += 1
            else:
                n_trajs_f += 1

            dec_step = None
            dec_type = None
            st = td["step"]
            pr = td["prob"]
            for i in range(pr.shape[0]):
                p = float(pr[i])
                if p >= thr:
                    dec_step = int(st[i])
                    dec_type = "success"
                    break
                if p <= low:
                    dec_step = int(st[i])
                    dec_type = "failure"
                    break

            if dec_step is not None:
                decided += 1
                s = max(n_total - dec_step, 0)
                saved_steps += s
                saved_list.append(s)
                if dec_type == "success":
                    succ_dec += 1
                    saved_steps_succ_branch += s
                    if label == 1:
                        corr_succ += 1
                else:
                    fail_dec += 1
                    saved_steps_fail_branch += s
                    if label == 0:
                        corr_fail += 1

        ps = corr_succ / succ_dec if succ_dec > 0 else None
        pf = corr_fail / fail_dec if fail_dec > 0 else None
        oa = (corr_succ + corr_fail) / decided if decided > 0 else None
        s_hit_rate = corr_succ / n_trajs_s if n_trajs_s > 0 else None
        f_hit_rate = corr_fail / n_trajs_f if n_trajs_f > 0 else None
        share_s = saved_steps_succ_branch / total_steps if total_steps > 0 else 0.0
        share_f = saved_steps_fail_branch / total_steps if total_steps > 0 else 0.0
        return {
            "threshold": float(thr),
            "total_trajs": total_trajs,
            "decided_trajs": decided,
            "decided_ratio": decided / total_trajs if total_trajs > 0 else 0.0,
            "success_decided": succ_dec,
            "failure_decided": fail_dec,
            "corr_success_decisions": corr_succ,
            "corr_failure_decisions": corr_fail,
            "precision_success": ps,
            "precision_failure": pf,
            "overall_accuracy": oa,
            "count_trajs_label_success": n_trajs_s,
            "count_trajs_label_failure": n_trajs_f,
            "label_success_hit_rate": s_hit_rate,
            "label_failure_hit_rate": f_hit_rate,
            "saved_steps_via_success_branch": int(saved_steps_succ_branch),
            "saved_steps_via_failure_branch": int(saved_steps_fail_branch),
            "share_steps_saved_via_success_decisions": share_s,
            "share_steps_saved_via_failure_decisions": share_f,
            "total_steps": total_steps,
            "total_saved_steps": saved_steps,
            "saving_ratio": saved_steps / total_steps if total_steps > 0 else 0.0,
            "avg_saved_per_decided": float(np.mean(saved_list)) if saved_list else 0.0,
            "median_saved_per_decided": float(np.median(saved_list)) if saved_list else 0.0,
            "avg_saved_per_all": saved_steps / total_trajs if total_trajs > 0 else 0.0,
        }

    all_evals_s = [_eval_asymmetric(float(t), anchor_fail) for t in search_thresholds]

    def _best_asym_s_for_target(target_prec: float) -> dict | None:
        'Public-release English note.'
        feas = [
            dict(ev)
            for ev in all_evals_s
            if ev["precision_success"] is not None
            and ev["precision_success"] >= target_prec
            and ev["success_decided"] > 0
        ]
        if not feas:
            return None
        return max(
            feas,
            key=lambda e: (
                float(e["saving_ratio"]),
                float(e["share_steps_saved_via_success_decisions"]),
                float(e["threshold_s"]),
            ),
        )

    by_ps: list[dict] = []
    for target_prec in precision_levels:
        picked = _best_asym_s_for_target(target_prec)
        if picked is not None:
            picked = dict(picked)
            picked["target_precision"] = target_prec
            picked["constraint"] = "Prec(S)>="
            picked["threshold"] = picked["threshold_s"]
            picked["fixed_failure_p_max"] = anchor_fail
            # Public-release English note.
            picked["column_total_share"] = float(picked["saving_ratio"])
            by_ps.append(picked)
        else:
            by_ps.append({
                "target_precision": target_prec,
                "constraint": "Prec(S)>=",
                "threshold": None,
                "note": f"No threshold found with Prec(S)>={target_prec} (need success-side decisions)",
            })

    all_evals_f = [_eval_asymmetric(anchor_succ, float(pf)) for pf in failure_p_grid]

    def _best_asym_f_for_target(target_prec: float) -> dict | None:
        'Public-release English note.'
        feas = [
            dict(ev)
            for ev in all_evals_f
            if ev["precision_failure"] is not None
            and ev["precision_failure"] >= target_prec
            and ev["failure_decided"] > 0
        ]
        if not feas:
            return None
        return max(
            feas,
            key=lambda e: (
                float(e["saving_ratio"]),
                float(e["share_steps_saved_via_failure_decisions"]),
                float(e["failure_p_max"]),
            ),
        )

    by_pf: list[dict] = []
    for target_prec in precision_levels:
        picked = _best_asym_f_for_target(target_prec)
        if picked is not None:
            picked = dict(picked)
            picked["target_precision"] = target_prec
            picked["constraint"] = "Prec(F)>="
            picked["threshold"] = picked["failure_p_max"]
            picked["fixed_thr_success"] = anchor_succ
            picked["column_total_share"] = float(picked["saving_ratio"])
            by_pf.append(picked)
        else:
            by_pf.append({
                "target_precision": target_prec,
                "constraint": "Prec(F)>=",
                "threshold": None,
                "note": f"No threshold found with Prec(F)>={target_prec} (need failure-side decisions)",
            })

    all_evals_sym = [_eval_at_threshold_symmetric(float(t)) for t in search_thresholds]

    def _first_sym(pred) -> dict | None:
        for ev in all_evals_sym:
            if pred(ev):
                return dict(ev)
        return None

    # Public-release English note.
    by_pj = []
    for target_prec in precision_levels:
        picked = _first_sym(
            lambda ev, tp=target_prec: ev["precision_success"] is not None
            and ev["precision_failure"] is not None
            and ev["precision_success"] >= tp
            and ev["precision_failure"] >= tp
            and ev["success_decided"] > 0
            and ev["failure_decided"] > 0
        )
        if picked is not None:
            picked = dict(picked)
            picked["target_precision"] = target_prec
            picked["constraint"] = "Prec(S)>= and Prec(F)>="
            by_pj.append(picked)
        else:
            by_pj.append({
                "target_precision": target_prec,
                "constraint": "Prec(S)>= and Prec(F)>=",
                "threshold": None,
                "note": (
                    f"No thr with both Prec(S)>={target_prec} and Prec(F)>={target_prec} "
                    "(need both sides decided)"
                ),
            })

    # Public-release English note.
    all_pair_evals: list[dict] = []
    for ts in search_thresholds:
        for pf in failure_p_grid:
            all_pair_evals.append(_eval_asymmetric(float(ts), float(pf)))

    def _best_dual_joint_for_target(target_prec: float) -> dict | None:
        feas = [
            dict(ev)
            for ev in all_pair_evals
            if ev["precision_success"] is not None
            and ev["precision_failure"] is not None
            and float(ev["precision_success"]) >= target_prec
            and float(ev["precision_failure"]) >= target_prec
            and int(ev["success_decided"]) > 0
            and int(ev["failure_decided"]) > 0
        ]
        if not feas:
            return None
        return max(
            feas,
            key=lambda e: (
                float(e["saving_ratio"]),
                float(e["total_saved_steps"]),
                float(e["threshold_s"]),
                float(e["failure_p_max"]),
            ),
        )

    by_dual: list[dict] = []
    for target_prec in precision_levels:
        picked = _best_dual_joint_for_target(target_prec)
        if picked is not None:
            picked["target_precision"] = target_prec
            picked["constraint"] = "dual_asymmetric_joint_prec_s_and_prec_f"
            by_dual.append(picked)
        else:
            by_dual.append({
                "target_precision": target_prec,
                "constraint": "dual_asymmetric_joint_prec_s_and_prec_f",
                "threshold_s": None,
                "threshold_f": None,
                "failure_p_max": None,
                "note": (
                    'Public-release English note.'
                    'Public-release English note.'
                ),
            })

    n_trajs_s_global = sum(1 for td in traj_data if td["label"] == 1)
    n_trajs_f_global = len(traj_data) - n_trajs_s_global
    total_steps_global = int(sum(td["n_total"] for td in traj_data))
    total_steps_s_global = int(sum(td["n_total"] for td in traj_data if td["label"] == 1))
    total_steps_f_global = int(sum(td["n_total"] for td in traj_data if td["label"] != 1))

    return {
        "by_precision_success": by_ps,
        "by_precision_failure": by_pf,
        "by_precision_joint": by_pj,
        "by_precision_dual_thr": by_dual,
        "_anchors": {
            "p_fail_fixed_for_success_scan": anchor_fail,
            "thr_success_fixed_for_failure_scan": anchor_succ,
        },
        "_dataset": {
            "count_trajs_label_success": int(n_trajs_s_global),
            "count_trajs_label_failure": int(n_trajs_f_global),
            "total_steps": total_steps_global,
            "total_steps_label_success": total_steps_s_global,
            "total_steps_label_failure": total_steps_f_global,
            "total_trajs": len(traj_data),
        },
    }


def compute_bucketed_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    step_indices: np.ndarray,
    y_rank_for_auc: np.ndarray | None = None,
) -> list[dict]:
    'Public-release English note.'
    results = []
    for bucket_name, lo, hi in config.STEP_BUCKETS:
        mask = (step_indices >= lo) & (step_indices <= hi)
        if mask.sum() < 10:
            results.append({"bucket": bucket_name, "n_samples": int(mask.sum()), "skipped": True})
            continue
        y_b = y_true[mask]
        yr = y_rank_for_auc[mask] if y_rank_for_auc is not None else None
        m = compute_metrics(y_b, y_prob[mask], y_rank_for_auc=yr)
        m["bucket"] = bucket_name
        m["n_positive"] = int(np.sum(y_b))
        m["n_negative"] = int(len(y_b) - np.sum(y_b))
        results.append(m)
    return results


def compute_threshold_decision_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    step_indices: np.ndarray,
    n_steps_total: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict]:
    'Public-release English note.'
    thresholds = thresholds or DEFAULT_DECISION_THRESHOLDS
    results = []

    step_indices = np.asarray(step_indices)
    n_steps_total = np.asarray(n_steps_total)
    remaining_steps = np.maximum(n_steps_total - step_indices, 0).astype(np.float32)
    total_n = len(y_true)

    for thr in thresholds:
        success_mask = y_prob >= thr
        failure_mask = y_prob <= (1.0 - thr)
        decided_mask = success_mask | failure_mask
        undecided_mask = ~decided_mask

        tp = int(np.sum(success_mask & (y_true == 1)))
        fp = int(np.sum(success_mask & (y_true == 0)))
        tn = int(np.sum(failure_mask & (y_true == 0)))
        fn = int(np.sum(failure_mask & (y_true == 1)))

        n_decided = int(decided_mask.sum())
        n_success_side = int(success_mask.sum())
        n_failure_side = int(failure_mask.sum())
        n_undecided = int(undecided_mask.sum())

        precision_success = tp / (tp + fp) if (tp + fp) > 0 else None
        precision_failure = tn / (tn + fn) if (tn + fn) > 0 else None

        saved_steps_decided = remaining_steps[decided_mask]
        avg_saved_steps_decided = float(saved_steps_decided.mean()) if n_decided > 0 else 0.0
        total_saved_steps_decided = float(saved_steps_decided.sum()) if n_decided > 0 else 0.0
        avg_saved_steps_all = float(total_saved_steps_decided / total_n) if total_n > 0 else 0.0

        results.append(
            {
                "threshold": float(thr),
                "n_total": int(total_n),
                "n_decided": n_decided,
                "n_undecided": n_undecided,
                "decided_ratio": float(n_decided / total_n) if total_n > 0 else 0.0,
                "undecided_ratio": float(n_undecided / total_n) if total_n > 0 else 0.0,
                "n_success_side": n_success_side,
                "n_failure_side": n_failure_side,
                "success_side_ratio": float(n_success_side / total_n) if total_n > 0 else 0.0,
                "failure_side_ratio": float(n_failure_side / total_n) if total_n > 0 else 0.0,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision_success": precision_success,
                "precision_failure": precision_failure,
                "avg_saved_steps_decided": avg_saved_steps_decided,
                "total_saved_steps_decided": total_saved_steps_decided,
                "avg_saved_steps_all_samples": avg_saved_steps_all,
            }
        )
    return results


def compute_feature_group_contribution(
    model,
    feature_names: list[str],
    is_lgbm: bool = False,
) -> dict:
    'Public-release English note.'
    from feature_engineer import NUMERIC_FEATURES, BOOL_FEATURES

    GROUP_A_NUM = {
        "prefix_step_idx", "steps_observed_so_far", "actions_so_far",
        "observations_so_far", "tool_messages_so_far", "tool_calls_so_far",
        "distinct_tools_so_far", "prefix_action_chars", "prefix_feedback_chars",
        "task_prompt_chars",
    }
    GROUP_A_BOOL = {"has_any_action"}
    GROUP_B_NUM = {"last_step_tool_count", "last_step_action_chars", "last_step_feedback_chars"}
    GROUP_B_BOOL = {
        "last_step_has_tool_output", "last_step_has_observation",
        "last_step_tool_error_seen", "last_step_traceback_seen",
        "last_step_test_fail_seen", "last_step_test_pass_seen",
    }
    GROUP_C_NUM = {
        "read_view_so_far", "read_search_so_far", "edit_create_so_far",
        "edit_replace_so_far", "edit_insert_so_far", "edit_undo_so_far",
        "edits_so_far", "tests_so_far", "run_python_so_far", "run_cli_so_far",
        "git_ops_so_far", "cleanup_so_far", "submit_so_far",
        "bash_calls_so_far", "editor_calls_so_far",
    }
    GROUP_D_NUM = {
        "first_edit_step", "first_test_step", "first_run_python_step",
        "first_submit_step", "first_error_step", "first_traceback_step",
        "first_read_step",
    }
    GROUP_D_BOOL = {
        "first_edit_seen", "first_test_seen", "first_submit_seen",
        "first_error_seen", "first_traceback_seen",
    }
    GROUP_E_NUM = {
        "steps_since_last_edit", "steps_since_last_test", "steps_since_last_submit",
        "steps_since_last_error", "steps_since_last_traceback", "steps_since_last_read",
    }
    GROUP_F_NUM = {
        "read_to_edit_ratio", "edit_to_test_ratio", "bash_to_editor_ratio",
        "error_per_action_ratio", "submit_per_action_ratio",
        "feedback_chars_per_action", "action_chars_per_step", "distinct_tools_per_step",
    }
    GROUP_G_NUM = {"last_fail_count", "best_fail_count_so_far", "fail_count_delta_from_prev_test"}
    GROUP_G_BOOL = {
        "traceback_seen", "tool_error_seen", "assertion_error_seen",
        "type_error_seen", "value_error_seen", "syntax_error_seen",
        "import_error_seen", "file_not_found_seen", "timeout_seen",
        "permission_error_seen", "test_fail_seen", "test_pass_seen",
        "all_tests_passed_seen", "test_improving_seen",
    }
    GROUP_H_NUM = {"long_no_edit_streak", "long_read_streak"}
    GROUP_H_BOOL = {
        "repeated_same_action_consecutive", "repeated_same_search_consecutive",
        "repeated_same_view_consecutive", "looping_read_seen",
        "edit_failed_seen", "submit_without_test_seen", "premature_submit_seen",
        "multi_submit_seen", "submit_then_edit_again_seen", "test_after_submit_seen",
    }
    GROUP_J_NUM = {
        "thought_steps_so_far", "thought_density",
        "prefix_thought_chars", "avg_thought_chars_per_step", "last_thought_chars",
        "assistant_content_steps_so_far",
        "prefix_assistant_content_chars", "avg_assistant_content_chars_per_step",
        "last_assistant_content_chars",
        "thought_equals_content_rate",
        "thought_action_overlap_avg", "content_action_overlap_avg",
    }

    dense_groups = {
        "Dense_A (Progress)": GROUP_A_NUM | GROUP_A_BOOL,
        "Dense_B (LastStep)": GROUP_B_NUM | GROUP_B_BOOL,
        "Dense_C (ActionCount)": GROUP_C_NUM,
        "Dense_D (Milestone)": GROUP_D_NUM | GROUP_D_BOOL,
        "Dense_E (Recency)": GROUP_E_NUM,
        "Dense_F (Ratio)": GROUP_F_NUM,
        "Dense_G (Observation)": GROUP_G_NUM | GROUP_G_BOOL,
        "Dense_H (Looping)": GROUP_H_NUM | GROUP_H_BOOL,
        "Dense_J (Cognitive)": GROUP_J_NUM,
    }

    tfidf_groups = {
        "TfIdf_task_prompt": "tfidf_task_prompt",
        "TfIdf_action": {"tfidf_prefix_action", "tfidf_last_action"},
        "TfIdf_feedback": {"tfidf_prefix_feedback", "tfidf_last_feedback"},
        "TfIdf_thought": {"tfidf_prefix_thought", "tfidf_last_thought"},
        "TfIdf_assistant_content": {"tfidf_prefix_assistant_content", "tfidf_last_assistant_content"},
    }

    if is_lgbm:
        importance = model.feature_importance(importance_type="gain")
        names = model.feature_name()
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
        names = feature_names
    else:
        return {}

    name_imp = dict(zip(names, importance))

    group_results = {}
    total_importance = float(np.sum(importance))

    for group_name, group_features in dense_groups.items():
        group_imp = 0.0
        count = 0
        for fname in names:
            base_name = fname.split("__")[0] if "__" in fname and not fname.startswith("tfidf") else fname
            if base_name in group_features:
                group_imp += name_imp.get(fname, 0.0)
                count += 1
        group_results[group_name] = {
            "total_importance": float(group_imp),
            "share": float(group_imp / total_importance) if total_importance > 0 else 0.0,
            "n_features": count,
        }

    for group_name, prefixes in tfidf_groups.items():
        if isinstance(prefixes, str):
            prefixes = {prefixes}
        group_imp = 0.0
        count = 0
        for fname in names:
            for pfx in prefixes:
                if fname.startswith(pfx + "__"):
                    group_imp += name_imp.get(fname, 0.0)
                    count += 1
                    break
        group_results[group_name] = {
            "total_importance": float(group_imp),
            "share": float(group_imp / total_importance) if total_importance > 0 else 0.0,
            "n_features": count,
        }

    cat_group_imp = 0.0
    cat_count = 0
    for fname in names:
        for cat_col in ["last_step_action_major_type", "last_step_action_primary_subtype", "model_id"]:
            if fname.startswith(cat_col + "__"):
                cat_group_imp += name_imp.get(fname, 0.0)
                cat_count += 1
                break
    group_results["Dense_Cat (Categorical)"] = {
        "total_importance": float(cat_group_imp),
        "share": float(cat_group_imp / total_importance) if total_importance > 0 else 0.0,
        "n_features": cat_count,
    }

    return group_results


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    save_path: Path,
    y_logit: np.ndarray | None = None,
):
    'Public-release English note.'
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)

    if y_logit is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Public-release English note.
    ax = axes[0]
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=15, strategy="uniform")
    ax.plot(mean_pred, fraction_pos, "o-", label=model_name)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve - {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Public-release English note.
    ax = axes[1]
    ax.hist(y_prob[y_true == 0], bins=50, alpha=0.5, label="Negative (unresolved)", density=True)
    ax.hist(y_prob[y_true == 1], bins=50, alpha=0.5, label="Positive (resolved)", density=True)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Probability Distribution - {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if y_logit is not None:
        ax = axes[2]
        lo = np.asarray(y_logit, dtype=np.float64)
        m0 = (y_true == 0) & np.isfinite(lo)
        m1 = (y_true == 1) & np.isfinite(lo)
        ax.hist(lo[m0], bins=50, alpha=0.5, label="Negative (unresolved)", density=True)
        ax.hist(lo[m1], bins=50, alpha=0.5, label="Positive (resolved)", density=True)
        ax.set_xlabel("Log-odds (decision_function, pre-sigmoid)")
        ax.set_ylabel("Density")
        ax.set_title(f"Log-odds Distribution - {model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Calibration plot saved to {save_path}")


def plot_roc_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    save_path: Path,
    y_rank_for_auc: np.ndarray | None = None,
):
    'Public-release English note.'
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    scores = (
        np.asarray(y_rank_for_auc, dtype=np.float64)
        if y_rank_for_auc is not None
        else np.asarray(y_prob, dtype=np.float64)
    )

    # ROC
    ax = axes[0]
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc_val = roc_auc_score(y_true, scores)
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PR
    ax = axes[1]
    prec, rec, _ = precision_recall_curve(y_true, scores)
    pr_auc = average_precision_score(y_true, scores)
    ax.plot(rec, prec, label=f"{model_name} (AP={pr_auc:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC/PR plot saved to {save_path}")


def plot_metrics_by_step(
    bucketed_results: list[dict],
    model_name: str,
    save_path: Path,
):
    'Public-release English note.'
    valid = [r for r in bucketed_results if not r.get("skipped")]
    if not valid:
        return

    buckets = [r["bucket"] for r in valid]
    aucs = [
        float(r["roc_auc"]) if r.get("roc_auc") is not None else float("nan") for r in valid
    ]
    pr_aucs = [
        float(r["pr_auc"]) if r.get("pr_auc") is not None else float("nan") for r in valid
    ]
    briers = [r.get("brier_score", 0) for r in valid]
    n_samples = [r["n_samples"] for r in valid]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = range(len(buckets))

    axes[0, 0].bar(x, aucs, color="steelblue")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(buckets, rotation=15)
    axes[0, 0].set_title("ROC-AUC by Step Bucket")
    _aucs_fin = [v for v in aucs if np.isfinite(v)]
    if _aucs_fin:
        lo = max(0.0, min(_aucs_fin) - 0.05)
        hi = min(1.0, max(_aucs_fin) + 0.05)
        axes[0, 0].set_ylim(lo, hi)
    else:
        axes[0, 0].set_ylim(0.0, 1.0)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].bar(x, pr_aucs, color="darkorange")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(buckets, rotation=15)
    axes[0, 1].set_title("PR-AUC by Step Bucket")
    _pr_fin = [v for v in pr_aucs if np.isfinite(v)]
    if _pr_fin:
        lo = max(0.0, min(_pr_fin) - 0.05)
        hi = min(1.0, max(_pr_fin) + 0.05)
        axes[0, 1].set_ylim(lo, hi)
    else:
        axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(x, briers, color="forestgreen")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(buckets, rotation=15)
    axes[1, 0].set_title("Brier Score by Step Bucket")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].bar(x, n_samples, color="gray")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(buckets, rotation=15)
    axes[1, 1].set_title("Sample Count by Step Bucket")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Step-Bucketed Metrics - {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Step-bucketed metrics plot saved to {save_path}")


def plot_feature_importance_lr(
    model,
    feature_names: list[str],
    top_k: int = 40,
    save_path: Path | None = None,
):
    'Public-release English note.'
    coefs = model.coef_[0]
    abs_coefs = np.abs(coefs)
    top_idx = np.argsort(abs_coefs)[-top_k:][::-1]

    names = [feature_names[i] if i < len(feature_names) else f"f_{i}" for i in top_idx]
    vals = coefs[top_idx]

    fig, ax = plt.subplots(figsize=(10, max(8, top_k * 0.3)))
    colors = ["forestgreen" if v > 0 else "firebrick" for v in vals]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient")
    ax.set_title(f"Top {top_k} LR Feature Importances (Green=Positive, Red=Negative)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Feature importance plot saved to {save_path}")
    else:
        plt.close()

    return list(zip(names, vals.tolist()))


def plot_feature_importance_lgbm(
    booster,
    top_k: int = 40,
    save_path: Path | None = None,
):
    'Public-release English note.'
    importance = booster.feature_importance(importance_type="gain")
    names = booster.feature_name()
    top_idx = np.argsort(importance)[-top_k:][::-1]

    top_names = [names[i] for i in top_idx]
    top_vals = importance[top_idx]

    fig, ax = plt.subplots(figsize=(10, max(8, top_k * 0.3)))
    ax.barh(range(len(top_names)), top_vals, color="steelblue")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Gain")
    ax.set_title(f"Top {top_k} LightGBM Feature Importances (Gain)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"LightGBM feature importance plot saved to {save_path}")
    else:
        plt.close()

    return list(zip(top_names, top_vals.tolist()))


def plot_precision_constraint_savings_curves(
    model_name: str,
    prec_bundle: dict,
    save_path: Path,
) -> bool:
    'Public-release English note.'
    if isinstance(prec_bundle, list) or not prec_bundle:
        return False
    rows_s = prec_bundle.get("by_precision_success") or []
    rows_f = prec_bundle.get("by_precision_failure") or []
    rows_d = prec_bundle.get("by_precision_dual_thr") or []

    def _extract(rows: list) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for r in sorted(rows, key=lambda x: float(x.get("target_precision", 0))):
            if r.get("threshold") is None:
                continue
            xs.append(float(r["target_precision"]))
            yv = r.get("column_total_share")
            if yv is None:
                yv = r.get("saving_ratio")
            ys.append(float(yv) * 100.0)
        return xs, ys

    def _extract_dual(rows: list) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for r in sorted(rows, key=lambda x: float(x.get("target_precision", 0))):
            if r.get("threshold_s") is None:
                continue
            if r.get("failure_p_max") is None and r.get("threshold_f") is None:
                continue
            if r.get("saving_ratio") is None:
                continue
            xs.append(float(r["target_precision"]))
            ys.append(float(r["saving_ratio"]) * 100.0)
        return xs, ys

    x_s, y_s = _extract(rows_s)
    x_f, y_f = _extract(rows_f)
    x_d, y_d = _extract_dual(rows_d)
    if not x_s and not x_f and not x_d:
        return False

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    if x_s:
        ax.plot(
            x_s,
            y_s,
            "o-",
            label='Public-release English note.',
            markersize=6,
            color="#1f77b4",
        )
    if x_f:
        ax.plot(
            x_f,
            y_f,
            "s-",
            label='Public-release English note.',
            markersize=6,
            color="#ff7f0e",
        )
    if x_d:
        ax.plot(
            x_d,
            y_d,
            "^-",
            label='Public-release English note.',
            markersize=6,
            color="#2ca02c",
        )
    ax.set_xlabel('Public-release English note.')
    ax.set_ylabel('Public-release English note.')
    ax.set_title('Public-release English note.')
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ticks = sorted(set(x_s + x_f + x_d))
    if ticks:
        ax.set_xticks(ticks)
    all_y = y_s + y_f + y_d
    if all_y:
        ax.set_ylim(0, min(100.0, max(all_y) * 1.08 + 3))
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Precision-savings curve saved to {save_path}")
    return True


def generate_full_report(
    all_results: dict,
    save_dir: Path,
    report_metadata: dict | None = None,
):
    'Public-release English note.'
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("SWE-smith Prefix Success Prediction - Evaluation Report")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("## Metrics & Threshold Guide")
    report_lines.append("")
    report_lines.append("- ROC-AUC / PR-AUC: higher is better")
    report_lines.append("- LogLoss / Brier: lower is better")
    report_lines.append("- Prec(S): precision on success-side decisions (p >= threshold), higher is better")
    report_lines.append("- Prec(F): precision on failure-side decisions (p <= 1-threshold), higher is better")
    report_lines.append("- Decide%: decision coverage under current threshold")
    report_lines.append("- AvgSave(dec): average saved steps on decided samples")
    report_lines.append("- AvgSave(all): average saved steps over all samples (global efficiency gain)")
    report_lines.append("")
    report_lines.append("Threshold trade-off:")
    report_lines.append("- Higher threshold usually improves precision but reduces coverage")
    report_lines.append("- Lower threshold usually increases coverage but may hurt precision")
    report_lines.append("")
    report_lines.append("Trajectory-level savings:")
    report_lines.append("- SavedSteps: for each trajectory, scan from step 0; when prediction crosses threshold, remaining steps are saved")
    report_lines.append("- Prec(S)/Prec(F): precision of success/failure decisions at trajectory level")
    report_lines.append("- Overall Acc: accuracy of all decided trajectories")
    report_lines.append("- SavingRatio: total saved steps / total possible steps across all trajectories")
    report_lines.append("")
    report_lines.append("Precision-constrained thresholds (two independent tables):")
    report_lines.append("- Table A: smallest thr such that Prec(S) >= target (optimize success-side quality first)")
    report_lines.append("- Table B: smallest thr such that Prec(F) >= target (optimize failure-side quality first)")
    report_lines.append("- At that thr, both Prec(S) and Prec(F) are reported; they need not both hit the same target.")
    report_lines.append("")
    report_lines.append("Process-only interpretation:")
    report_lines.append("- If process-only remains strong, current state dynamics are highly predictive")
    report_lines.append("- If it drops heavily, model relied more on task prior difficulty/model identity")
    report_lines.append("")

    # Public-release English note.
    report_lines.append("## Overall Metrics Summary")
    report_lines.append("")
    header = f"{'Model':<35} {'ROC-AUC':>8} {'PR-AUC':>8} {'LogLoss':>8} {'Brier':>8} {'N':>8}"
    report_lines.append(header)
    report_lines.append("-" * len(header))

    def _fmt_opt(v, w=8):
        if v is None:
            return f"{'N/A':>{w}}"
        try:
            if isinstance(v, float) and np.isnan(v):
                return f"{'N/A':>{w}}"
        except Exception:
            pass
        return f"{float(v):>{w}.4f}"

    for model_name, result in all_results.items():
        m = result.get("metrics", {})
        line = (
            f"{model_name:<35} "
            f"{_fmt_opt(m.get('roc_auc'))} "
            f"{_fmt_opt(m.get('pr_auc'))} "
            f"{_fmt_opt(m.get('log_loss'))} "
            f"{_fmt_opt(m.get('brier_score'))} "
            f"{m.get('n_samples', 0):>8d}"
        )
        report_lines.append(line)

    report_lines.append("")
    report_lines.append("")

    # Public-release English note.
    if report_metadata:
        sr = report_metadata.get("stratified_repo_section_lines")
        if sr:
            report_lines.extend(sr)
            report_lines.append("")

    # Public-release English note.
    def _safe_auc(model_key: str):
        m = all_results.get(model_key, {}).get("metrics", {})
        v = m.get("roc_auc")
        return float(v) if v is not None else None

    auc_with_model = _safe_auc("C_Dense_AF_Thought_LR")
    auc_no_task = _safe_auc("Abl_NoTaskPrompt_LR")
    auc_no_model = _safe_auc("Abl_NoModel_LR")
    auc_process_only = _safe_auc("Abl_ProcessOnly_LR")
    auc_j_lgbm = _safe_auc("J_LightGBM_Dense_AF_Thought")
    auc_no_task_lgbm = _safe_auc("Abl_NoTaskPrompt_LightGBM")
    auc_no_model_lgbm = _safe_auc("Abl_NoModel_LightGBM")
    auc_process_lgbm = _safe_auc("Abl_ProcessOnly_LightGBM")

    report_lines.append("## Key Counterfactual Experiments")
    report_lines.append("")
    report_lines.append(f"- with_model_id + with_task_prompt (C LR): {auc_with_model}")
    report_lines.append(f"- without_task_prompt (Abl_NoTaskPrompt_LR): {auc_no_task}")
    report_lines.append(f"- without_model_id (Abl_NoModel_LR): {auc_no_model}")
    report_lines.append(f"- process_only (no task prompt + no model_id): {auc_process_only}")
    report_lines.append(f"- same backbone LightGBM (J, Dense+AF+Thought): {auc_j_lgbm}")
    report_lines.append(f"- Abl_NoTaskPrompt_LightGBM: {auc_no_task_lgbm}")
    report_lines.append(f"- Abl_NoModel_LightGBM: {auc_no_model_lgbm}")
    report_lines.append(f"- Abl_ProcessOnly_LightGBM: {auc_process_lgbm}")
    if auc_with_model is not None and auc_no_task is not None:
        report_lines.append(f"- delta(remove task prompt) = {auc_no_task - auc_with_model:+.4f} (AUC, LR)")
    if auc_with_model is not None and auc_no_model is not None:
        report_lines.append(f"- delta(remove model_id) = {auc_no_model - auc_with_model:+.4f} (AUC, LR)")
    if auc_with_model is not None and auc_process_only is not None:
        report_lines.append(f"- delta(process_only vs C) = {auc_process_only - auc_with_model:+.4f} (AUC, LR)")
    if auc_j_lgbm is not None and auc_no_task_lgbm is not None:
        report_lines.append(f"- delta(remove task prompt, LGBM) = {auc_no_task_lgbm - auc_j_lgbm:+.4f} (AUC)")
    report_lines.append("")

    # Public-release English note.
    abl_lgbm_pairs = [
        ("Abl_DenseOnly_LR", "Abl_DenseOnly_LightGBM"),
        ("Abl_NoThoughtContent_LR", "Abl_NoThoughtContent_LightGBM"),
        ("Abl_NoAssistantContent_LR", "Abl_NoAssistantContent_LightGBM"),
        ("Abl_Base_LR", "Abl_Base_LightGBM"),
        ("Abl_NoTaskPrompt_LR", "Abl_NoTaskPrompt_LightGBM"),
        ("Abl_NoFeedback_LR", "Abl_NoFeedback_LightGBM"),
        ("Abl_NoAction_LR", "Abl_NoAction_LightGBM"),
        ("Abl_NoThought_LR", "Abl_NoThought_LightGBM"),
        ("Abl_NoModel_LR", "Abl_NoModel_LightGBM"),
        ("Abl_ProcessOnly_LR", "Abl_ProcessOnly_LightGBM"),
    ]
    report_lines.append("## Ablation: Logistic Regression vs LightGBM (ROC-AUC)")
    report_lines.append("")
    hdr = f"{'Ablation (same features)':<42} {'LR AUC':>10} {'LGBM AUC':>10} {'Δ(LGBM-LR)':>12}"
    report_lines.append(hdr)
    report_lines.append("-" * len(hdr))
    for lr_k, lgb_k in abl_lgbm_pairs:
        a_lr = _safe_auc(lr_k)
        a_lgb = _safe_auc(lgb_k)
        lr_s = f"{a_lr:.4f}" if a_lr is not None else "-"
        lgb_s = f"{a_lgb:.4f}" if a_lgb is not None else "-"
        if a_lr is not None and a_lgb is not None:
            d_s = f"{a_lgb - a_lr:+.4f}"
        else:
            d_s = "-"
        short = lr_k.replace("_LR", "")
        report_lines.append(f"{short:<42} {lr_s:>10} {lgb_s:>10} {d_s:>12}")
    report_lines.append("")

    # Public-release English note.
    abl_pairs = [
        ("Abl_DenseOnly_LR", "Abl_DenseOnly_LightGBM"),
        ("Abl_NoThoughtContent_LR", "Abl_NoThoughtContent_LightGBM"),
        ("Abl_NoAssistantContent_LR", "Abl_NoAssistantContent_LightGBM"),
        ("Abl_Base_LR", "Abl_Base_LightGBM"),
        ("Abl_NoTaskPrompt_LR", "Abl_NoTaskPrompt_LightGBM"),
        ("Abl_NoFeedback_LR", "Abl_NoFeedback_LightGBM"),
        ("Abl_NoAction_LR", "Abl_NoAction_LightGBM"),
        ("Abl_NoThought_LR", "Abl_NoThought_LightGBM"),
        ("Abl_NoModel_LR", "Abl_NoModel_LightGBM"),
        ("Abl_ProcessOnly_LR", "Abl_ProcessOnly_LightGBM"),
    ]
    any_lgbm_abl = any(
        all_results.get(lr, {}).get("metrics", {}).get("roc_auc") is not None
        and all_results.get(lgb, {}).get("metrics", {}).get("roc_auc") is not None
        for lr, lgb in abl_pairs
    )
    if any_lgbm_abl:
        report_lines.append("## Ablation LR vs LightGBM (same feature matrix, ROC-AUC)")
        report_lines.append("")
        report_lines.append(f"{'Ablation':<42} {'LR':>10} {'LightGBM':>10} {'LGB-LR':>10}")
        report_lines.append("-" * 76)
        for lr_k, lgb_k in abl_pairs:
            a_lr = _safe_auc(lr_k)
            a_lgb = _safe_auc(lgb_k)
            if a_lr is None and a_lgb is None:
                continue
            d = (a_lgb - a_lr) if (a_lr is not None and a_lgb is not None) else None
            ds = f"{d:+.4f}" if d is not None else "-"
            lr_s = f"{a_lr:.4f}" if a_lr is not None else "-"
            lgb_s = f"{a_lgb:.4f}" if a_lgb is not None else "-"
            short = lr_k.replace("_LR", "")
            report_lines.append(f"{short:<42} {lr_s:>10} {lgb_s:>10} {ds:>10}")
        report_lines.append("")

    # Public-release English note.
    for model_name, result in all_results.items():
        bucketed = result.get("bucketed", [])
        if not bucketed:
            continue
        report_lines.append(f"## Step-Bucketed Metrics - {model_name}")
        report_lines.append("")
        report_lines.append(
            'Public-release English note.'
        )
        header = (
            f"{'Bucket':<15} {'ROC-AUC':>8} {'PR-AUC':>8} {'LogLoss':>8} {'Brier':>8} "
            f"{'N':>6} {'+':>5} {'-':>5}"
        )
        report_lines.append(header)
        report_lines.append("-" * len(header))
        for r in bucketed:
            if r.get("skipped"):
                report_lines.append(
                    f"{r['bucket']:<15}{'SKIPPED (n<10)':<47}{r['n_samples']:>8d}"
                )
            else:
                ra = r.get("roc_auc")
                pa = r.get("pr_auc")
                ra_s = f"{ra:>8.4f}" if ra is not None else f"{'N/A':>8}"
                pa_s = f"{pa:>8.4f}" if pa is not None else f"{'N/A':>8}"
                np_ = r.get("n_positive")
                nn_ = r.get("n_negative")
                ps = f"{np_:>5d}" if np_ is not None else f"{'?':>5}"
                ns = f"{nn_:>5d}" if nn_ is not None else f"{'?':>5}"
                ll = r.get("log_loss")
                br = r.get("brier_score")
                ll_s = f"{float(ll):>8.4f}" if ll is not None and pd.notna(ll) else f"{'N/A':>8}"
                br_s = f"{float(br):>8.4f}" if br is not None and pd.notna(br) else f"{'N/A':>8}"
                line = (
                    f"{r['bucket']:<15} "
                    f"{ra_s} "
                    f"{pa_s} "
                    f"{ll_s} "
                    f"{br_s} "
                    f"{r['n_samples']:>6d} "
                    f"{ps} "
                    f"{ns}"
                )
                report_lines.append(line)
        report_lines.append("")

    # Public-release English note.
    checks = (report_metadata or {}).get("implementation_checks", {})
    if checks:
        report_lines.append("## Mixed-Model Implementation Checks")
        report_lines.append("")
        report_lines.append(f"- dense standardization applied: {checks.get('dense_standardization_applied')}")
        report_lines.append(f"- Dense+AF shape: {checks.get('dense_af_train_shape')}")
        report_lines.append(f"- Dense+AF+Thought shape: {checks.get('dense_af_thought_train_shape')}")
        report_lines.append(f"- Dense+Full shape: {checks.get('dense_full_train_shape')}")
        report_lines.append(f"- Dense+AF column match: {checks.get('dense_af_tfidf_match')}")
        report_lines.append(f"- Dense+AF+Thought column match: {checks.get('dense_af_thought_tfidf_match')}")
        report_lines.append(f"- Dense+Full column match: {checks.get('dense_full_tfidf_match')}")
        report_lines.append(f"- Dense+AF nnz: {checks.get('dense_af_nnz')}")
        report_lines.append(f"- Dense+AF+Thought nnz: {checks.get('dense_af_thought_nnz')}")
        report_lines.append(f"- Dense+Full nnz: {checks.get('dense_full_nnz')}")
        if "C_Dense_AF_Thought_LR_n_iter_max" in checks:
            report_lines.append(f"- C_Dense_AF_Thought_LR n_iter_max: {checks.get('C_Dense_AF_Thought_LR_n_iter_max')}")
        if "D_Dense_Full_LR_n_iter_max" in checks:
            report_lines.append(f"- D_Dense_Full_LR n_iter_max: {checks.get('D_Dense_Full_LR_n_iter_max')}")
        report_lines.append("")

    # Public-release English note.
    for model_name, result in all_results.items():
        traj_rows = result.get("trajectory_savings", [])
        if not traj_rows:
            continue
        report_lines.append(f"## Trajectory-Level Decision & Savings - {model_name}")
        report_lines.append("")
        if traj_rows:
            r0 = traj_rows[0]
            report_lines.append(f"  Total trajectories: {r0['total_trajs']}")
            report_lines.append(f"  Average steps per trajectory: {r0['avg_steps_all_trajs']:.2f}")
            report_lines.append(f"  Median steps per trajectory: {r0['median_steps_all_trajs']:.1f}")
        report_lines.append("")
        header = (
            f"{'Thr':>5} {'Decide%':>8} {'Succ#':>6} {'Fail#':>6} "
            f"{'Prec(S)':>8} {'Prec(F)':>8} {'OvrlAcc':>8} "
            f"{'AvgSave':>8} {'MedSave':>8} {'SaveRatio':>10} {'TotalSaved':>11}"
        )
        report_lines.append(header)
        report_lines.append("-" * len(header))
        for r in traj_rows:
            ps = r.get("precision_success")
            pf = r.get("precision_failure")
            oa = r.get("overall_accuracy")
            line = (
                f"{r['threshold']:>5.2f} "
                f"{r['decided_ratio'] * 100:>7.2f}% "
                f"{r['success_decided']:>6d} "
                f"{r['failure_decided']:>6d} "
                f"{(ps if ps is not None else float('nan')):>8.4f} "
                f"{(pf if pf is not None else float('nan')):>8.4f} "
                f"{(oa if oa is not None else float('nan')):>8.4f} "
                f"{r['avg_saved_per_decided_traj']:>8.2f} "
                f"{r['median_saved_per_decided_traj']:>8.1f} "
                f"{r['saving_ratio'] * 100:>9.2f}% "
                f"{r['total_saved_steps']:>11d}"
            )
            report_lines.append(line)
        report_lines.append("")

    def _append_precision_constraint_table(title: str, subtitle: str, prec_rows: list):
        report_lines.append(title)
        report_lines.append("")
        report_lines.append(f"  {subtitle}")
        report_lines.append("")
        header = (
            f"{'Target':>8} {'Thr':>6} {'Decide%':>8} {'Succ#':>6} {'Fail#':>6} "
            f"{'Prec(S)':>8} {'Prec(F)':>8} {'OvrlAcc':>8} "
            f"{'AvgSave':>8} {'MedSave':>8} {'SaveRatio':>10}"
        )
        report_lines.append(header)
        report_lines.append("-" * len(header))
        for r in prec_rows:
            if r.get("threshold") is None:
                note = r.get("note", "N/A")
                report_lines.append(f"{r['target_precision']:>8.2f}   {note}")
                continue
            ps = r.get("precision_success")
            pf = r.get("precision_failure")
            oa = r.get("overall_accuracy")
            line = (
                f"{r['target_precision']:>8.2f} "
                f"{r['threshold']:>6.3f} "
                f"{r['decided_ratio'] * 100:>7.2f}% "
                f"{r['success_decided']:>6d} "
                f"{r['failure_decided']:>6d} "
                f"{(ps if ps is not None else float('nan')):>8.4f} "
                f"{(pf if pf is not None else float('nan')):>8.4f} "
                f"{(oa if oa is not None else float('nan')):>8.4f} "
                f"{r['avg_saved_per_decided']:>8.2f} "
                f"{r['median_saved_per_decided']:>8.1f} "
                f"{r['saving_ratio'] * 100:>9.2f}%"
            )
            report_lines.append(line)
        report_lines.append("")

    def _append_merged_precision_levels_table(
        model_name: str, rows_s: list, rows_f: list, prec_bundle: dict | None = None
    ):
        'Public-release English note.'
        report_lines.append(f"## Savings at Precision Levels - {model_name}")
        report_lines.append("")
        report_lines.append(
            'Public-release English note.'
        )
        report_lines.append(
            'Public-release English note.'
            'Public-release English note.'
            'Public-release English note.'
        )
        report_lines.append(
            'Public-release English note.'
            'Public-release English note.'
        )
        report_lines.append(
            'Public-release English note.'
            'Public-release English note.'
            'Public-release English note.'
        )
        anc = (prec_bundle or {}).get("_anchors") or {}
        pfa = anc.get("p_fail_fixed_for_success_scan")
        tsa = anc.get("thr_success_fixed_for_failure_scan")
        if pfa is not None and tsa is not None:
            report_lines.append(
                'Public-release English note.'
            )
        report_lines.append(
            'Public-release English note.'
        )
        report_lines.append("")

        ds = (prec_bundle or {}).get("_dataset") or {}
        by_t_s = {r["target_precision"]: r for r in rows_s}
        by_t_f = {r["target_precision"]: r for r in rows_f}
        rows_dual = (prec_bundle or {}).get("by_precision_dual_thr") or []
        by_t_dual = {r["target_precision"]: r for r in rows_dual} if rows_dual else {}
        all_targets = sorted(set(by_t_s.keys()) | set(by_t_f.keys()))

        T_global = ds.get("total_steps")
        if T_global is None or int(T_global) <= 0:
            _ref = None
            for _rows in (rows_s, rows_f):
                for _r in _rows:
                    if _r.get("total_steps"):
                        _ref = _r
                        break
                if _ref:
                    break
            T_global = int(_ref["total_steps"]) if _ref else 0

        def _pct_saved(saved: int, ttot: int) -> tuple[str, str]:
            if ttot and ttot > 0:
                return str(int(saved)), f"{100.0 * float(saved) / float(ttot):.2f}%"
            return str(int(saved)), "-"

        for t in all_targets:
            rs = by_t_s.get(t)
            rf = by_t_f.get(t)
            rd = by_t_dual.get(t)

            if rs and rs.get("threshold") is not None:
                ts_v = float(rs.get("threshold_s", rs["threshold"]))
                ps = rs.get("precision_success")
                ps_txt = f"{float(ps) * 100:.2f}%" if ps is not None else "-"
                t_r = int(rs.get("total_steps") or T_global or 0)
                s_br = int(rs.get("saved_steps_via_success_branch", 0))
                s_n, s_pct = _pct_saved(s_br, t_r)
                cs = rs.get("corr_success_decisions")
                ns = rs.get("count_trajs_label_success")
                if ns is None:
                    ns = ds.get("count_trajs_label_success")
                ns_i = int(ns) if ns is not None else 0
                cs_i = int(cs) if cs is not None else 0
                cov_s = f"{100.0 * cs_i / ns_i:.1f}%" if ns_i > 0 else "-"
                pos_line = (
                    f"Thr_S={ts_v:.4f}  Prec(S)={ps_txt}  "
                    'Public-release English note.'
                    'Public-release English note.'
                )
            else:
                _n = (rs.get("note") or "").replace("\n", " ").strip() if rs else ""
                pos_line = f"-{'(' + _n + ')' if _n else ''}"

            if rf and rf.get("threshold") is not None:
                pf_v = float(rf.get("failure_p_max", rf.get("threshold_f", rf["threshold"])))
                pff = rf.get("precision_failure")
                pf_txt = f"{float(pff) * 100:.2f}%" if pff is not None else "-"
                t_f = int(rf.get("total_steps") or T_global or 0)
                f_br = int(rf.get("saved_steps_via_failure_branch", 0))
                f_n, f_pct = _pct_saved(f_br, t_f)
                cf = rf.get("corr_failure_decisions")
                nf_ = rf.get("count_trajs_label_failure")
                if nf_ is None:
                    nf_ = ds.get("count_trajs_label_failure")
                nf_i = int(nf_) if nf_ is not None else 0
                cf_i = int(cf) if cf is not None else 0
                cov_f = f"{100.0 * cf_i / nf_i:.1f}%" if nf_i > 0 else "-"
                neg_line = (
                    f"P_fail={pf_v:.4f}  Prec(F)={pf_txt}  "
                    'Public-release English note.'
                    'Public-release English note.'
                )
            else:
                _n = (rf.get("note") or "").replace("\n", " ").strip() if rf else ""
                neg_line = f"-{'(' + _n + ')' if _n else ''}"

            pf_dual = rd.get("failure_p_max") if rd else None
            if pf_dual is None and rd:
                pf_dual = rd.get("threshold_f")
            if (
                rd
                and rd.get("saving_ratio") is not None
                and rd.get("threshold_s") is not None
                and pf_dual is not None
            ):
                ts_j = float(rd["threshold_s"])
                pf_j = float(pf_dual)
                ps_j = rd.get("precision_success")
                pf_jj = rd.get("precision_failure")
                pss = f"{float(ps_j) * 100:.2f}%" if ps_j is not None else "-"
                pfs = f"{float(pf_jj) * 100:.2f}%" if pf_jj is not None else "-"
                t_j = int(rd.get("total_steps") or T_global or 0)
                sj_ts = int(rd.get("saved_steps_on_success_label_trajs", 0))
                sj_tf = int(rd.get("saved_steps_on_failure_label_trajs", 0))
                tot_s = int(rd.get("total_saved_steps", sj_ts + sj_tf))
                dr = rd.get("decided_ratio")
                dr_txt = f"{float(dr) * 100:.1f}%" if dr is not None else "-"
                dec_j = int(rd.get("decided_trajs", 0))
                nt_j = int(rd.get("total_trajs") or ds.get("total_trajs") or 0)
                decided_txt = 'Public-release English note.' if nt_j > 0 else dr_txt
                a_n, a_pct = _pct_saved(sj_ts, t_j)
                b_n, b_pct = _pct_saved(sj_tf, t_j)
                c_n, c_pct = _pct_saved(tot_s, t_j)
                ht = rd.get("trajectory_hit_rate")
                hit_txt = f"{float(ht) * 100:.2f}%" if ht is not None else "-"
                joint_line = (
                    'Public-release English note.'
                    'Public-release English note.'
                )
            else:
                joint_line = 'Public-release English note.'

            report_lines.append(f"  ── Target = {t:.2f} ─────────────────────────────────────────────")
            report_lines.append('Public-release English note.')
            report_lines.append('Public-release English note.')
            report_lines.append('Public-release English note.')
            report_lines.append("")

        ref = None
        for rows in (rows_s, rows_f):
            for r in rows:
                if r.get("threshold") is not None:
                    ref = r
                    break
            if ref:
                break
        ns = ds.get("count_trajs_label_success")
        nf = ds.get("count_trajs_label_failure")
        ts = ds.get("total_steps")
        tss = ds.get("total_steps_label_success")
        tsf = ds.get("total_steps_label_failure")
        if ns is None and ref is not None:
            ns = ref.get("count_trajs_label_success")
        if nf is None and ref is not None:
            nf = ref.get("count_trajs_label_failure")
        if ts is None and ref is not None:
            ts = ref.get("total_steps")
        if tss is None and ref is not None:
            tss = ref.get("total_steps_label_success")
        if tsf is None and ref is not None:
            tsf = ref.get("total_steps_label_failure")
        ntot = ds.get("total_trajs")
        if ntot is None and ref is not None:
            ntot = ref.get("total_trajs")
        report_lines.append(
            'Public-release English note.'
            'Public-release English note.'
            f"T={ts if ts is not None else '-'}  Ts={tss if tss is not None else '-'}  "
            'Public-release English note.'
            'Public-release English note.'
        )
        report_lines.append("")

    # Public-release English note.
    for model_name, result in all_results.items():
        prec_bundle = result.get("precision_level_savings") or {}
        if isinstance(prec_bundle, list):
            # Public-release English note.
            if not prec_bundle:
                continue
            _append_precision_constraint_table(
                f"## Savings at Precision Levels (legacy single-threshold) - {model_name}",
                'Public-release English note.',
                prec_bundle,
            )
            continue
        rows_s = prec_bundle.get("by_precision_success") or []
        rows_f = prec_bundle.get("by_precision_failure") or []
        if not rows_s and not rows_f:
            continue
        _append_merged_precision_levels_table(model_name, rows_s, rows_f, prec_bundle)
        fig_path = save_dir / f"precision_savings_curve_{model_name}.png"
        if plot_precision_constraint_savings_curves(model_name, prec_bundle, fig_path):
            report_lines.append('Public-release English note.')
            report_lines.append("")

    # Public-release English note.
    for model_name, result in all_results.items():
        rows = result.get("threshold_table", [])
        if not rows:
            continue
        report_lines.append(f"## [Legacy] Sample-Level Threshold Decision - {model_name}")
        report_lines.append("")
        header = (
            f"{'Thr':>5} {'Decide%':>8} {'Succ%':>8} {'Fail%':>8} "
            f"{'Prec(S)':>8} {'Prec(F)':>8} {'AvgSave(dec)':>13} {'AvgSave(all)':>12}"
        )
        report_lines.append(header)
        report_lines.append("-" * len(header))
        for r in rows:
            ps = r.get("precision_success")
            pf = r.get("precision_failure")
            line = (
                f"{r['threshold']:>5.2f} "
                f"{r['decided_ratio'] * 100:>7.2f}% "
                f"{r['success_side_ratio'] * 100:>7.2f}% "
                f"{r['failure_side_ratio'] * 100:>7.2f}% "
                f"{(ps if ps is not None else float('nan')):>8.4f} "
                f"{(pf if pf is not None else float('nan')):>8.4f} "
                f"{r['avg_saved_steps_decided']:>13.3f} "
                f"{r['avg_saved_steps_all_samples']:>12.3f}"
            )
            report_lines.append(line)
        report_lines.append("")

    # Public-release English note.
    for model_name, result in all_results.items():
        fg = result.get("feature_group_contribution", {})
        if not fg:
            continue
        report_lines.append(f"## Feature Group Contribution - {model_name}")
        report_lines.append("")
        header = f"{'Group':<35} {'TotalImp':>10} {'Share%':>8} {'#Feats':>7}"
        report_lines.append(header)
        report_lines.append("-" * len(header))
        sorted_groups = sorted(fg.items(), key=lambda x: x[1]["share"], reverse=True)
        for gname, ginfo in sorted_groups:
            line = (
                f"{gname:<35} "
                f"{ginfo['total_importance']:>10.4f} "
                f"{ginfo['share'] * 100:>7.2f}% "
                f"{ginfo['n_features']:>7d}"
            )
            report_lines.append(line)
        report_lines.append("")

    # Public-release English note.
    for model_name, result in all_results.items():
        fi = result.get("feature_importance", [])
        if not fi:
            continue
        report_lines.append(f"## Top Features - {model_name}")
        report_lines.append("")
        for fname, fval in fi[:30]:
            report_lines.append(f"  {fname:<60} {fval:>+10.4f}")
        report_lines.append("")

    report_text = "\n".join(report_lines)

    # Public-release English note.
    report_path = save_dir / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"Evaluation report saved to {report_path}")

    # Public-release English note.
    json_path = save_dir / "evaluation_results.json"
    save_json(all_results, json_path)

    # Public-release English note.
    rows = []
    for model_name, result in all_results.items():
        m = result.get("metrics", {})
        m["model"] = model_name
        rows.append(m)
    csv_path = save_dir / "metrics_summary.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"Metrics CSV saved to {csv_path}")

    return report_text
