"""Validation-only probability calibration helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss


def _clip_prob(prob: np.ndarray, eps: float) -> np.ndarray:
    return np.clip(np.asarray(prob, dtype=np.float64), eps, 1.0 - eps)


def _logit(prob: np.ndarray, eps: float) -> np.ndarray:
    clipped = _clip_prob(prob, eps)
    return np.log(clipped / (1.0 - clipped))


def _safe_log_loss(y_true: np.ndarray, prob: np.ndarray) -> float:
    try:
        return float(log_loss(y_true, _clip_prob(prob, 1e-6), labels=[0, 1]))
    except Exception:
        return float("nan")


def _safe_brier(y_true: np.ndarray, prob: np.ndarray) -> float:
    try:
        return float(brier_score_loss(y_true, np.asarray(prob, dtype=np.float64)))
    except Exception:
        return float("nan")


@dataclass
class SigmoidProbabilityCalibrator:
    """One-dimensional Platt/sigmoid calibration on raw model probabilities."""

    estimator: Any | None
    constant_prob: float | None = None
    eps: float = 1e-6

    def predict(self, raw_prob: np.ndarray) -> np.ndarray:
        raw_prob = np.asarray(raw_prob, dtype=np.float64).ravel()
        if self.constant_prob is not None:
            return np.full(raw_prob.shape, float(self.constant_prob), dtype=np.float64)
        if self.estimator is None:
            return _clip_prob(raw_prob, self.eps)
        x = _logit(raw_prob, self.eps).reshape(-1, 1)
        return self.estimator.predict_proba(x)[:, 1].astype(np.float64)


def fit_sigmoid_calibrator(
    raw_prob_valid: np.ndarray,
    y_valid: np.ndarray,
    sample_weight: np.ndarray | None = None,
    *,
    eps: float = 1e-6,
) -> SigmoidProbabilityCalibrator:
    """Fit a validation-only sigmoid calibrator.

    The base model is already trained.  This learns only a monotonic-ish
    probability mapping from validation raw probabilities to validation labels.
    """
    raw_prob_valid = np.asarray(raw_prob_valid, dtype=np.float64).ravel()
    y_valid = np.asarray(y_valid, dtype=int).ravel()
    if raw_prob_valid.shape[0] != y_valid.shape[0]:
        raise ValueError(
            f"raw_prob_valid/y_valid length mismatch: {raw_prob_valid.shape[0]} vs {y_valid.shape[0]}"
        )
    classes = np.unique(y_valid)
    if len(classes) < 2:
        return SigmoidProbabilityCalibrator(
            estimator=None,
            constant_prob=float(classes[0]),
            eps=eps,
        )

    x_valid = _logit(raw_prob_valid, eps).reshape(-1, 1)
    estimator = LogisticRegression(
        C=1e6,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=0,
    )
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = np.asarray(sample_weight, dtype=np.float64).ravel()
    estimator.fit(x_valid, y_valid, **fit_kwargs)
    return SigmoidProbabilityCalibrator(estimator=estimator, constant_prob=None, eps=eps)


def calibration_summary_row(
    model_name: str,
    calibrator: SigmoidProbabilityCalibrator,
    y_valid: np.ndarray,
    raw_prob_valid: np.ndarray,
    y_test: np.ndarray,
    raw_prob_test: np.ndarray,
) -> dict[str, Any]:
    cal_prob_valid = calibrator.predict(raw_prob_valid)
    cal_prob_test = calibrator.predict(raw_prob_test)
    coef = None
    intercept = None
    if calibrator.estimator is not None:
        coef = float(calibrator.estimator.coef_.ravel()[0])
        intercept = float(calibrator.estimator.intercept_.ravel()[0])

    return {
        "model": model_name,
        "method": "sigmoid_platt_on_valid_logits",
        "coef": coef,
        "intercept": intercept,
        "valid_raw_mean": float(np.mean(raw_prob_valid)),
        "valid_cal_mean": float(np.mean(cal_prob_valid)),
        "valid_label_rate": float(np.mean(y_valid)),
        "valid_raw_brier": _safe_brier(y_valid, raw_prob_valid),
        "valid_cal_brier": _safe_brier(y_valid, cal_prob_valid),
        "valid_raw_logloss": _safe_log_loss(y_valid, raw_prob_valid),
        "valid_cal_logloss": _safe_log_loss(y_valid, cal_prob_valid),
        "test_raw_mean": float(np.mean(raw_prob_test)),
        "test_cal_mean": float(np.mean(cal_prob_test)),
        "test_label_rate": float(np.mean(y_test)),
        "test_raw_brier": _safe_brier(y_test, raw_prob_test),
        "test_cal_brier": _safe_brier(y_test, cal_prob_test),
        "test_raw_logloss": _safe_log_loss(y_test, raw_prob_test),
        "test_cal_logloss": _safe_log_loss(y_test, cal_prob_test),
    }
