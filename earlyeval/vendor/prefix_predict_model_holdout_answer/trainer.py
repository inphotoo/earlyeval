'Public-release English note.'
from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

import config
from utils import get_logger, timer

logger = get_logger("trainer")


class SingleClassBinaryProbaEstimator:
    'Public-release English note.'

    def __init__(self, only_positive: bool):
        self.only_positive = bool(only_positive)
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X, y, sample_weight=None):
        return self

    def predict_proba(self, X):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        p1 = 1.0 if self.only_positive else 0.0
        out = np.zeros((n, 2), dtype=np.float64)
        out[:, 1] = p1
        out[:, 0] = 1.0 - p1
        return out

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_logistic_regression(
    X_train: sparse.csr_matrix | np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray | None,
    X_valid: sparse.csr_matrix | np.ndarray,
    y_valid: np.ndarray,
    model_name: str = "lr",
    C_grid: list[float] | None = None,
) -> Any:
    'Public-release English note.'
    y_train = np.asarray(y_train).astype(int).ravel()
    u_tr = np.unique(y_train)
    if len(u_tr) < 2:
        logger.warning(
            'Public-release English note.'
            'Public-release English note.'
        )
        return SingleClassBinaryProbaEstimator(only_positive=(int(u_tr[0]) == 1))

    C_grid = C_grid or config.LR_C_GRID
    logger.info(f"[{model_name}] Training LR with C grid: {C_grid}")

    best_score = -1
    best_C = C_grid[0]
    use_gpu_backend = False

    for C in C_grid:
        with timer(logger, f"[{model_name}] C={C}"):
            lr, this_gpu = _fit_lr_with_auto_backend(
                X_train=X_train,
                y_train=y_train,
                w_train=w_train,
                C=C,
            )
            use_gpu_backend = use_gpu_backend or this_gpu

            # valid AUC
            from sklearn.metrics import roc_auc_score, log_loss
            y_pred_prob = lr.predict_proba(X_valid)[:, 1]
            auc = roc_auc_score(y_valid, y_pred_prob)
            ll = log_loss(y_valid, y_pred_prob)
            logger.info(f"  C={C}: AUC={auc:.4f}, LogLoss={ll:.4f}")

            if auc > best_score:
                best_score = auc
                best_C = C

    logger.info(f"[{model_name}] Best C={best_C} with AUC={best_score:.4f}")
    logger.info(f"[{model_name}] Backend used during search: {'GPU(cuML)' if use_gpu_backend else 'CPU(sklearn)'}")

    # Public-release English note.
    final_lr, final_gpu = _fit_lr_with_auto_backend(
        X_train=X_train,
        y_train=y_train,
        w_train=w_train,
        C=best_C,
    )
    logger.info(f"[{model_name}] Final backend: {'GPU(cuML)' if final_gpu else 'CPU(sklearn)'}")

    return final_lr


def _fit_lr_with_auto_backend(
    X_train: sparse.csr_matrix | np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray | None,
    C: float,
):
    'Public-release English note.'
    if config.LR_PREFER_GPU:
        model = _fit_cuml_lr(X_train, y_train, w_train, C)
        if model is not None:
            return model, True
    is_sparse = sparse.issparse(X_train)
    solver = config.LR_CPU_SOLVER_SPARSE if is_sparse else config.LR_CPU_SOLVER_DENSE
    cpu_kwargs = dict(
        C=C,
        penalty="l2",
        solver=solver,
        max_iter=config.LR_CPU_MAX_ITER,
        class_weight="balanced",
        random_state=config.SPLIT_SEED,
    )
    # Public-release English note.
    if solver == "liblinear":
        cpu_kwargs["n_jobs"] = -1
    model = LogisticRegression(
        **cpu_kwargs,
    )
    model.fit(X_train, y_train, sample_weight=w_train)
    return model, False


def _fit_cuml_lr(
    X_train: sparse.csr_matrix | np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray | None,
    C: float,
):
    try:
        import cupy as cp
        from cuml.linear_model import LogisticRegression as CuLogisticRegression
    except Exception:
        return None

    try:
        if sparse.issparse(X_train):
            from cupyx.scipy import sparse as cpx_sparse
            X_gpu = cpx_sparse.csr_matrix(X_train)
        else:
            X_gpu = cp.asarray(X_train)
        y_gpu = cp.asarray(y_train.astype(np.float32))
        w_gpu = cp.asarray(w_train.astype(np.float32)) if w_train is not None else None

        model = CuLogisticRegression(
            C=C,
            penalty="l2",
            max_iter=config.LR_GPU_MAX_ITER,
            fit_intercept=True,
            tol=1e-4,
        )
        model.fit(X_gpu, y_gpu, sample_weight=w_gpu)
        return _CuMLEstimatorAdapter(model)
    except Exception as e:
        logger.warning(f"cuML LR failed, fallback to sklearn CPU. reason={e}")
        return None


class _CuMLEstimatorAdapter:
    'Public-release English note.'

    def __init__(self, model):
        self.model = model
        self.coef_ = None
        try:
            coef = getattr(model, "coef_", None)
            if coef is not None:
                if hasattr(coef, "get"):
                    coef = coef.get()
                self.coef_ = np.asarray(coef)
        except Exception:
            self.coef_ = None

    def predict_proba(self, X):
        import cupy as cp
        if sparse.issparse(X):
            from cupyx.scipy import sparse as cpx_sparse
            X_gpu = cpx_sparse.csr_matrix(X)
        else:
            X_gpu = cp.asarray(X)
        probs = self.model.predict_proba(X_gpu)
        if hasattr(probs, "get"):
            probs = probs.get()
        return np.asarray(probs)

    def __getattr__(self, item):
        return getattr(self.model, item)


class _SingleClassLGBMPredictor:
    'Public-release English note.'

    def __init__(self, p_pos: float):
        self.p_pos = float(p_pos)
        self.best_iteration = 0

    def predict(self, X, num_iteration=None, raw_score=False, **kwargs):
        n = X.shape[0] if hasattr(X, "shape") else len(X)
        return np.full(n, self.p_pos, dtype=np.float64)


def train_lightgbm(
    X_train: sparse.csr_matrix | np.ndarray,
    y_train: np.ndarray,
    w_train: np.ndarray | None,
    X_valid: sparse.csr_matrix | np.ndarray,
    y_valid: np.ndarray,
    w_valid: np.ndarray | None,
    feature_names: list[str] | None = None,
    model_name: str = "lgbm",
) -> Any:
    'Public-release English note.'
    import lightgbm as lgb

    y_train = np.asarray(y_train).astype(int).ravel()
    u_tr = np.unique(y_train)
    if len(u_tr) < 2:
        p = 1.0 if int(u_tr[0]) == 1 else 0.0
        logger.warning(
            'Public-release English note.'
            'Public-release English note.'
        )
        return _SingleClassLGBMPredictor(p)

    logger.info(f"[{model_name}] Training LightGBM, X shape: {X_train.shape}")

    params = dict(config.LGBM_PARAMS)

    dtrain = lgb.Dataset(
        X_train, label=y_train, weight=w_train,
        feature_name=feature_names or "auto",
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(
        X_valid, label=y_valid, weight=w_valid,
        feature_name=feature_names or "auto",
        reference=dtrain,
        free_raw_data=False,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50),
    ]

    with timer(logger, f"[{model_name}] LightGBM training"):
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

    logger.info(f"[{model_name}] Best iteration: {booster.best_iteration}")
    logger.info(f"[{model_name}] Best valid logloss: {booster.best_score['valid']['binary_logloss']:.4f}")

    return booster


def save_model(model, path: Path):
    'Public-release English note.'
    is_lgbm_booster = False
    try:
        import lightgbm as lgb
        is_lgbm_booster = isinstance(model, lgb.Booster)
    except Exception:
        is_lgbm_booster = False

    if is_lgbm_booster:
        model.save_model(str(path))
    else:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


def load_model(path: Path):
    'Public-release English note.'
    if str(path).endswith(".lgb"):
        import lightgbm as lgb
        return lgb.Booster(model_file=str(path))
    with open(path, "rb") as f:
        return pickle.load(f)
