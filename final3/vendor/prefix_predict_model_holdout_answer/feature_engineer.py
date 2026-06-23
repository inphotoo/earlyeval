"""
特征工程模块。

从 prefix_table 构建：
1. Dense 手工特征矩阵 (A~H 组 + J 组)
2. Sparse TF-IDF 特征矩阵 (I 组，含 thought / assistant_content)
3. 合并后的完整特征矩阵
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config
from answer_features import (
    ANSWER_BOOL_FEATURES,
    ANSWER_CATEGORICAL_FEATURES,
    ANSWER_NUMERIC_FEATURES,
    ANSWER_TEXT_COLUMNS,
)
from utils import get_logger, timer

logger = get_logger("feature_engineer")

# ── Dense 特征列定义 ──
# 数值型
NUMERIC_FEATURES = [
    # A 组
    "prefix_step_idx", "steps_observed_so_far", "actions_so_far",
    "observations_so_far", "tool_messages_so_far", "tool_calls_so_far",
    "distinct_tools_so_far", "prefix_action_chars", "prefix_feedback_chars",
    "task_prompt_chars",
    # B 组
    "last_step_tool_count", "last_step_action_chars", "last_step_feedback_chars",
    # C 组
    "read_view_so_far", "read_search_so_far", "edit_create_so_far",
    "edit_replace_so_far", "edit_insert_so_far", "edit_undo_so_far",
    "edits_so_far", "tests_so_far", "run_python_so_far", "run_cli_so_far",
    "git_ops_so_far", "cleanup_so_far", "submit_so_far",
    "bash_calls_so_far", "editor_calls_so_far",
    # D 组 (nullable int → fill -1)
    "first_edit_step", "first_test_step", "first_run_python_step",
    "first_submit_step", "first_error_step", "first_traceback_step",
    "first_read_step",
    # E 组 (nullable int → fill -1)
    "steps_since_last_edit", "steps_since_last_test", "steps_since_last_submit",
    "steps_since_last_error", "steps_since_last_traceback", "steps_since_last_read",
    # F 组
    "read_to_edit_ratio", "edit_to_test_ratio", "bash_to_editor_ratio",
    "error_per_action_ratio", "submit_per_action_ratio",
    "feedback_chars_per_action", "action_chars_per_step", "distinct_tools_per_step",
    # G 组 (nullable int → fill -1)
    "last_fail_count", "best_fail_count_so_far", "fail_count_delta_from_prev_test",
    # H 组
    "long_no_edit_streak", "long_read_streak",
    # J 组: Cognitive / Narrative 统计
    "thought_steps_so_far", "thought_density",
    "prefix_thought_chars", "avg_thought_chars_per_step", "last_thought_chars",
    "assistant_content_steps_so_far",
    "prefix_assistant_content_chars", "avg_assistant_content_chars_per_step",
    "last_assistant_content_chars",
    "thought_equals_content_rate",
    "thought_action_overlap_avg", "content_action_overlap_avg",
] + ANSWER_NUMERIC_FEATURES

# 布尔型
BOOL_FEATURES = [
    # A 组
    "has_any_action",
    # B 组
    "last_step_has_tool_output", "last_step_has_observation",
    "last_step_tool_error_seen", "last_step_traceback_seen",
    "last_step_test_fail_seen", "last_step_test_pass_seen",
    # D 组
    "first_edit_seen", "first_test_seen", "first_submit_seen",
    "first_error_seen", "first_traceback_seen",
    # G 组
    "traceback_seen", "tool_error_seen", "assertion_error_seen",
    "type_error_seen", "value_error_seen", "syntax_error_seen",
    "import_error_seen", "file_not_found_seen", "timeout_seen",
    "permission_error_seen", "test_fail_seen", "test_pass_seen",
    "all_tests_passed_seen", "test_improving_seen",
    # H 组
    "repeated_same_action_consecutive", "repeated_same_search_consecutive",
    "repeated_same_view_consecutive", "looping_read_seen",
    "edit_failed_seen", "submit_without_test_seen", "premature_submit_seen",
    "multi_submit_seen", "submit_then_edit_again_seen", "test_after_submit_seen",
] + ANSWER_BOOL_FEATURES

# 分类型
CATEGORICAL_FEATURES = [
    "last_step_action_major_type",
    "last_step_action_primary_subtype",
] + ANSWER_CATEGORICAL_FEATURES

# ── TF-IDF 文本列（按层级组织）──

# 第一层：action + feedback（基础）
TFIDF_ACTION_FEEDBACK = {
    "tfidf_task_prompt": "task_prompt_text",
    "tfidf_prefix_action": "prefix_action_text",
    "tfidf_prefix_feedback": "prefix_feedback_text",
    "tfidf_last_action": "last_action_text",
    "tfidf_last_feedback": "last_feedback_text",
}

# 第二层：thought（推荐）
TFIDF_THOUGHT = {
    "tfidf_prefix_thought": "prefix_thought_text",
    "tfidf_last_thought": "last_thought_text",
}

# 第三层：assistant_content（可选消融项）
TFIDF_ASSISTANT_CONTENT = {
    "tfidf_prefix_assistant_content": "prefix_assistant_content_text",
    "tfidf_last_assistant_content": "last_assistant_content_text",
}

# Gold-answer text blocks. These intentionally use final SWE-bench answer metadata
# for the model-holdout + answer-feature experiment.
TFIDF_GOLD_ANSWER = dict(ANSWER_TEXT_COLUMNS)

# 向后兼容：完整合集
TEXT_COLUMNS = {
    **TFIDF_ACTION_FEEDBACK,
    **TFIDF_THOUGHT,
    **TFIDF_ASSISTANT_CONTENT,
    **TFIDF_GOLD_ANSWER,
}


class FeatureEngineer:
    """
    特征工程器，负责拟合和转换。

    tfidf_level 控制 TF-IDF 文本层级：
      "action_feedback"  — 仅 action + feedback（Baseline B/E）
      "with_thought"     — + thought（Baseline C/F，主模型推荐）
      "with_content"     — + assistant_content（Baseline D，消融用）
    """

    TFIDF_LEVELS = {
        "action_feedback": TFIDF_ACTION_FEEDBACK,
        "with_thought": {**TFIDF_ACTION_FEEDBACK, **TFIDF_THOUGHT},
        "with_content": {**TFIDF_ACTION_FEEDBACK, **TFIDF_THOUGHT, **TFIDF_ASSISTANT_CONTENT},
        "with_gold_answer": {
            **TFIDF_ACTION_FEEDBACK,
            **TFIDF_THOUGHT,
            **TFIDF_ASSISTANT_CONTENT,
            **TFIDF_GOLD_ANSWER,
        },
    }

    def __init__(
        self,
        include_model_id: bool = False,
        tfidf_level: str = "with_thought",
    ):
        self.include_model_id = include_model_id
        self.tfidf_level = tfidf_level
        self.active_text_columns: dict[str, str] = dict(
            self.TFIDF_LEVELS.get(tfidf_level, self.TFIDF_LEVELS["with_thought"])
        )
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.tfidf_vectorizers: dict[str, TfidfVectorizer] = {}
        self.tfidf_reducers: dict[str, TruncatedSVD] = {}
        self.numeric_scaler: Optional[StandardScaler] = None
        self.dense_feature_names: list[str] = []
        self.fitted = False

    def fit(self, df: pd.DataFrame):
        """在训练集上拟合编码器和 TF-IDF。"""
        logger.info(f"Fitting feature engineer on {len(df)} samples")
        logger.info(f"  tfidf_level={self.tfidf_level}, "
                     f"text columns: {list(self.active_text_columns.keys())}")

        # ── 分类编码器 ──
        cats = list(CATEGORICAL_FEATURES)
        if self.include_model_id:
            cats.append("model_id")

        for col in cats:
            le = LabelEncoder()
            vals = df[col].fillna("__MISSING__").astype(str)
            # 保证 OOV/缺失槽位在词表中，否则 transform 侧把未见类映射到 __MISSING__ 时会 KeyError
            labels = sorted(set(vals.tolist()) | {"__MISSING__"})
            le.fit(labels)
            self.label_encoders[col] = le
            logger.info(f"  LabelEncoder[{col}]: {len(le.classes_)} classes")

        # ── Dense 数值标准化器（仅数值列）──
        if config.DENSE_STANDARDIZE:
            num_df = df[NUMERIC_FEATURES].copy()
            for col in num_df.columns:
                num_df[col] = pd.to_numeric(num_df[col], errors="coerce").fillna(-1)
            self.numeric_scaler = StandardScaler(with_mean=True, with_std=True)
            self.numeric_scaler.fit(num_df.values.astype(np.float32))
            logger.info("  StandardScaler[numeric]: enabled")
        else:
            self.numeric_scaler = None

        # ── TF-IDF（按 active level 拟合）──
        for name, col in self.active_text_columns.items():
            vec = TfidfVectorizer(
                ngram_range=config.TFIDF_NGRAM_RANGE,
                min_df=config.TFIDF_MIN_DF,
                max_features=config.TFIDF_MAX_FEATURES,
                sublinear_tf=True,
                dtype=np.float32,
            )
            texts = df[col].fillna("").astype(str)
            vec.fit(texts)
            self.tfidf_vectorizers[name] = vec
            n_features = len(vec.vocabulary_)
            logger.info(f"  TfidfVectorizer[{name}]: {n_features} features")

            if config.TFIDF_ENABLE_SVD and n_features > 2:
                max_comp = min(
                    config.TFIDF_SVD_DIM_PER_BLOCK,
                    n_features - 1,
                    len(df) - 1,
                )
                if max_comp >= 2:
                    X_fit = vec.transform(texts)
                    reducer = TruncatedSVD(
                        n_components=max_comp,
                        random_state=config.TFIDF_SVD_RANDOM_STATE,
                    )
                    reducer.fit(X_fit)
                    self.tfidf_reducers[name] = reducer
                    logger.info(f"  TruncatedSVD[{name}]: {n_features} -> {max_comp} dims")

        self._build_dense_feature_names()
        self.fitted = True
        logger.info(f"Total dense features: {len(self.dense_feature_names)}")

    def _build_dense_feature_names(self):
        names = list(NUMERIC_FEATURES) + list(BOOL_FEATURES)
        for col, le in self.label_encoders.items():
            for cls in le.classes_:
                names.append(f"{col}__{cls}")
        self.dense_feature_names = names

    def transform_dense(self, df: pd.DataFrame) -> np.ndarray:
        """将 prefix_df 转为 dense 特征矩阵。"""
        parts = []

        # 数值特征
        num_df = df[NUMERIC_FEATURES].copy()
        for col in num_df.columns:
            num_df[col] = pd.to_numeric(num_df[col], errors="coerce").fillna(-1)
        num_vals = num_df.values.astype(np.float32)
        if self.numeric_scaler is not None:
            num_vals = self.numeric_scaler.transform(num_vals).astype(np.float32)
        parts.append(num_vals)

        # 布尔特征
        bool_df = df[BOOL_FEATURES].copy()
        for col in bool_df.columns:
            bool_df[col] = bool_df[col].astype(float).fillna(0)
        parts.append(bool_df.values.astype(np.float32))

        # 分类 one-hot
        for col, le in self.label_encoders.items():
            vals = df[col].fillna("__MISSING__").astype(str)
            known = set(le.classes_)
            # 旧版 pickle 可能未把 __MISSING__ 编入 classes_；兼容：未见类 → __MISSING__ 或首个已知类
            if "__MISSING__" in known:
                unk = "__MISSING__"
            else:
                unk = le.classes_[0]
            vals = vals.apply(lambda x: x if x in known else unk)
            encoded = le.transform(vals)
            onehot = np.zeros((len(df), len(le.classes_)), dtype=np.float32)
            onehot[np.arange(len(df)), encoded] = 1.0
            parts.append(onehot)

        X = np.hstack(parts)
        return X

    def transform_tfidf_subset(self, df: pd.DataFrame, column_names: list[str]) -> sparse.csr_matrix:
        """将 prefix_df 转为指定列的 TF-IDF 稀疏矩阵。"""
        tfidf_parts = []
        for name in column_names:
            if name in self.tfidf_vectorizers:
                col = self.active_text_columns[name]
                vec = self.tfidf_vectorizers[name]
                texts = df[col].fillna("").astype(str)
                X_tfidf = vec.transform(texts)
                reducer = self.tfidf_reducers.get(name)
                if reducer is not None:
                    # 每个 TF-IDF block 独立降维，再做 block-level 融合。
                    X_tfidf = sparse.csr_matrix(
                        reducer.transform(X_tfidf).astype(np.float32)
                    )
                tfidf_parts.append(X_tfidf)
        return sparse.hstack(tfidf_parts, format="csr") if tfidf_parts else sparse.csr_matrix((len(df), 0))

    def transform_tfidf(self, df: pd.DataFrame) -> sparse.csr_matrix:
        """将 prefix_df 转为 TF-IDF 稀疏矩阵（仅已拟合的列）。"""
        return self.transform_tfidf_subset(df, list(self.active_text_columns.keys()))

    def transform_combined(self, df: pd.DataFrame, tfidf_column_names: list[str]) -> sparse.csr_matrix:
        """Dense + 指定 TF-IDF 列 拼接为一个稀疏矩阵。"""
        X_dense = self.transform_dense(df)
        X_tfidf = self.transform_tfidf_subset(df, tfidf_column_names)
        X_dense_sp = sparse.csr_matrix(X_dense)
        return sparse.hstack([X_dense_sp, X_tfidf], format="csr")

    def transform_all(self, df: pd.DataFrame) -> sparse.csr_matrix:
        """Dense + 全部 TF-IDF 拼接为一个稀疏矩阵。"""
        return self.transform_combined(df, list(self.active_text_columns.keys()))

    def get_all_feature_names(self) -> list[str]:
        """返回拼接后的全部特征名。"""
        names = list(self.dense_feature_names)
        for tfidf_name in self.active_text_columns:
            reducer = self.tfidf_reducers.get(tfidf_name)
            if reducer is not None:
                n_dim = reducer.n_components
                names.extend([f"{tfidf_name}__svd_{i}" for i in range(n_dim)])
            else:
                vec = self.tfidf_vectorizers[tfidf_name]
                feat_names = vec.get_feature_names_out()
                for fn in feat_names:
                    names.append(f"{tfidf_name}__{fn}")
        return names

    def get_tfidf_block_ranges(self) -> dict[str, tuple[int, int]]:
        """返回每个 TF-IDF 块在最终矩阵中的列范围（用于 ablation 列删除）。"""
        dense_dim = len(self.dense_feature_names)
        offset = dense_dim
        ranges = {}
        for name in self.active_text_columns:
            reducer = self.tfidf_reducers.get(name)
            if reducer is not None:
                n = reducer.n_components
            else:
                vec = self.tfidf_vectorizers[name]
                n = len(vec.vocabulary_)
            ranges[name] = (offset, offset + n)
            offset += n
        return ranges

    def get_tfidf_feature_names_for_columns(self, column_names: list[str]) -> list[str]:
        """返回指定 TF-IDF 子集（含降维后）的特征名。"""
        names = []
        for tfidf_name in column_names:
            if tfidf_name not in self.active_text_columns:
                continue
            reducer = self.tfidf_reducers.get(tfidf_name)
            if reducer is not None:
                n_dim = reducer.n_components
                names.extend([f"{tfidf_name}__svd_{i}" for i in range(n_dim)])
            else:
                vec = self.tfidf_vectorizers[tfidf_name]
                feat_names = vec.get_feature_names_out()
                names.extend([f"{tfidf_name}__{fn}" for fn in feat_names])
        return names

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"FeatureEngineer saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "FeatureEngineer":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"FeatureEngineer loaded from {path}")
        return obj
