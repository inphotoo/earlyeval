"""
项目全局配置。
所有路径、超参数、特征开关统一在此管理。
"""
import os
from pathlib import Path

# ── 项目根目录 ──
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNTIME_ROOT = (
    PROJECT_ROOT.parents[2] / "outputs" / "vendor_runtime" / "prefix_predict_model_holdout_answer"
)
RUNTIME_ROOT = Path(os.environ.get("FINAL3_VENDOR_RUNTIME_ROOT", str(DEFAULT_RUNTIME_ROOT)))
DATA_DIR = RUNTIME_ROOT / "data"
MODEL_DIR = RUNTIME_ROOT / "models"
REPORT_DIR = RUNTIME_ROOT / "reports"
LOG_DIR = RUNTIME_ROOT / "logs"

for d in [DATA_DIR, MODEL_DIR, REPORT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── GPU 设置：使用 3 号 GPU ──
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ── 输入数据 ──
# 用户需要修改此路径指向 tool-*.parquet 所在目录
PARQUET_INPUT_DIR = os.environ.get(
    "SWE_PARQUET_DIR",
    str(PROJECT_ROOT / "raw_data"),
)

# ── Step 重建 ──
STEP_TABLE_PATH = DATA_DIR / "step_table.parquet"
PREFIX_TABLE_PATH = DATA_DIR / "prefix_table.parquet"
PREFIX_TABLE_FILTERED_PATH = DATA_DIR / "prefix_table_filtered.parquet"
FEATURE_MATRIX_PATH = DATA_DIR / "feature_matrix.npz"  # sparse
DENSE_FEATURE_PATH = DATA_DIR / "dense_features.parquet"

# Prefix 构建时的分块写盘大小（按样本条数计）
# 如果 prefix 样本很多，可以适当调小以降低单次内存占用。
PREFIX_CHUNK_SIZE = int(os.environ.get("SWE_PREFIX_CHUNK_SIZE", 50000))

# ── 数据切分 ──
SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# ── 动作分类 ──
MAJOR_TYPE_MAP = {
    "read_view": "read",
    "read_search": "read",
    "edit_create": "edit",
    "edit_replace": "edit",
    "edit_insert": "edit",
    "edit_undo": "edit",
    "test": "execute",
    "run_python": "execute",
    "run_cli": "execute",
    "git": "execute",
    "cleanup": "cleanup",
    "submit": "submit",
}

ALL_SUBTYPES = [
    "read_view", "read_search",
    "edit_create", "edit_replace", "edit_insert", "edit_undo",
    "test", "run_python", "run_cli", "git",
    "cleanup", "submit",
]

# ── TF-IDF 配置 ──
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 5
TFIDF_MAX_FEATURES = 30000
# 按文本块分别降维：每个 TF-IDF block 先向量化，再做 TruncatedSVD。
# 这样既保留各块语义独立性，也把总维度压到几百量级。
TFIDF_ENABLE_SVD = True
TFIDF_SVD_DIM_PER_BLOCK = int(os.environ.get("SWE_TFIDF_SVD_DIM_PER_BLOCK", 64))
TFIDF_SVD_RANDOM_STATE = SPLIT_SEED

# ── LR 超参搜索 ──
LR_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
LR_PREFER_GPU = True
LR_GPU_MAX_ITER = 2000
LR_CPU_MAX_ITER = 4000
LR_CPU_SOLVER_DENSE = "lbfgs"
LR_CPU_SOLVER_SPARSE = "liblinear"

# ── Dense 预处理 ──
# 对 LR 的 dense 分支，标准化可显著降低尺度差异导致的收敛问题。
DENSE_STANDARDIZE = True

# ── LightGBM ──
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 8,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "seed": SPLIT_SEED,
    "device": "gpu",
    "gpu_device_id": 0,  # 已通过 CUDA_VISIBLE_DEVICES=1 映射
}

# ── 数据过滤 ──
MIN_TRAJECTORY_STEPS = 5

# ── 评估：prefix 深度分桶（0 单独；1~30 每 3 步；31~50 每 3 步，末桶 49~50；其余进长尾）──
STEP_BUCKETS = [
    ("step=0", 0, 0),
    ("step=1~3", 1, 3),
    ("step=4~6", 4, 6),
    ("step=7~9", 7, 9),
    ("step=10~12", 10, 12),
    ("step=13~15", 13, 15),
    ("step=16~18", 16, 18),
    ("step=19~21", 19, 21),
    ("step=22~24", 22, 24),
    ("step=25~27", 25, 27),
    ("step=28~30", 28, 30),
    ("step=31~33", 31, 33),
    ("step=34~36", 34, 36),
    ("step=37~39", 37, 39),
    ("step=40~42", 40, 42),
    ("step=43~45", 43, 45),
    ("step=46~48", 46, 48),
    ("step=49~51", 49, 51),
    ("step>51", 52, 999),
]

# ── 轨迹级「按 Prec(S)/Prec(F) 选 Thr」的候选阈值网格 ──
# 对称规则：成功侧先触发用 p>=thr，失败侧先触发用 p<=1-thr。
# 若仅从 0.5 起搜，在概率整体偏低时可能永远找不到满足 Prec(S)≥目标的 Thr_S（报告里左列会全是「—」）。
PRECISION_SAVINGS_THR_GRID_START = 0.05
PRECISION_SAVINGS_THR_GRID_STOP = 1.0
PRECISION_SAVINGS_THR_GRID_STEP = 0.005

# ── 轨迹级早停：非对称双阈值（不再用「同一 thr + p≤1−thr」对称规则）──
# 扫 **Thr_S**（满足 Prec(S)≥目标）时，失败侧固定为「先遇 p≤ANCHOR_P_FAIL」；
# 扫 **P_fail**（满足 Prec(F)≥目标）时，成功侧固定为「先遇 p≥ANCHOR_THR_SUCCESS」。
# 二者与最终 **联合扫描**（左列 Thr_S + 右列 P_fail）一致。
PRECISION_SAVINGS_ANCHOR_P_FAIL = 0.25
PRECISION_SAVINGS_ANCHOR_THR_SUCCESS = 0.55
# 失败侧截断 p 的搜索网格（字面量：p<=p_fail 触发失败）
PRECISION_SAVINGS_FAILURE_P_GRID_START = 0.02
PRECISION_SAVINGS_FAILURE_P_GRID_STOP = 0.80
PRECISION_SAVINGS_FAILURE_P_GRID_STEP = 0.005

# ── 日志配置 ──
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s"
