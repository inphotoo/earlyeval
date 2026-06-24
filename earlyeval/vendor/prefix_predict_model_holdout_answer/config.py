'Public-release English note.'
import os
from pathlib import Path

# Public-release English note.
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RUNTIME_ROOT = (
    PROJECT_ROOT.parents[2] / "outputs" / "vendor_runtime" / "prefix_predict_model_holdout_answer"
)
RUNTIME_ROOT = Path(os.environ.get("EARLYEVAL_VENDOR_RUNTIME_ROOT", str(DEFAULT_RUNTIME_ROOT)))
DATA_DIR = RUNTIME_ROOT / "data"
MODEL_DIR = RUNTIME_ROOT / "models"
REPORT_DIR = RUNTIME_ROOT / "reports"
LOG_DIR = RUNTIME_ROOT / "logs"

for d in [DATA_DIR, MODEL_DIR, REPORT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Public-release English note.
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Public-release English note.
# Public-release English note.
PARQUET_INPUT_DIR = os.environ.get(
    "SWE_PARQUET_DIR",
    str(PROJECT_ROOT / "raw_data"),
)

# Public-release English note.
STEP_TABLE_PATH = DATA_DIR / "step_table.parquet"
PREFIX_TABLE_PATH = DATA_DIR / "prefix_table.parquet"
PREFIX_TABLE_FILTERED_PATH = DATA_DIR / "prefix_table_filtered.parquet"
FEATURE_MATRIX_PATH = DATA_DIR / "feature_matrix.npz"  # sparse
DENSE_FEATURE_PATH = DATA_DIR / "dense_features.parquet"

# Public-release English note.
# Public-release English note.
PREFIX_CHUNK_SIZE = int(os.environ.get("SWE_PREFIX_CHUNK_SIZE", 50000))

# Public-release English note.
SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# Public-release English note.
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

# Public-release English note.
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 5
TFIDF_MAX_FEATURES = 30000
# Public-release English note.
# Public-release English note.
TFIDF_ENABLE_SVD = True
TFIDF_SVD_DIM_PER_BLOCK = int(os.environ.get("SWE_TFIDF_SVD_DIM_PER_BLOCK", 64))
TFIDF_SVD_RANDOM_STATE = SPLIT_SEED

# Public-release English note.
LR_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
LR_PREFER_GPU = True
LR_GPU_MAX_ITER = 2000
LR_CPU_MAX_ITER = 4000
LR_CPU_SOLVER_DENSE = "lbfgs"
LR_CPU_SOLVER_SPARSE = "liblinear"

# Public-release English note.
# Public-release English note.
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
    "gpu_device_id": 0,  # Public-release English note.
}

# Public-release English note.
MIN_TRAJECTORY_STEPS = 5

# Public-release English note.
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

# Public-release English note.
# Public-release English note.
# Public-release English note.
PRECISION_SAVINGS_THR_GRID_START = 0.05
PRECISION_SAVINGS_THR_GRID_STOP = 1.0
PRECISION_SAVINGS_THR_GRID_STEP = 0.005

# Public-release English note.
# Public-release English note.
# Public-release English note.
# Public-release English note.
PRECISION_SAVINGS_ANCHOR_P_FAIL = 0.25
PRECISION_SAVINGS_ANCHOR_THR_SUCCESS = 0.55
# Public-release English note.
PRECISION_SAVINGS_FAILURE_P_GRID_START = 0.02
PRECISION_SAVINGS_FAILURE_P_GRID_STOP = 0.80
PRECISION_SAVINGS_FAILURE_P_GRID_STEP = 0.005

# Public-release English note.
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s"
