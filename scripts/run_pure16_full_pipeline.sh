#!/usr/bin/env bash
# Full pure-16 main pipeline: retrain the dual-head LightGBM on the 16-agent
# pool, regenerate every downstream artifact and post-hoc reporter, then
# refresh the paper-facing CSVs. Stops at the first failing step.
#
# Usage (background, recommended; ~3-4h):
#   nohup bash scripts/run_pure16_full_pipeline.sh > pure16_pipeline.log 2>&1 &
#   tail -f pure16_pipeline.log
set -euo pipefail

cd "$(dirname "$0")/.."
PKG_ROOT="$(pwd)"
MAIN="paper/experiments/rq_final_lightgbm_17/lightgbm_main"

# Always use the swebench env interpreter; the base anaconda python has an
# incompatible numpy/pandas build that fails on import.
PY="${PYTHON_BIN:-/home/ugproj/anaconda3/envs/swebench/bin/python}"

# Conservative defaults: 2 parallel folds x 8 threads (~3GB RSS/fold,
# ~6GB total) for extra memory safety. Raise MAX_PARALLEL_FOLDS to go faster.
export FORCE="${FORCE:-1}"
export MAX_PARALLEL_FOLDS="${MAX_PARALLEL_FOLDS:-2}"
export LGBM_THREADS_PER_FOLD="${LGBM_THREADS_PER_FOLD:-8}"

echo "[pipeline] === 0. cleanup stale processes and interrupted partial folds ==="
pkill -f 'experiment rq-final --stage lightgbm-main' 2>/dev/null || true
pkill -f 'safe_stop_dual_head_retrain.py' 2>/dev/null || true
sleep 3
# No fold here has a _SUCCESS yet; drop interrupted partials for a clean run.
rm -rf "${MAIN}/folds"/* 2>/dev/null || true

echo "[pipeline] === 1. retrain main LightGBM on the pure 16-agent pool ==="
bash scripts/run_rq_final_03_main_lightgbm_execute.sh

echo "[pipeline] === 2. sanity gate: exactly 16 folds, each trained on 15 agents ==="
"$PY" - <<'PY'
import glob, json, sys
fs = sorted(glob.glob("paper/experiments/rq_final_lightgbm_17/lightgbm_main/folds/*/split_metadata.json"))
metas = [json.load(open(f)) for f in fs]
bad = [f for f, m in zip(fs, metas) if len(m["train_models"]) != 15]
print(f"folds={len(fs)} train_models[0]={len(metas[0]['train_models']) if metas else 'NA'} bad={len(bad)}")
sys.exit(0 if (len(fs) == 16 and not bad) else 1)
PY

echo "[pipeline] === 3. per-fold summary ==="
bash scripts/run_rq_final_04_summarize_lightgbm_current.sh

echo "[pipeline] === 4. valid-accuracy policy sweep ==="
bash scripts/run_rq_final_05_lightgbm_policy_sweep_valid_acc.sh

echo "[pipeline] === 5. reporting_detail (Table 4 frontier + split_check_counts) ==="
"$PY" -m final3.experiments.build_reporting_detail

echo "[pipeline] === 6. internal_review_swe16 (token/rank/stop-signal); needs network ==="
"$PY" paper/experiments/rq_final_lightgbm_17/build_internal_review_swe16.py \
  --run-dir paper/experiments/rq_final_lightgbm_17/lightgbm_main \
  --tokenizer-mode component_sum_approx

echo "[pipeline] === 7. latency/cost proxy ==="
bash scripts/run_rq_final_12_main_latency_cost.sh

echo "[pipeline] === 8. refresh paper-facing CSVs ==="
"$PY" paper/icse_submission_draft/data/refresh_tables.py

echo "[pipeline] === DONE. All artifacts and tables rebuilt on the pure-16 main model. ==="
