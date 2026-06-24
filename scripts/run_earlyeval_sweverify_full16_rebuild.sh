#!/usr/bin/env bash
# Full SWEVerify full-16 main pipeline: retrain the dual-head LightGBM on the 16-agent
# pool, regenerate every downstream artifact and post-hoc reporter, then
# refresh the paper-facing CSVs. Stops at the first failing step.
#
# Usage (background, recommended; ~3-4h):
#   nohup bash scripts/run_earlyeval_sweverify_full16_rebuild.sh > sweverify_full16_rebuild.log 2>&1 &
#   tail -f sweverify_full16_rebuild.log
set -euo pipefail

cd "$(dirname "$0")/.."
PKG_ROOT="$(pwd)"
MAIN="paper/experiments/earlyeval_lightgbm/lightgbm_main"

# Always use the swebench env interpreter; the base anaconda python has an
# incompatible numpy/pandas build that fails on import.
PY="${PYTHON_BIN:-python}"

# Conservative defaults: 2 parallel folds x 8 threads (~3GB RSS/fold,
# ~6GB total) for extra memory safety. Raise MAX_PARALLEL_FOLDS to go faster.
export FORCE="${FORCE:-1}"
export MAX_PARALLEL_FOLDS="${MAX_PARALLEL_FOLDS:-2}"
export LGBM_THREADS_PER_FOLD="${LGBM_THREADS_PER_FOLD:-8}"

echo "[pipeline] === 0. cleanup stale processes and interrupted partial folds ==="
pkill -f 'experiment paper-suite --stage lightgbm-main' 2>/dev/null || true
pkill -f 'safe_stop_dual_head_retrain.py' 2>/dev/null || true
sleep 3
# No fold here has a _SUCCESS yet; drop interrupted partials for a clean run.
rm -rf "${MAIN}/folds"/* 2>/dev/null || true

echo "[pipeline] === 1. retrain main LightGBM on the paper full-16 agent pool ==="
bash scripts/run_earlyeval_03_main_lightgbm_execute.sh

echo "[pipeline] === 2. sanity gate: exactly 16 folds, each trained on 15 agents ==="
"$PY" - <<'PY'
import glob, json, sys
fs = sorted(glob.glob("paper/experiments/earlyeval_lightgbm/lightgbm_main/folds/*/split_metadata.json"))
metas = [json.load(open(f)) for f in fs]
bad = [f for f, m in zip(fs, metas) if len(m["train_models"]) != 15]
print(f"folds={len(fs)} train_models[0]={len(metas[0]['train_models']) if metas else 'NA'} bad={len(bad)}")
sys.exit(0 if (len(fs) == 16 and not bad) else 1)
PY

echo "[pipeline] === 3. per-fold summary ==="
bash scripts/run_earlyeval_04_summarize_lightgbm_current.sh

echo "[pipeline] === 4. valid-accuracy policy sweep ==="
bash scripts/run_earlyeval_05_lightgbm_policy_sweep_valid_acc.sh

echo "[pipeline] === 5. reporting_detail (Table 4 frontier + split_check_counts) ==="
"$PY" -m earlyeval.experiments.build_reporting_detail

echo "[pipeline] === 6. internal_review_swe16 (token/rank/stop-signal); needs network ==="
"$PY" paper_reporting/build_internal_review_swe16.py \
  --run-dir paper/experiments/earlyeval_lightgbm/lightgbm_main \
  --tokenizer-mode component_sum_approx

echo "[pipeline] === 7. latency/cost proxy ==="
bash scripts/run_earlyeval_12_main_latency_cost.sh

echo "[pipeline] === 8. refresh paper-facing CSVs ==="
"$PY" paper_reporting/build_rq_tables_bundle.py

echo "[pipeline] === DONE. All artifacts and tables rebuilt on the SWEVerify full-16 main model. ==="
