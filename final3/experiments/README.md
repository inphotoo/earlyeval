# final3 Experiment Sets

`experiments/` owns runnable experiment-set code. It is intentionally separate
from `core/`: core modules define reusable contracts and algorithms, while
experiment modules say which data to read, which comparison to run, and where
paper-facing outputs go.

Current modules:

- `paper_bundle.py`: materializes current ICSE paper inputs under `paper/data/`.
- `registry.py`: reads `configs/experiment_registry.yaml`.
- `rq_final.py`: final paper RQ planning, low-memory prefix audit, split manifest generation, and smoke orchestration.
- `rq_final_ablation.py`: SWEVerify-only feature/component ablation runner that reuses the locked split and preserves raw prediction parquet files.
- `robustness_15pct.py`: Toolathlon / TerminalBench process-feature robustness run with random 15% model holdout.

Run order for paper work:

```bash
python -m final3.cli check preflight --experiment all
python -m final3.cli experiment materialize-paper --mode link
python -m final3.cli report paper-tables
```

Run the final RQ smoke path:

```bash
python -m final3.cli experiment rq-final --stage smoke
```

The smoke path does not train models. It writes a run plan, prefix audit tables,
and leave-one-test-model split manifests under `paper/experiments/rq_final_smoke/`.

Postprocess completed LightGBM folds into a valid-selected accuracy frontier:

```bash
bash scripts/run_rq_final_05_lightgbm_policy_sweep_valid_acc.sh
```

This reads each completed fold's raw valid/test prediction parquet files, selects
thresholds on valid at target decision-accuracy levels from 0.75 to 0.95, then
maximizes saved steps among policies that pass that valid accuracy target. It
applies the selected policy unchanged to test. Outputs go under
`paper/experiments/rq_final_lightgbm_17/lightgbm_main/policy_sweeps/valid_accuracy_075_095/`.

Run the SWEVerify ablation smoke or full serial sweep:

```bash
bash scripts/run_rq_final_08_ablation_smoke.sh
bash scripts/run_rq_final_08_ablation_execute.sh
bash scripts/run_rq_final_08_ablation_random4.sh
bash scripts/run_rq_final_08_ablation_balanced4.sh
```

These write per-profile outputs under
`paper/experiments/rq_final_lightgbm_17/ablations/sweverify/<run_subdir>/`.
Each fold keeps `valid_predictions_safe_stop.parquet` and
`test_predictions_safe_stop.parquet` for later analysis.
Use `run_rq_final_08_ablation_balanced4.sh` for the paper-facing quick
ablation slice: it fixes four representative held-out models across the
filtered 16-model capability range instead of relying on a random seed.

Run the lightweight robustness baselines on Toolathlon and TerminalBench:

```bash
bash scripts/run_rq_final_07_robustness_15pct.sh
```

This holds out a deterministic random 15% of models as final test, splits the
remaining models into train/valid by `instance_id`, trains a process-only
LightGBM dual-head predictor, selects the safe-stop policy on valid, and applies
it unchanged to test. Outputs go under
`paper/experiments/rq_final_lightgbm_17/robustness_15pct_model_holdout/`.
