# Examples

Tiny inputs used by the smoke tests. **Everything here is synthetic.** These files exist so that a
fresh clone can exercise the code paths without benchmark data, trained models, or GPU time. None of
the values are measurements, and nothing in this directory should be cited or compared against the
paper.

## `smoke_predictions.csv`

A hand-constructed prefix-prediction table for 8 trajectories across 3 fictional agents, with the
schema the safe-stop policy expects:

| Column | Meaning |
|:---|:---|
| `traj_id` | Trajectory identifier; all prefixes of one run share it |
| `instance_id` | Benchmark task identifier |
| `model_id` | Agent (scaffold + base model) that produced the run |
| `label` | Ground-truth final outcome, `1` = resolved |
| `prefix_step_idx` | Step index of this prefix within the trajectory |
| `n_steps_total` | Length of the full trajectory |
| `prob_cal_safe_success__I_LightGBM_Dense_AF` | Calibrated success-head probability |
| `prob_cal_safe_failure__I_LightGBM_Dense_AF` | Calibrated failure-head probability |

The probability columns follow the naming rule `prob_{raw\|cal}_safe_{success\|failure}__{predictor}`,
where `cal` denotes Platt-calibrated scores. The predictor name and score mode come from the policy
preset in [`configs/policy_presets.yaml`](../configs/policy_presets.yaml).

The eight trajectories are constructed to cover every branch of the decision rule: three cross the
success threshold, three cross the failure threshold, and two stay unconfident to completion — so
the resulting summary exercises early-success stops, early-failure stops, and undecided runs.

Run it with either of:

```bash
python -m earlyeval.cli pipeline current-safe-stop --mode smoke

python -m earlyeval.cli policy apply \
  --preset current_safe_stop \
  --predictions examples/smoke_predictions.csv \
  --output-dir outputs/current_safe_stop_smoke
```

To apply the policy to real predictions, point `--predictions` at your own table with the same
columns. Only `traj_id`, `label`, `prefix_step_idx`, and the two probability columns are required.
