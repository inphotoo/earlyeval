# policies

`policies/` contains safe-stop decision logic and policy application helpers.

## Files

- `safe_stop.py`: dual-head safe-stop decisions, aggregate metrics, and
  per-agent summaries.
- `presets.py`: loads `configs/policy_presets.yaml` and provides fallback
  presets.
- `apply.py`: reads prediction tables, applies a policy, and writes decisions,
  summaries, and metadata.

## Required Prediction Columns

- `traj_id`
- `label`
- `prefix_step_idx`
- `orig_model_id` or `model_id`
- `prob_cal_safe_success__I_LightGBM_Dense_AF`
- `prob_cal_safe_failure__I_LightGBM_Dense_AF`

## Example

```bash
python -m final3.cli policy apply \
  --preset current_safe_stop \
  --predictions examples/smoke_predictions.csv \
  --output-dir outputs/current_safe_stop_smoke
```

Outputs include `policy_decisions.csv`, `policy_summary.csv`,
`policy_per_agent.csv`, and `run_metadata.json`.
