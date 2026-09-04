# Examples

A tiny input used by the smoke tests, so that a fresh clone can exercise the safe-stop code path
without benchmark data, trained models, or GPU time.

## `smoke_predictions.csv`

Nine trajectories from three real mini-SWE-agent runs on SWE-bench Verified.

**What is real.** Task ids, agent names, ground-truth `resolved` labels, and trajectory lengths come
from the official [SWE-bench experiments](https://github.com/SWE-bench/experiments) repository,
`evaluation/verified/<run_id>/per_instance_details.json`:

| `model_id` | Leaderboard run | Resolve rate |
|:---|:---|:---:|
| `mini-swe-agent__claude-opus-4.5` | `20251124_mini-v1.16.0_claude-opus-4-5-20251101` | 74.4% |
| `mini-swe-agent__gpt-5.2-high` | `20251211_mini-v1.17.2_gpt-5.2-2025-12-11-high` | 71.8% |
| `mini-swe-agent__glm-4.6` | `20251201_mini-v1.17.1_glm-4.6` | 55.4% |

The nine tasks span `astropy`, `django`, `matplotlib`, and `requests`. `n_steps_total` is the run's
real `api_calls` count.

**What is not real.** The two probability columns are the output of a trained EarlyEval predictor,
which is not part of this code-only release, so they cannot be taken from any public source. They
are generated as monotone curves consistent with each run's real label, purely so the fixture
exercises every branch of the decision rule. **Do not read them as measurements, cite them, or
compare them against the paper.**

## Schema

| Column | Meaning | Source |
|:---|:---|:---|
| `traj_id` | Trajectory identifier; all prefixes of one run share it | real |
| `instance_id` | SWE-bench Verified task identifier | real |
| `model_id` | Agent (scaffold + base model) that produced the run | real |
| `label` | Ground-truth final outcome, `1` = resolved | real |
| `prefix_step_idx` | Step index of this prefix within the trajectory | derived |
| `n_steps_total` | Length of the full trajectory | real |
| `prob_cal_safe_success__I_LightGBM_Dense_AF` | Calibrated success-head probability | illustrative |
| `prob_cal_safe_failure__I_LightGBM_Dense_AF` | Calibrated failure-head probability | illustrative |

Probability columns follow the naming rule `prob_{raw|cal}_safe_{success|failure}__{predictor}`,
where `cal` denotes Platt-calibrated scores. The predictor name and score mode come from the policy
preset in [`configs/policy_presets.yaml`](../configs/policy_presets.yaml).

Only `traj_id`, `label`, `prefix_step_idx`, and the two probability columns are required by the
policy; the rest are carried through for readability.

## Running it

```bash
python -m earlyeval.cli pipeline current-safe-stop --mode smoke

python -m earlyeval.cli policy apply \
  --preset current_safe_stop \
  --predictions examples/smoke_predictions.csv \
  --output-dir outputs/current_safe_stop_smoke
```

Three runs halt on the success head, three on the failure head, and three stay unconfident to
completion, so the summary covers early-success stops, early-failure stops, and undecided runs.

## Regenerating

[`build_fixture.py`](build_fixture.py) re-downloads the public results and rewrites the CSV:

```bash
python examples/build_fixture.py
```

Change `RUNS` in that script to build the fixture from different leaderboard runs. To apply the
policy to real predictions instead, point `--predictions` at your own table with the same columns.
