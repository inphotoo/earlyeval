# configs

`configs/` contains portable configuration used by the earlyeval experiment
entrypoints.

## Files

- `policy_presets.yaml`: safe-stop policy presets, including the current main
  preset.
- `paths.example.yaml`: portable path template. Copy it to `paths.yaml` for
  machine-local overrides.
- `experiment_registry.yaml`: registry of experiment groups and entrypoints.
- `earlyeval.yaml`: paper experiment configuration for SWE-bench Verified,
  TerminalBench, Toolathlon, ablations, robustness runs, and cost audits.

## Examples

```bash
python -m earlyeval.cli policy apply \
  --preset current_safe_stop \
  --predictions examples/smoke_predictions.csv \
  --output-dir outputs/current_safe_stop_smoke
```

```bash
python -m earlyeval.cli check preflight --experiment all
```

Keep `paths.yaml` local. It is intentionally excluded from the public release.
