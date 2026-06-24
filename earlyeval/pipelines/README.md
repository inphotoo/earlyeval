# pipelines

`pipelines/` composes lower-level modules into user-facing workflows.

## Current Safe Stop

`current_safe_stop.py` exposes `run_current_safe_stop`:

- `mode=smoke`: uses the bundled smoke prediction example.
- `mode=main`: requires `--predictions` and writes a main-run output folder.
- `mode=full`: reserved for registry-driven full workflows.

```bash
python -m earlyeval.cli pipeline current-safe-stop --mode smoke
```

```bash
python -m earlyeval.cli pipeline current-safe-stop \
  --mode main \
  --predictions /path/to/test_predictions_safe_stop.parquet
```

Pipelines should coordinate `core/`, `policies/`, `models/`, and `reports/`
without hiding new command-line behavior outside `earlyeval/cli.py`.
