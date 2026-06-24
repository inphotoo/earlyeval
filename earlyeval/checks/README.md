# earlyeval Checks

`checks/` contains checks that should run before expensive experiments.

The main entry is:

```bash
python -m earlyeval.cli check preflight --experiment all
```

Outputs are written to `paper/checks/preflight/` so the paper bundle records
what data, code, and dependencies were present before a run.

