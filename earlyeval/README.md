# earlyeval

`earlyeval/` is the source package for the EarlyEval experiments. It contains the
training orchestration, feature wrappers, safe-stop policies, reporting helpers,
and vendored baseline code required by the paper experiments.

The code release is intended to be self-contained at the source-code level:
runtime Python modules are included under `earlyeval/` and `earlyeval/vendor/`.
Large data tables, trained feature-engineer pickles, model outputs, and
prediction parquet files are treated as artifacts and are not stored in Git.

Useful entry points:

```bash
python -m earlyeval.cli --help
python -m earlyeval.cli experiment paper-suite --stage smoke
```

