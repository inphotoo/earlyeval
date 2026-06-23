# final3

`final3/` is the source package for the EarlyEval experiments. It contains the
training orchestration, feature wrappers, safe-stop policies, reporting helpers,
and vendored baseline code required by the paper experiments.

The code release is intended to be self-contained at the source-code level:
runtime Python modules are included under `final3/` and `final3/vendor/`.
Large data tables, trained feature-engineer pickles, model outputs, and
prediction parquet files are treated as artifacts and are not stored in Git.

Useful entry points:

```bash
python -m final3.cli --help
python -m final3.cli experiment rq-final --stage smoke
```

