# legacy

`legacy/` is only a compatibility layer for old command names. It is not a
separate source-code dependency.

The actual answer-aware training, feature, policy, and posthoc scripts used by
the paper experiments are vendored inside:

```text
earlyeval/vendor/prefix_predict_model_holdout_answer/
```

## Files

- `wrappers.py`: maps historical script names to the corresponding earlyeval CLI
  entry points.

## Example

```bash
python -m earlyeval.cli legacy explain safe_stop_dual_head_retrain.py
```

If an old entry point is not mapped, the command prints the known mappings.

