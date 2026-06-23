# legacy

`legacy/` is only a compatibility layer for old command names. It is not a
separate source-code dependency.

The actual answer-aware training, feature, policy, and posthoc scripts used by
the final experiments are vendored inside:

```text
final3/vendor/prefix_predict_model_holdout_answer/
```

## Files

- `wrappers.py`: maps historical script names to the corresponding final3 CLI
  entry points.

## Example

```bash
python -m final3.cli legacy explain safe_stop_dual_head_retrain.py
```

If an old entry point is not mapped, the command prints the known mappings.

