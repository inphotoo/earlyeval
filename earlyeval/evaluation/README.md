# evaluation

`evaluation/` contains reusable metrics and audit helpers. It supports policy
replay, pipelines, and reports without owning any training entrypoint.

## Files

- `metrics.py`: small metric utilities such as `safe_div`.
- `leakage.py`: split and model-holdout leakage checks for DataFrames.

## Example

```python
from earlyeval.evaluation.leakage import audit_split_frame

audit = audit_split_frame(frame)
if not audit["ok"]:
    print(audit)
```

Place policy-changing logic in `policies/`, not here.
