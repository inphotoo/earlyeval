# benchmarks

`benchmarks/` normalizes raw benchmark records into the shared earlyeval
trajectory contract. It does not train models or choose policies.

## Entry Point

`normalize.py` supports `swebench`, `terminalbench`, `toolathlon`, and
`generic` inputs.

```bash
python -m earlyeval.cli data normalize \
  --benchmark terminalbench \
  --input examples/tiny_terminalbench.jsonl \
  --output-dir outputs/tiny_terminalbench_normalized
```

Outputs include normalized trajectories and a quality audit with trajectory,
instance, model, resolved-rate, and empty-message counts.

## Contract

Normalized records should contain at least `benchmark`, `instance_id`,
`traj_id`, `model_id`, `resolved`, `messages`, and optional `patch` metadata.
