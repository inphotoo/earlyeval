<!-- Thanks for contributing to EarlyEval. Keep it focused; one concern per PR reviews best. -->

## Summary

<!-- What changes, and why. -->

## Type of change

- [ ] Bug fix
- [ ] New feature or benchmark support
- [ ] Documentation
- [ ] Refactor / portability
- [ ] Vendored pipeline change (see below)

## How was this verified?

<!-- State what you actually ran. "Ran the smoke test" is fine for a docs fix;
     a change to the trainer needs more. -->

- [ ] `python -m earlyeval.cli pipeline current-safe-stop --mode smoke`
- [ ] `python -m earlyeval.cli check preflight --paths-config configs/paths.example.yaml`
- [ ] Other (describe):

## Impact on published numbers

- [ ] No effect on any number reported in the paper
- [ ] Changes a reported number — details below

<!-- If a number moves, say which one, by how much, and on what fold. -->

## Checklist

- [ ] Change is focused on a single concern
- [ ] No artifacts committed (prefix tables, pickles, models, predictions, generated CSV/TeX)
- [ ] Determinism preserved, or nondeterminism explicitly called out
- [ ] Docs updated if behavior or commands changed
- [ ] Vendored code under `earlyeval/vendor/` is untouched, or the change is justified above
