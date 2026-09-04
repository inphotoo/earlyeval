# Contributing to EarlyEval

Thanks for your interest in EarlyEval. This repository is the code release behind
[arXiv:2609.02783](https://arxiv.org/abs/2609.02783), so it carries a research obligation the
average project does not: **the code must keep reproducing the numbers in the paper.** The notes
below exist mostly to protect that property.

## Ways to help

- **Bug reports** — anything that fails to run, or runs but disagrees with the paper.
- **Portability fixes** — hard-coded paths, platform assumptions, dependency breakage.
- **New benchmarks** — wiring an additional agentic benchmark into the trajectory contract.
- **Documentation** — clearer setup notes, corrected commands, better examples.

If you are planning something substantial, please open an issue first so we can agree on the shape
of the change before you invest the time.

## Development setup

```bash
git clone https://github.com/inphotoo/earlyeval.git
cd earlyeval

python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements-github.txt
```

Python 3.10 or newer. There is no packaging manifest — run everything with `python -m` from the
repository root.

Verify your environment before changing anything:

```bash
python -m earlyeval.cli pipeline current-safe-stop --mode smoke
python -m earlyeval.cli check preflight --paths-config configs/paths.example.yaml
```

The smoke test runs against a bundled fixture and needs no benchmark data, so it should pass on a
clean checkout. If it does not, that is a bug worth reporting on its own.

Preflight is a different kind of signal: without benchmark data it will report `"ok": false` and a
long list of missing paths, which is expected. Compare its output before and after your change
rather than reading it as pass/fail.

## Conventions

- **Match the surrounding code.** Type hints with `from __future__ import annotations`, f-strings,
  `pathlib` over string paths, and explicit keyword arguments in public functions.
- **Comment sparingly.** Explain a constraint the code cannot express; skip narration of what the
  next line does.
- **Keep the CLI honest.** Commands print JSON to stdout. Heavy stages stay dry-run unless the caller
  passes `--execute`.
- **Determinism matters.** Seeds are fixed at 42. If a change introduces nondeterminism, say so
  explicitly in the pull request.
- **Do not commit artifacts.** Prefix tables, pickles, trained models, prediction files, and
  generated CSV/TeX tables are outputs. `.gitignore` already excludes them; please do not add
  exceptions without a good reason.

## Touching the vendored code

`earlyeval/vendor/prefix_predict_model_holdout_answer/` and
`earlyeval/vendor/architecture_baselines/` are vendored copies of the pipelines used for the paper
runs. They are deliberately frozen. Changes there can silently move published numbers, so a pull
request that modifies them needs to explain what moved and why, and ideally show a before/after on
at least one fold.

## Pull requests

1. Branch from `main`.
2. Keep the change focused — one concern per pull request reviews far better than five.
3. State what you ran to convince yourself it works. "Ran the smoke test" is a fine answer for a
   docs fix; a change to the trainer needs more.
4. Note any change to a published number, however small.

## Reporting bugs

Please include:

- Your OS, Python version, and the output of `pip freeze | grep -Ei 'lightgbm|scikit|pandas|numpy'`
- The exact command you ran
- The full traceback or the surprising output
- Which benchmark and config were involved, if applicable

## Questions

Open an issue with the `question` label. For matters specific to the paper rather than the code,
contact the corresponding author listed in [CITATION.cff](CITATION.cff).
