# Contributing

Thanks for looking at the project. The rules below keep every result honest and reproducible.

## Set up
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make test            # the full suite
make lint            # ruff
```

## Ground rules for any experiment
1. Use development films only (released before 2023). The frozen test set is for the single final report and is never used to choose anything.
2. Validate chronologically (`hitflop.data.outer_folds`). Never shuffle.
3. Use information available before release only. New features go through the leakage checks in `tests/test_leakage.py`.
4. A result is called better or worse only when the paired-bootstrap 95% interval excludes zero and the Benjamini-Hochberg q-value is below 0.05 (`hitflop.protocol`). Otherwise write "no clear difference".
5. Log the run with `protocol.log_run` and keep the config, seed and numbers.

## Pull requests
Run `make test lint` first. Describe what changed and what evidence supports it.
