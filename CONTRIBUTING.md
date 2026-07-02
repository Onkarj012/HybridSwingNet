# Contributing to StockXpert

## Principles

- Treat this repository as a research codebase, not a metric leaderboard.
- Prefer reproducibility over clever shortcuts.
- Keep generated artifacts out of git unless they are intentionally curated examples.
- Document negative results, regressions, and caveats as clearly as positive outcomes.

## Setup

```bash
uv sync
uv run pytest
```

If you prefer `pip`, install from `requirements.txt` and then `pip install -e .`.

## Before Opening a PR

- Run the relevant tests for your change.
- Update docs or configs when behavior changes.
- Keep experiment-specific outputs in `runs/` locally, not in git.
- Avoid committing caches, logs, notebooks outputs, or private credentials.

## Code Guidelines

- Put reusable logic in `src/stockxpert/` instead of large scripts whenever possible.
- Keep scripts focused on orchestration and reporting.
- Preserve backward compatibility when moving modules that existing experiments still import.
- Add tests for bug fixes and contract changes.

## Research Reporting

- State the dataset window, universe, and horizon when reporting results.
- Separate training metrics from holdout or backtest metrics.
- Do not present illustrative run outputs as production-ready performance claims.

## Pull Request Notes

Helpful PRs usually include:

- a short problem statement
- the exact files or pipeline stages changed
- validation steps that were run
- remaining risks or open questions
