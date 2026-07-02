# H10 Hold-to-Horizon Strategy Reproduction Note

This note records the H10 strategy validation package generated after the V3
backtest accounting audit. The current evidence is useful, but it does not yet
support a publication-ready claim that H10 Top-1 was selected on 2024 data.

## Strategy definitions

### H10 Top-1 Hold

- Input: saved `predictions.csv` from `scripts/backtest_v3.py`.
- Signal: use only horizon `10` rows.
- Ranking: for each signal date, sort symbols by calibrated `p_up` descending.
- Entry: buy the single top-ranked symbol at the next trading day's open.
- Holding period: 10 trading days.
- Exit: sell at the close on the horizon-end day; force-close at the final
  evaluation date if the horizon extends past the available window.
- Capital: Rs 100,000.
- Positioning: 10 slots (`top_k * horizon`), so each new trade receives about
  10% of initial capital.
- Costs: 0 bps for the headline no-cost research run.

### H10 Top-3 Hold

Same as H10 Top-1, except the top 3 symbols are bought each signal day. This
uses 30 slots, so each trade receives about 3.33% of initial capital.

### Conservative ATR Strategy

This is the strategy behind the old paper summary, not the new H10 hold rule.
It is defined in `scripts/backtest_historical.py` as the `conservative`
profile:

- Capital: Rs 100,000.
- Max concurrent positions: 3.
- Horizons: H5, H7, H10.
- Long threshold: `p_up >= 0.65`.
- Old short threshold: `p_up <= 0.35`.
- Stop: 1.5 ATR.
- Target: 2.5 ATR.
- Entry: next trading day's open.
- Exit: minute-level target/stop, horizon expiry, or backtest-end force close.

The old paper artifact `runs/v3_backtest_latest` reported +24.41%, but that
run included 32 short trades and was generated under the old short-close cash
accounting bug. It should not be used as a final paper result.

## Validation package results

These numbers come from `runs/h10_validation_package/summary_matrix.csv`.
Per-run `summary.json`, `trade_log.csv`, and `equity_curve.csv` files are under
`runs/h10_validation_package/`. All replay rows passed reconciliation,
entry-timing, max-open-position, and frozen-setting checks. The package was
generated with the canonical minute-derived price source used by
`scripts/replay_h10_hold_strategy.py`.

### 2024 selection evidence

Only the available 2024 prediction segment is present locally:
2024-09-05 to 2024-12-31. It does not support selecting H10 Top-1.

| Strategy | Window | Return | Sharpe | Trades | Win rate | Max drawdown |
|---|---|---:|---:|---:|---:|---:|
| H7 Top-3 Hold | 2024-09-05 to 2024-12-31 | -8.09% | -2.06 | 197 | 41.1% | 11.52% |
| H7 Top-1 Hold | 2024-09-05 to 2024-12-31 | -9.37% | -1.73 | 64 | 43.8% | 17.64% |
| H10 Top-3 Hold | 2024-09-05 to 2024-12-31 | -10.21% | -2.93 | 194 | 34.0% | 12.40% |
| H10 Top-1 Hold | 2024-09-05 to 2024-12-31 | -10.81% | -2.57 | 65 | 36.9% | 13.79% |

### Frozen 2025 H10 Top-1 final test

| Cost | Return | Sharpe | Trades | Win rate | Avg trade return | Max drawdown |
|---:|---:|---:|---:|---:|---:|---:|
| 0 bps | +24.63% | 1.33 | 216 | 56.5% | +1.12% | 12.74% |
| 6 bps | +23.46% | 1.27 | 216 | 56.5% | +1.06% | 12.85% |
| 20 bps | +20.73% | 1.14 | 216 | 55.1% | +0.92% | 13.10% |
| 41 bps | +15.64% | 0.89 | 215 | 54.4% | +0.66% | 13.48% |

### 2025 baselines

| Baseline | Return | Sharpe | Trades | Win rate | Max drawdown |
|---|---:|---:|---:|---:|---:|
| Equal-weight buy-and-hold | +5.41% | 0.92 | 45 | 66.7% | 4.44% |
| 20-day momentum Top-1 | +3.19% | 0.27 | 181 | 53.0% | 9.57% |
| Random Top-1, seed 7 | +6.29% | 0.52 | 218 | 52.8% | 11.09% |
| Corrected Conservative ATR, long-only | +10.39% | 0.99 | 141 | 44.0% | 9.85% |
| Corrected Conservative ATR, long-only full-cost | -1.23% | -0.03 | 141 | 41.1% | 11.56% |

### 2026 partial stress window

2026-01-01 to 2026-02-23 is a partial stress/regime window, not a full
out-of-sample year.

| Cost | Return | Sharpe | Trades | Win rate | Max drawdown |
|---:|---:|---:|---:|---:|---:|
| 0 bps | -2.95% | -1.37 | 32 | 37.5% | 7.02% |
| 6 bps | -3.13% | -1.45 | 32 | 37.5% | 7.09% |
| 20 bps | -3.54% | -1.64 | 32 | 34.4% | 7.26% |
| 41 bps | -5.95% | -2.98 | 32 | 31.3% | 7.34% |

H10 Top-1 remains strong on the frozen 2025 test and survives realistic cost
sensitivity. However, the available 2024 selection evidence is negative for all
tested H7/H10 variants and ranks H10 Top-1 last in that selection slice. That
means the current package is not sufficient to claim a clean 2024-selected,
2025-tested publishable strategy without additional pre-2025 evidence or a
different frozen selection rule.

## Reproduction commands

Run from the repository root.

Build the validation package:

```bash
.venv/bin/python scripts/replay_h10_validation_package.py \
  --package \
  --price-source minute \
  --out-dir runs/h10_validation_package
```

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_h10_hold_strategy.py \
  --predictions runs/v3_2025_long_off_0bps/predictions.csv \
  --data-dir data/nifty500 \
  --start 2025-01-01 \
  --end 2025-12-31 \
  --horizon 10 \
  --top-k 1 \
  --out-dir runs/h10_top1_2025_replay
```

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_h10_hold_strategy.py \
  --predictions runs/v3_2025_long_off_0bps/predictions.csv \
  --data-dir data/nifty500 \
  --start 2025-01-01 \
  --end 2025-12-31 \
  --horizon 10 \
  --top-k 3 \
  --out-dir runs/h10_top3_2025_replay
```

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_h10_hold_strategy.py \
  --predictions runs/v3_backtest_2026_oos_full/predictions.csv \
  --data-dir data/nifty500 \
  --start 2026-01-01 \
  --end 2026-02-23 \
  --horizon 10 \
  --top-k 1 \
  --out-dir runs/h10_top1_2026_partial_replay
```

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_h10_hold_strategy.py \
  --predictions runs/v3_backtest_2026_oos_full/predictions.csv \
  --data-dir data/nifty500 \
  --start 2026-01-01 \
  --end 2026-02-23 \
  --horizon 10 \
  --top-k 3 \
  --out-dir runs/h10_top3_2026_partial_replay
```

Each command writes:

- `summary.json`
- `trade_log.csv`
- `equity_curve.csv`

The summary includes `reconciliation_delta`, which should be near zero:

```text
final_equity - initial_capital ~= sum(trade_log.net_pnl)
```

## Conservative ATR comparison commands

The corrected long-only Conservative ATR no-cost artifact currently lives at:

```text
runs/v3_2025_long_off_0bps/
```

The corrected full-cost artifact currently lives at:

```text
runs/v3_2025_long_off_fullcost/
```

To regenerate the no-cost long-only Conservative ATR run from the V3 model:

```bash
PYTHONPATH=src .venv/bin/python scripts/backtest_v3.py \
  --run-dir runs/20260522_181848_v3_nifty100_from_nifty500 \
  --data-dir data/nifty500 \
  --start 2025-01-01 \
  --end 2025-12-31 \
  --profile conservative \
  --brokerage-pct 0.0 \
  --out-dir runs/v3_2025_long_off_0bps_regen
```

To regenerate a full Indian delivery-cost variant:

```bash
PYTHONPATH=src .venv/bin/python scripts/backtest_v3.py \
  --run-dir runs/20260522_181848_v3_nifty100_from_nifty500 \
  --data-dir data/nifty500 \
  --start 2025-01-01 \
  --end 2025-12-31 \
  --profile conservative \
  --full-cost-india-delivery \
  --out-dir runs/v3_2025_long_off_fullcost_regen
```

## Publication-readiness caveats

Before using H10 Top-1 as a Scientific Reports headline strategy:

1. Select the strategy using pre-2025 validation data only, preferably 2024.
2. Freeze H10, top-K, holding period, capital allocation, and costs before
   final 2025 testing.
3. Report no-cost and full-cost variants.
4. Include NIFTY 50, equal-weight, momentum, random top-1, and Conservative
   ATR baselines.
5. Fix or ablate stock-trait leakage before final model retraining.
6. Treat 2026 Q1 as a stressed long-only regime, not as an interchangeable
   full-year out-of-sample result.
