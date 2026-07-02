# V3 Portfolio Backtest — 2025 Out-of-Sample

> **Model:** `runs/20260522_181848_v3_nifty100_from_nifty500`  
> **Period:** 2025-01-01 → 2025-12-31  
> **Source:** Local run `runs/v3_backtest_latest/` (gitignored — this file is the committed summary)

## Portfolio results

| Profile | Return | Sharpe | Max DD | Trades | Win Rate | Profit Factor |
|---------|-------:|-------:|-------:|-------:|---------:|--------------:|
| **Conservative** | **+24.41%** | **1.70** | **7.0%** | 128 | 46.9% | 1.29 |
| Moderate | +4.87% | 0.53 | 7.8% | 266 | 44.4% | 0.95 |
| Aggressive | -7.54% | -0.69 | 9.4% | 1070 | 46.3% | 0.93 |
| Intraday | -10.19% | -0.73 | 13.7% | 777 | 44.1% | 0.86 |
| Swing | +4.32% | 0.46 | 11.1% | 194 | 42.8% | 0.81 |
| MaxDiversified | -1.71% | -0.08 | 9.1% | 1521 | 46.5% | 0.90 |

## Reproduce locally

```bash
PYTHONPATH=src python scripts/backtest_v3.py \
  --run-dir runs/20260522_181848_v3_nifty100_from_nifty500 \
  --start 2025-01-01 --end 2025-12-31
```

Output: `runs/v3_backtest_latest/backtest_report.md`
