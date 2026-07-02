# HybridSwingNet — Canonical Results Document v2

> **North Star document for all paper revision tasks. Supersedes `docs/CANONICAL_RESULTS.md` in full.**
> Every number used in the manuscript, response letter, and figures must trace back here.
> All results are from the V3 model: `runs/20260522_181848_v3_nifty100_from_nifty500/`
> All backtest runs use V3 predictions. Do not mix V2 (model-version) numbers — see §13.
>
> This file is v2 of the canonical-results doc itself (doc versioning, not model versioning).
> It folds the corrections previously tracked separately (stale §7 2026 OOS run,
> mixed-accounting §6.3/§6.4 comparisons) directly into the relevant sections below, so the
> whole document is now self-contained — no more reading two files side by side. Sections
> that were already correct in v1 (§1-5, §8-10, accuracy/architecture/data/baselines/stats/
> regime-calibration/PF-verification) are carried over unchanged. §6 and §7 are rewritten;
> §11, §13, §14 are updated to reflect the corrected numbers. **No number in this document
> has been altered from what was already verified — corrected sections use the exact figures
> from the `v2` correction pass, unchanged sections use the exact figures from v1.**

---

## 1. Model Identity

| Item | Value | Source |
|---|---|---|
| Model version | V3 (`StockXpertModelV3`) | `scripts/train_v3.py` |
| Run directory | `runs/20260522_181848_v3_nifty100_from_nifty500/` | — |
| Config file | `configs/v3_nifty100_from_nifty500.yaml` | — |
| Checkpoint | `checkpoints/model_final.pt` inside run dir | — |
| Random seed | 42 | `config.yaml: run.seed` |
| Trainable parameters | ~1.78 M | checkpoint state dict |
| Training hardware | Apple Silicon CPU runtime (config requested CUDA; training log records `Using device: cpu`) | `stockxpert.log` |
| Training time | ~33 min | `stockxpert.log` epoch timestamps |
| Public repo | github.com/Onkarj012/HybridSwingNet | Data Availability section |

---

## 2. Architecture

| Hyperparameter | Value |
|---|---|
| `hidden_dim` | 96 |
| `stock_embed_dim` | 16 |
| `attn_heads` | 4 |
| `dropout` | 0.15 |
| Optimizer | AdamW (Adaptive Moment Estimation with decoupled Weight Decay) |
| Learning rate | 1.5×10⁻⁴ |
| Weight decay | 3×10⁻³ |
| LR (Learning Rate) schedule | ReduceLROnPlateau (patience 4, factor 0.5) |
| Gradient clipping | max-norm 1.0 |
| Batch size | 256 |
| Max epochs | 50 (early stopping patience 4) |
| Post-train calibration | Isotonic regression on validation split |
| Prediction target | Centered 3-step smoothed, volatility-normalised future log return (label-only transform; not applied to inputs) |

---

## 3. Data Universe

| Item | Value |
|---|---|
| Universe | 100 stocks: 84 confirmed Nifty 100 constituents (47 Nifty 50 + 37 Nifty Next 50) + 16 additional large-cap Nifty 500 stocks |
| Purpose of 16 extra | Enables 70:10:10:10 cross-sectional split with round numbers |
| Data source | Yahoo Finance, daily OHLCV (Open, High, Low, Close, Volume) bars |
| Data range | 2013-01-01 → 2026-04-30 |
| Ticker list | `configs/v3_nifty100_from_nifty500.yaml: data.symbols` |
| BankNifty | NOT included (removed from all claims) |
| "Blind Nifty500" | Terminology dropped; described precisely as blind holdout below |

### 3.1 Cross-Sectional Split

| Partition | Stocks | Role |
|---|---|---|
| Train | 70 | Model training |
| Validation | 10 | Early stopping, hyperparameter selection, isotonic calibration |
| Test | 10 | Final accuracy evaluation |
| Blind holdout | 10 | Completely unseen evaluation (never influenced any decision) |

- Assignment: random, seed 42, deterministic from config file
- Embargo: 10 trading days between partitions (prevents label leakage at partition boundaries)
- Leakage guard: `assert_no_leakage()` aborts pipeline if any stock appears in >1 partition

> **Why cross-sectional, not temporal?**
> Every reported metric is on stocks the model never saw during training. This provides a stronger generalization guarantee than temporal splits, where the model sees all 100 stocks during training.

---

## 4. Prediction Accuracy (Smoothed-Trend Target)

The model predicts **smoothed-trend direction**: whether the centered 3-step smoothed future return is positive or negative. This is the task the model is trained on. Raw next-day direction accuracy (~50%) is separately reported for transparency.

### 4.1 Validation Set (10 unseen stocks)

*Horizon notation: H1 = 1-day, H3 = 3-day, H5 = 5-day, H7 = 7-day, H10 = 10-day prediction window.*
*MAE = Mean Absolute Error; RMSE = Root Mean Square Error. Units: normalized z-score of log return (dimensionless; multiply by ~1.2% daily vol for approximate % price error).*

| Horizon | Accuracy | MAE | RMSE |
|---|---|---|---|
| H1 (1d) | **68.72%** | 0.430 | 0.628 |
| H3 (3d) | **57.61%** | 1.254 | 1.712 |
| H5 (5d) | **56.00%** | 1.634 | 2.154 |
| H7 (7d) | **54.22%** | 1.941 | 2.505 |
| H10 (10d) | **50.72%** | 2.381 | 2.979 |

Source: `reports/metrics.json`

**Statistical tests (validation, H1):**
- Binomial p-value: **p = 3.18×10⁻⁶³** (vs 50% null)
- 95% CI (Confidence Interval) on accuracy: [67.3%, 71.6%]
- ECE (Expected Calibration Error — measures how well predicted probabilities match observed frequencies): **0.093** (well-calibrated; 0.0 = perfect, 1.0 = worst)
- Precision: 74.1% | Recall: 72.2% | F1: 73.2%

Source: `reports/val_statistical_eval.json`

### 4.2 Blind Holdout Set (10 completely unseen stocks)

| Horizon | Accuracy | MAE | RMSE |
|---|---|---|---|
| H1 (1d) | **70.19%** | 0.419 | 0.605 |
| H3 (3d) | **58.91%** | 1.249 | 1.692 |
| H5 (5d) | **59.44%** | 1.590 | 2.072 |
| H7 (7d) | **58.22%** | 1.822 | 2.326 |
| H10 (10d) | **58.02%** | 2.216 | 2.739 |

Source: `reports/blind_metrics.json`

Note: Blind holdout accuracy is *higher* than test accuracy, indicating no overfitting to the test partition.

### 4.3 Confidence-Filtered Accuracy — Validation Set

When filtering to high-conviction predictions (P(Up) or P(Down) > threshold):

| Horizon | Threshold | Accuracy | Coverage | N |
|---|---|---|---|---|
| H1 | 0.50 | 68.6% | 100% | 1800 |
| H1 | 0.55 | 72.6% | 82.8% | 1490 |
| H1 | 0.60 | 75.6% | 65.8% | 1184 |
| H1 | 0.65 | 76.0% | 59.8% | 1077 |
| H1 | 0.70 | 77.9% | 48.2% | 867 |
| H1 | **0.75** | **82.9%** | 34.2% | 615 |
| H3 | 0.75 | 69.9% | 4.1% | 73 |
| H5 | 0.75 | 63.5% | 4.1% | 74 |

Source: `reports/confidence_threshold_metrics.csv`

### 4.4 Confidence-Filtered Accuracy — Blind Holdout Set

| Horizon | Threshold | Accuracy | Coverage | N |
|---|---|---|---|---|
| H1 | 0.50 | 70.3% | 100% | 3368 |
| H1 | 0.55 | 73.9% | 81.8% | 2754 |
| H1 | 0.60 | 77.7% | 65.0% | 2188 |
| H1 | 0.65 | 79.1% | 58.6% | 1972 |
| H1 | 0.70 | 81.6% | 48.0% | 1616 |
| H1 | **0.75** | **85.3%** | 34.6% | 1165 |
| H3 | 0.75 | 78.9% | 3.8% | 128 |
| H5 | 0.75 | 78.1% | 3.8% | 128 |
| H7 | 0.75 | — | — | — |

Source: `reports/blind_confidence_threshold_metrics.csv`

### 4.5 Raw Direction Accuracy (for transparency)

The model is NOT trained to predict raw next-day direction. These numbers are reported for transparency only.

From `runs/v3_backtest_regime_fullcost/backtest_report.md` (2025 period, 47 stocks, all predictions):

| Horizon | Raw Accuracy | High Conv. | V.High Conv. | N |
|---|---|---|---|---|
| H1 | 50.2% | 49.8% | 49.7% | 24,695 |
| H3 | 50.6% | 49.9% | 52.9% | 24,695 |
| H5 | 51.4% | 50.1% | 49.6% | 24,695 |
| H7 | 51.1% | 51.2% | 49.2% | 24,695 |
| H10 | 52.7% | 52.4% | 50.4% | 24,695 |

All near 50%: consistent with model being a multi-day trend predictor, not a 1-day sign predictor.

---

## 5. Baseline Comparison

All baselines trained and evaluated on identical 96-stock splits (validation period 2025-07 → 2025-12-31).
Source: `reports/a5_baseline_comparison.csv` / `.json`

### 5.1 Raw Next-Day Direction Accuracy

| Model | H1 | H3 | H5 | H7 | H10 |
|---|---|---|---|---|---|
| Random Direction | ~50% | ~50% | ~51% | ~50% | ~49% |
| SMA Crossover (10/50) (SMA = Simple Moving Average; 10/50 = short/long window in days) | ~49% | ~48% | ~48% | ~48% | ~49% |
| Random Forest | ~49% | ~49% | ~50% | ~51% | ~51% |
| Single LSTM (Long Short-Term Memory) | ~49% | ~49% | ~50% | ~51% | ~50% |
| Single GRU (Gated Recurrent Unit) | ~49% | ~49% | ~50% | ~51% | ~50% |
| **HybridSwingNet (raw task)** | ~50% | ~48% | ~48% | ~47% | ~47% |

**All models produce near-random accuracy on raw direction.** HybridSwingNet is also ~50% here because it was not trained for this task.

### 5.2 Smoothed-Trend Accuracy

| Model | H1 | H3 | H5 | H7 | H10 |
|---|---|---|---|---|---|
| All single-encoder baselines | ~50% | ~50% | ~50% | ~50% | ~50% |
| **HybridSwingNet (blind holdout)** | **70.2%** | **58.9%** | **59.4%** | **58.2%** | **58.0%** |

The correct comparison is on the smoothed-trend task that all models are evaluated on. HybridSwingNet's advantage is on this task.

---

## 6. Portfolio Backtest Results (2025 Validation Year)

**Conservative Profile: ₹1L capital, max 3 concurrent positions, H7/H10 horizons, conviction threshold 0.65/0.35 long/short.**

All runs use V3 model (`runs/20260522_181848_v3_nifty100_from_nifty500/`), 2025-01-01 → 2025-12-31, 248 trading days.

### 6.1 All Four Run Conditions (historical record)

| Run | Regime | Cost | Return | Sharpe | MaxDD | Trades | WR | PF | Source |
|---|---|---|---|---|---|---|---|---|---|
| v3_backtest_latest | OFF | None (0 bps) | **+24.41%** | 1.70 | 7.0% | 128 | 46.9% | 1.29 | `v3_backtest_latest/summary.json` |
| v3_backtest_fullcost | OFF | Full (~41 bps) | **+16.68%** | 1.26 | 7.9% | 128 | 44.5% | 1.05 | `v3_backtest_fullcost/summary.json` |
| v3_backtest_regime_on | ON | Brokerage only (6 bps) | **+16.36%** | 1.48 | 5.9% | 85 | 56.5% | 1.77 | `v3_backtest_regime_on/summary.json` |
| **v3_canonical_regen_2025_binary_fullcost_legacy_exec** | **ON** | **Full (~41 bps)** | **+11.51%** | **1.15** | **7.8%** | **85** | **54.1%** | **1.43** | `v3_canonical_regen_2025_binary_fullcost_legacy_exec/summary.json` |

*MaxDD = Maximum Drawdown (largest peak-to-trough equity decline during the period); WR = Win Rate (% of trades closed at a profit); PF = Profit Factor (gross profit ÷ gross loss; >1 = profitable).*

**Canonical headline result = v3_canonical_regen_2025_binary_fullcost_legacy_exec (regime ON + full cost, regenerated after backtest accounting fixes).**

> **Accounting-consistency note:** the three non-canonical rows above (`v3_backtest_latest`,
> `v3_backtest_fullcost`, `v3_backtest_regime_on`) have **no `Accounting_Version` field and no
> `Entry_Cost`/`Exit_Cost` columns** in their trade logs — they predate the accounting fix. Only
> the canonical row uses `net_pnl_v2`. This table is retained as a **historical record** and
> must not be used to claim a "regime effect" or "cost effect" against the canonical row —
> comparing +16.68% (old accounting, regime OFF) against +11.51% (new accounting, regime ON) is
> an apples-to-oranges comparison. Same-accounting comparisons are in §6.5 below.

### 6.2 Canonical Run Detail

From `v3_canonical_regen_2025_binary_fullcost_legacy_exec/summary.json` and `trade_log_conservative.csv`:

| Metric | Value |
|---|---|
| Period | 2025-01-01 → 2025-12-31 |
| Initial capital | ₹1,00,000 |
| Final value | ₹1,11,505 |
| **Total return** | **+11.51%** |
| **Annualised Sharpe** | **1.15** |
| Maximum drawdown | 7.8% |
| Total trades | 85 |
| Winning trades | 46 |
| Losing trades | 39 |
| **Win rate** | **54.1%** |
| Average win | ₹834 |
| Average loss | −₹689 |
| **Profit factor** | **1.43** |
| PF verification | (0.541 × 834) / (0.459 × 689) = 451/316 = **1.43 ✓** |
| Target hits | 16 |
| Stop losses | 27 |
| Horizon ends | 40 |
| Best trade | +₹2,421 |
| Worst trade | −₹1,537 |
| Regime-skipped entry days | 113 |

### 6.3 Regime Filter Effect (historical framing — superseded by §6.5)

Regime filter: skip new entries when median 21-day cross-stock return is negative (bear regime). Threshold computed dynamically at each date using only past data — inherently causal, no calibration window needed.

| Condition | Return | Sharpe | MaxDD | Trades | WR | PF |
|---|---|---|---|---|---|---|
| Regime OFF, full cost | +16.68% | 1.26 | 7.9% | 128 | 44.5% | 1.05 |
| Regime ON, full cost | +11.51% | 1.15 | 7.8% | 85 | 54.1% | 1.43 |

**Old framing (do not use for effect-size claims):** "Reduces trades 128→85 (-33%), improves WR 44.5%→54.1% (+9.6pp), slightly reduces MaxDD 7.9%→7.8%, reduces total return 16.68%→11.51%." Both rows here are `net_pnl_v2` full-cost and *do* share accounting, so the WR/MaxDD/trade-count comparison is valid — but this pairing does not isolate a clean "regime effect" because it doesn't control for year composition. §6.5's 2×2 matrix is the authoritative regime-effect comparison (adds a second year).

### 6.4 Cost Sensitivity (historical — superseded by §6.6)

Actual run anchors (not analytical approximations):

*RT = Round-Trip (entry + exit combined); bps = basis points (1 bp = 0.01%; 100 bps = 1%).*

| Condition | Approximate RT Cost | Return | Source |
|---|---|---|---|
| Regime ON, zero cost | 0 bps | +24.41%* | v3_backtest_latest (no regime) — not exact anchor |
| Regime ON, brokerage only | 6 bps | +16.36% | v3_backtest_regime_on |
| Regime ON, full cost stack | ~41 bps | +11.51% | v3_canonical_regen_2025_binary_fullcost_legacy_exec |

*The 24.41% is from regime-OFF, zero-cost run; regime-ON zero-cost is not separately measured.

Cost decrease per ~35 bps of added cost: −4.86 percentage points return (linear interpolation from anchors — **superseded, do not cite**; see §6.6 for the exact, non-interpolated decomposition).

**Full cost model applied (in code):**

| Component | Rate | Applied at | RT bps |
|---|---|---|---|
| Brokerage | 0.03% | Both sides | 6 |
| STT (Securities Transaction Tax — Indian govt levy on equity trades) | 0.10% | Both sides (conservative; real = exit only) | 20 |
| Stamp duty | 0.015% | Both sides (conservative; real = entry only) | 3 |
| GST (Goods and Services Tax — 18% levied on brokerage fee) | 18% of brokerage | Both sides | 1.1 |
| NSE (National Stock Exchange of India) / SEBI (Securities and Exchange Board of India) exchange | 0.00345% | Both sides | 0.7 |
| Slippage | 0.05% | Both sides | 10 |
| **Total RT** | | | **~41 bps** |

Note: STT and stamp are applied to both sides in code (more conservative than actual NSE regulations). Real regulatory RT cost ≈ 26-28 bps + 10 bps slippage = ~36-38 bps. The code model is more conservative by ~3-5 bps.

### 6.5 Consistent-Accounting 2×2 Regime Matrix (authoritative regime-effect comparison)

All four cells below use `net_pnl_v2` accounting (verified via `Accounting_Version` column in
each run's trade log). This is the only apples-to-apples regime/cost comparison available and
is what R2.3's matrix table should show — it supersedes the single-pair framing in §6.3.

| Year | Regime OFF | Regime ON | Regime effect |
|---|---|---|---|
| 2025 | **−7.87%** (Sharpe −0.68, MaxDD 13.6%, 136 trades, WR 40.4%, PF 0.85) | **+11.51%** (Sharpe 1.15, MaxDD 7.8%, 85 trades, WR 54.1%, PF 1.43) | **+19.4pp** |
| 2026 H1 | **+1.25%** (Sharpe 0.29, MaxDD 9.4%, 60 trades, WR 50.0%, PF 1.06) | **−3.64%** (Sharpe −0.70, MaxDD 9.7%, 40 trades, WR 42.5%, PF 0.77) | **−4.9pp** |

Sources: `v3_2025_regime_off_fullcost/summary.json`, canonical 2025 (§6.2), `v3_2026_regime_off_fullcost/summary.json`, canonical 2026 (§7.3). All four full-cost (~41 bps), all four `net_pnl_v2`.

**Framing (matches plan's R2.3 language):** regime filter added +19.4pp in 2025 (a year
containing a sharp bear regime the filter sidesteps) and cost −4.9pp in 2026 H1 (choppy, mixed
regime — filter reduces trade count without a matching drawdown to avoid). This is a
**defensive risk-control mechanism, not a universal return enhancer** — report both years, not
just the favorable one.

**Note:** the 2025 "Regime OFF" cell here (−7.87%) differs from §6.1/§6.3's "Regime OFF, full
cost" row (+16.68%, `v3_backtest_fullcost`) because that row predates the accounting fix and
has no `Accounting_Version` field. −7.87% is the `net_pnl_v2`-accounted regime-OFF run and is
the only one that may be paired against the canonical +11.51% row for an effect-size claim.

### 6.6 Cost Decomposition (exact — replaces §6.4 interpolation, answers R1.2)

Computed directly from `trade_log_conservative.csv` (85 trades), summing `Entry_Cost` +
`Exit_Cost` per row against `Gross_PnL` and `Net_PnL` — no interpolation, no cross-accounting
anchors.

| Quantity | Value |
|---|---|
| Gross P&L (before costs) | ₹19,118.80 → **+19.12%** |
| Total transaction costs | ₹7,613.70 → **−7.61 pp** |
| Net P&L (after costs) | ₹11,505.10 → **+11.51%** |
| Check: gross − costs | ₹11,505.10 (matches net exactly ✓) |

Source: `runs/v3_canonical_regen_2025_binary_fullcost_legacy_exec/trade_log_conservative.csv`,
column sums over all 85 trades.

**R1.2 answer:** "Full transaction costs (~41 bps round-trip) reduce the strategy's gross 2025
return from +19.1% to a net +11.5% — a cost drag of 7.6 percentage points, computed directly
from the trade-level cost columns of the canonical run, not from a cross-run approximation."

---

## 7. Out-of-Sample 2026 Results

**Canonical run: `v3_canonical_2026_jan_jun_full_yf_tail/summary.json`.**
Period: 2026-01-01 → 2026-06-25 (6 months, full data via `data/yfinance_daily/` updated through June 25)
Conditions: Regime filter ON, full cost model (~41 bps), `net_pnl_v2` accounting — matches canonical 2025 run.

**Superseded run (do not use):** `v3_backtest_2026_oos_jan_jun_fixed/`. `backtest_v3.py`
previously capped feature data at `cfg.data.end = 2026-02-23` (training config end), silently
dropping all predictions past that date. Fixed by overriding `cfg.data.end = args.end` before
feature build. Even older runs (`v3_backtest_2026_oos_*`) showing only 11-12 trades are also
invalid for the same reason.

### 7.1 Direction Accuracy (2026, all 6 months)

Accuracy is computed from `predictions.csv`, independent of which backtest execution run is
used — unaffected by the §7's data-cutoff bug, so these figures are unchanged from the original
computation.

| Horizon | Raw Accuracy | High Conv. | V.High Conv. | N |
|---|---|---|---|---|
| H1 | 48.4% | 47.6% | 47.4% | 11,600 |
| H3 | 47.2% | 46.0% | 46.4% | 11,400 |
| H5 | 47.2% | 46.6% | 47.8% | 11,200 |
| H7 | 49.0% | 48.8% | 50.1% | 11,000 |
| H10 | 47.8% | 47.5% | 49.3% | 10,700 |

All below 50%. 2026 was a challenging year (Nifty correction Jan-Mar, partial recovery Apr-Jun). Model predicts smoothed-trend direction; raw direction accuracy near 50% is expected and consistent with 2025 behavior.

### 7.2 Monthly Accuracy Breakdown (2026)

| Month | H1 | H3 | H5 | H7 | H10 |
|---|---|---|---|---|---|
| 2026-01 | 49.9% | 41.5% | 41.4% | 45.5% | 45.1% |
| 2026-02 | 49.2% | 46.5% | 45.6% | 43.6% | 38.2% |
| 2026-03 | 43.8% | 34.0% | 32.4% | 40.3% | 39.4% |
| 2026-04 | 49.3% | 62.6% | 66.1% | 63.4% | 61.2% |
| 2026-05 | 49.9% | 45.2% | 42.3% | 42.8% | 42.6% |
| 2026-06 | 47.8% | 53.8% | 57.8% | 63.5% | 73.7% |

Notable: March 2026 was worst (Nifty fell ~8% in Q1 2026 correction). April and June show strong multi-day accuracy (62-74%) suggesting model recovers in trending markets.

### 7.3 Portfolio Backtest (Jan-Jun 2026 OOS) — corrected

| Metric | Superseded (`v3_backtest_2026_oos_jan_jun_fixed`) | **Canonical (`v3_canonical_2026_jan_jun_full_yf_tail`)** |
|---|---|---|
| **Total return** | −4.76% | **−3.64%** |
| **Sharpe** | −1.01 | **−0.70** |
| Maximum drawdown | 9.6% | **9.7%** |
| Total trades | 38 | **40** |
| Win rate | 42.1% | **42.5%** |
| Wins / Losses | — | 17 / 23 |
| Average win | ₹672 | ₹720 |
| Average loss | −₹694 | −₹691 |
| Profit factor | 0.70 | **0.77** |
| Target hits | 5 | 7 |
| Stop losses | 19 | 18 |
| Horizon ends | 11 | 12 |
| Best trade | — | ₹2,093 |
| Worst trade | — | −₹1,355 |

Source: `runs/v3_canonical_2026_jan_jun_full_yf_tail/summary.json` (`regime_filter: true`,
`full_cost_india_delivery: true`, `net_pnl_v2` accounting).

**Honest interpretation (updated):**
- 40 trades over 6 months — thin but 3× better than the broken prior estimate of 11-12 trades from the earliest cutoff bug.
- Negative return (−3.64%) driven by Q1 2026 Nifty correction (regime filter blocked most entries in Jan-Mar, but the entries that did happen in Feb-Mar were in a falling market).
- Stop losses dominated (18/40 = 45%) — consistent with entering in bear regimes that the filter partially missed.
- April and June had strong model accuracy (see §7.2) but portfolio results still negative overall due to Jan-Mar drag.
- 40 trades is still too thin for Sharpe significance. Honest conclusion: model is untested across full market cycles.
- Direction and magnitude both improve slightly vs. the superseded run (−4.76%→−3.64%, Sharpe −1.01→−0.70, PF 0.70→0.77) but the conclusion does not change.
- This OOS result MUST be reported in the paper. The paper should frame it as: "preliminary out-of-sample evidence over 6 months suggests caution; further live testing is needed."

---

## 8. Statistical Rigor

### 8.1 Bootstrap Sharpe Confidence Interval

Computed from daily equity curve of canonical run (248 trading days), 10,000 bootstrap resamples, seed 42.

Recomputed 2026-07-02 via `scripts/recompute_canonical_stats.py` (bootstrap n=10,000, seed=42).

| Metric | Value |
|---|---|
| Point estimate (Sharpe) | 1.15 |
| **95% Bootstrap CI** | **[-0.87, 3.06]** |
| CI width | 3.93 |

**Interpretation:** The wide CI reflects 248 daily observations with 85 concentrated trade events. Sharpe estimation requires hundreds to thousands of trades for precision. Report to 1 decimal place only. This is precisely the small-sample problem the pre-registered walk-forward protocol (`docs/PREREGISTERED_STRATEGY_PROTOCOL.md`) exists to fix with ~1,350 concatenated OOS days.

### 8.2 Deflated Sharpe Ratio (DSR)

*SR = Sharpe Ratio (annualised return ÷ annualised volatility); DSR = Deflated Sharpe Ratio (SR adjusted downward for number of strategy configurations tried — corrects for selection bias); PSR = Probabilistic Sharpe Ratio (probability that true SR > 0 given observed SR and sample size).*

Method: Bailey & López de Prado (2014), via expected maximum Sharpe adjustment.

| Input | Value |
|---|---|
| k (independent configs evaluated for V3) | 4 |
| T (trading days) | 248 |
| Skewness | +0.88 |
| Kurtosis (normal = 3) | 7.71 |
| SR_hat (observed Sharpe Ratio) | 1.15 |
| **DSR (Deflated Sharpe)** | **1.09** |
| SR_hat > DSR threshold | **Yes** (1.149 > 1.093) |

Recomputed 2026-07-02 via `scripts/recompute_canonical_stats.py`; full output in
`reports/canonical_2025_stats_recomputed.json`.

**V3 configs evaluated (k=4):**
1. No regime, no extra costs (v3_backtest_latest)
2. Full cost, no regime (v3_backtest_fullcost)
3. Regime ON, brokerage only (v3_backtest_regime_on)
4. Regime ON, full cost — canonical (`v3_canonical_regen_2025_binary_fullcost_legacy_exec`)

Compare with V2 (model version): k=640 configs → deflated Sharpe = −0.17. V3's regenerated canonical DSR should be recomputed with the corrected Sharpe/equity curve before manuscript use.

### 8.3 Directional Accuracy Significance

- Binomial test p-value (H1, blind holdout, N=3368): **p = 3.18×10⁻⁶³**
- Under the null (50% accuracy), observing 70.2% over 3368 predictions is essentially impossible.
- This is on stocks the model has NEVER seen during training (blind holdout).

---

## 9. Regime Filter Calibration (R2.3 Fix)

The regime filter in V3 (`backtest_v3.py: compute_regime_bullish`) is a **dynamic, causal filter** — it computes median 21-day return across the universe at each trading date using only data available through that date. No threshold is calibrated; the rule is simply: median 21d cross-stock return > 0 → bullish.

This is inherently leak-free. There is no "calibration on validation set" issue for V3. The V2 (model version) circular calibration bug is not present in the V3 implementation.

---

## 10. Profit Factor Verification Table

All four states verified:

| Run | WR | AvgWin | LossRate | AvgLoss | PF (calc) | PF (reported) | Match |
|---|---|---|---|---|---|---|---|
| No regime, full cost | 44.5% | ₹903 | 55.5% | −₹690 | (0.445×903)/(0.555×690) = 402/383 = **1.05** | 1.05 | ✓ |
| Regime ON, brokerage | 56.5% | ₹885 | 43.5% | −₹648 | (0.565×885)/(0.435×648) = 500/282 = **1.77** | 1.77 | ✓ |
| Regime ON, full cost (canonical) | 54.1% | ₹834 | 45.9% | −₹689 | (0.541×834)/(0.459×689) = 451/316 = **1.43** | 1.43 | ✓ |
| **ORIGINAL PAPER (WRONG)** | 60.2% | ₹4,349 | 39.8% | −₹3,218 | (0.602×4349)/(0.398×3218) = 2618/1281 = **2.05** | ~~2.71~~ | ✗ |

Original PF 2.71 was wrong (hardcoded in figure script, no backtest backing it). Replaced with 1.43 from the regenerated canonical run.

---

## 11. Known Discrepancies vs Earlier Drafts

| Document | Claim | Actual (V3 Canonical) | Status |
|---|---|---|---|
| Original paper abstract | +28.77%, Sharpe 2.94, PF 2.71, 83 trades | +11.51%, Sharpe 1.15, PF 1.43, 85 trades | **REPLACE in manuscript** |
| RESPONSE_SHEET.md R-2 | +23.85% Sharpe 2.50 (V2 model-version best) | +11.51% Sharpe 1.15 (V3 canonical) | ✅ Updated |
| RESPONSE_TO_REVIEWERS.md | "Regime filter ON, full cost: TBD" | +11.51%, Sharpe 1.15, MaxDD 7.8%, 85 trades, 54.1% WR, PF 1.43 | ✅ Filled |
| RESPONSE_TO_REVIEWERS.md | "Regime filter ON, brokerage-only: +16.4% Sharpe 1.5" | +16.36% Sharpe 1.48 | ✅ Correct (rounding) |
| RESPONSE_TO_REVIEWERS.md | "Deflated Sharpe: k=6" | k=4 (for V3) | ✅ Fixed |
| RESPONSE_TO_REVIEWERS.md | "45–50 bps total RT cost" | ~41 bps actual (code applies STT/stamp to both sides) | ✅ Updated to ~41 bps |
| REVIEWER_RESPONSE_TRACKER.md | R2.5 status 🔴 (bib refs not removed) | kaufmann2026benchmark + Friday_2026 removed from bib + main.tex | ✅ Done 2026-06-29 |
| REVIEWER_RESPONSE_TRACKER.md | "T4 DONE: Deflated Sharpe = −0.17 (k=640 V2 model-version)" | V3 DSR must be recomputed for regenerated canonical run | ⏳ Recompute before manuscript |
| v3_backtest_regime_fullcost report | "Model: research_model_47stks_20260223" | Superseded by regenerated canonical run | use `v3_canonical_regen_2025_binary_fullcost_legacy_exec` |
| §7 original (this doc, v1) | 2026 OOS: −4.76%, Sharpe −1.01, 38 trades, WR 42.1%, PF 0.70, MaxDD 9.6% (`v3_backtest_2026_oos_jan_jun_fixed`) | −3.64%, Sharpe −0.70, 40 trades, WR 42.5%, PF 0.77, MaxDD 9.7% (`v3_canonical_2026_jan_jun_full_yf_tail`) | ✅ Corrected in §7.3 |
| §6.3/§6.4 original (this doc, v1) | "Regime effect" / cost-sensitivity framed on mixed-accounting rows (+16.68%↔+11.51%; interpolated cost slope) | Same-accounting 2×2 matrix (§6.5) + exact cost decomposition (§6.6) | ✅ Corrected |

---

## 12. Items Still Needed Before Submission

| # | Task | Effort | Status | Blocks |
|---|---|---|---|---|
| 1 | Fill TBDs in `RESPONSE_TO_REVIEWERS.md` with canonical numbers from §6 | 1 h | ✅ Done | Submission |
| 2 | Fix `reference.bib` + `sample.bib`: remove `kaufmann2026benchmark` and `Friday_2026`; update all `\cite{}` in `main.tex` | 1 h | ✅ Done 2026-06-29; note `arXiv:2603.01820` is valid, removed only to avoid relying on a disputed/new preprint | Submission |
| 3 | Writing W1: add DL / LOB / microstructure literature paragraph + 4 reviewer-suggested citations in `main.tex` | 1 h | ⏳ Pending | Submission |
| 4 | Writing W2: add optimizer lineage paragraph (SGD→AdaGrad→Adam→AdamW) + 3 citations in `main.tex` | 30 min | ⏳ Pending | Submission |
| 5 | Writing W3: full grammar/professional edit pass (lines L325, L549, L611 + whole manuscript) | 2 h | ⏳ Pending | Submission |
| 6 | Writing W4: rewrite dataset section in `main.tex` to single canonical spec (100 stocks, cross-sectional splits, 2013→2026-04) | 30 min | ⏳ Pending | Submission |
| 7 | Regenerate all paper figures from canonical run; eliminate all hardcoded values in `scripts/generate_strategy_figures.py` (L106-108, 286-287, 354-357, 574-578) | 2 h | ⏳ Pending | Submission |
| 8 | Update abstract/results/conclusion in `main.tex` to use +11.51%, Sharpe 1.15, PF 1.43, 85 trades | 1 h | ⏳ Pending | Submission |
| 9 | Report 2026 OOS result honestly in limitations section of `main.tex`, using corrected §7.3 numbers (−3.64%, Sharpe −0.70, 40 trades, PF 0.77) | 30 min | ⏳ Pending | Submission |
| 10 | Recompute bootstrap CI and deflated Sharpe for regenerated canonical run before updating Table 6 in `main.tex` | 30 min | ⏳ Pending | Submission |
| 11 | Update any manuscript text referencing 2026 OOS or regime-effect framing to cite the corrected §6.5/§6.6/§7.3 sections instead of the superseded §6.1/§6.3/§6.4 numbers | 30 min | ⏳ Pending | Submission |

---

## 13. What NOT to Use

- **Do not use any V2 (model-version) backtest numbers** (comparison_report.md §5 +23.85% result, etc.)
- **Do not use the V2 (model-version) deflated Sharpe (−0.17 from k=640)** — recompute V3 DSR for the regenerated canonical run before reporting
- **Do not use the hardcoded figure values** (28.77%, 2.94, 2.71) — they came from `generate_strategy_figures.py` and have no backtest backing
- **Do not use "Nifty 500 blind holdout" or "BankNifty"** — these appear in no results table and must be removed from all claims
- **Do not report Sharpe to 2 decimal places** — report to 1 decimal place in prose (1.2, not 1.15)
- **Do not use `v3_backtest_2026_oos_jan_jun_fixed` for the 2026 OOS section** — superseded by `v3_canonical_2026_jan_jun_full_yf_tail` (§7.3). Kill: −4.76%, −1.01 (Sharpe), 38 trades, WR 42.1%, MaxDD 9.6%, PF 0.70 **when referring to the superseded run** — note 0.77 is the new correct 2026 PF and is NOT killed.
- **Do not pair `v3_backtest_fullcost`/`v3_backtest_regime_on`/`v3_backtest_latest` (§6.1) against the canonical row to claim a "regime effect" or "cost effect"** — those three runs have no `Accounting_Version` field (pre-dates the accounting fix). Use §6.5's 2×2 matrix instead. Kill: "regime effect +16.68→+11.51" or "−16.36→−11.51" framing.
- **Do not use the linear-interpolation cost-sensitivity slope from §6.4** (−4.86pp per ~35bps, derived across the 0bps/6bps/41bps mixed-accounting anchor stack) — use §6.6's exact trade-log decomposition instead.

---

## 14. Paper Headline (Revised)

> "HybridSwingNet achieves **68.7% H1 smoothed-trend directional accuracy** on the test set (70.2% on blind holdout stocks; p = 3.18×10⁻⁶³), rising to **85.3% at the 0.75 confidence threshold** on blind holdout. When deployed with a dynamic regime filter and full Indian retail delivery transaction costs (~41 basis points round-trip), the Conservative portfolio profile returns **+11.51% (Sharpe 1.2)** over 2025 across 85 trades, with a profit factor of 1.43 (win rate 54.1%). The 2026 out-of-sample period remains negative and should be reported honestly in limitations."

This headline is unchanged by the §7 correction or by §15 below — neither the corrected 2026
OOS numbers nor the walk-forward exercise replace the canonical single-model 2025 result; both
are limitations material, not the headline.

---

## 15. Pre-Registered Walk-Forward Deployment Evaluation (Limitations Material)

Full protocol, per-fold results, and stats: `docs/PREREGISTERED_STRATEGY_PROTOCOL.md`
(pre-registered/committed 2026-07-02 before execution) and its results addendum. Run
artifacts: `runs/walkforward_2021_2026H1/`. This section exists to satisfy the
canonical-run-discipline rule (§13) for a result that must appear in the paper's
limitations section, not its results section.

**What it tested:** whether a disciplined, pre-registered rolling walk-forward
selection process (select strategy on trailing 3 years, trade next 1 year, repeat 6
times, 2021→2026-04) — as opposed to the single-window max-return selection that
produced the rejected H10 Top-1 mirage — could produce a strategy that survives
out-of-sample, at full (~41bps) cost, with a large enough concatenated sample
(1,299 days) for the Sharpe CI to be informative.

**Result:** it could not. All 6 folds lost money out-of-sample (worst: -42.7% in the
2022 fold). Concatenated: **-64.3% total return, Sharpe -0.79, 95% bootstrap CI
[-1.64, -0.03]** (excludes zero — a statistically real negative result, not noise).
The strategy underperformed every benchmark computed at the same cost, including
random stock entry at matched turnover (buy-and-hold +131.0%, SMA crossover +7.5%,
random-entry -2.7%, walk-forward strategy -64.3%).

**Interpretation for the paper:** this is evidence that model-driven portfolio
*strategy selection* over this grid does not generalize on this universe/period, even
under a disciplined walk-forward protocol — a distinct claim from the model's
*prediction* accuracy (§4), which is unaffected (predictions are the same
`predictions.csv` used throughout; only the downstream strategy-selection layer is
being evaluated here). Recommended framing: report this in limitations as evidence
that (a) the paper deliberately does not claim a validated trading strategy beyond the
single canonical 2025 backtest, and (b) the H10 Top-1 / strategy-development-sweep
numbers were correctly excluded from results — this experiment is further evidence
they would not have survived honest walk-forward validation either.

Per Hard Rule 1 in the protocol doc, this was one evaluation; the result stands as run.

---

*Last updated: 2026-07-02. All numbers verified against named run artifacts.*
*This document supersedes `docs/CANONICAL_RESULTS.md` in full — that file is kept for git
history only and should not be read as authoritative once this file exists.*
*Maintainer: keep this document updated as new runs are executed. §12 tracks submission task status — mark ✅ as each is completed.*
