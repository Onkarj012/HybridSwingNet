# StockXpert — Comprehensive Feature & Indicator Documentation

> **System Overview:** StockXpert uses two parallel pipelines:
>
> 1. **TA Confluence Engine** — Rule-based scoring using 27+ technical indicators (daily recommendations)
> 2. **ML Feature Pipeline** — 90+ engineered features fed into a neural network for multi-horizon prediction

Both pipelines share the same core indicators (RSI, MACD, BB, etc.) but the ML pipeline adds many more derived features.

---

## Table of Contents

1. [Price & Returns Features](#1-price--returns-features)
2. [Momentum Oscillators](#2-momentum-oscillators)
3. [Trend Indicators](#3-trend-indicators)
4. [Volatility Features](#4-volatility-features)
5. [Volume & Money Flow Features](#5-volume--money-flow-features)
6. [Bollinger Band Features](#6-bollinger-band-features)
7. [Moving Average Features](#7-moving-average-features)
8. [Fibonacci Retracement Features](#8-fibonacci-retracement-features)
9. [Intraday & Candle Features](#9-intraday--candle-features)
10. [Market Microstructure Features](#10-market-microstructure-features)
11. [Sentiment Features](#11-sentiment-features)
12. [Event Calendar Features](#12-event-calendar-features)
13. [Cross-Asset & Macro Features](#13-cross-asset--macro-features)
14. [Cross-Sectional (Peer Comparison) Features](#14-cross-sectional-peer-comparison-features)
15. [Target Variables (Labels)](#15-target-variables-labels)
16. [TA Confluence Scoring Dimensions](#16-ta-confluence-scoring-dimensions)
17. [Features Per Model Module](#17-features-per-model-module)

---

## 1. Price & Returns Features

These capture the basic price action characteristics.

| Feature                  | Formula                            | Period  | Range          | Significance                                                                                                           | Used In |
| ------------------------ | ---------------------------------- | :-----: | -------------- | ---------------------------------------------------------------------------------------------------------------------- | ------- |
| `log_return`             | `ln(Close_t / Close_{t-1})`        |  1 day  | ~(-0.1, +0.1)  | Daily return in log-space. Foundation for volatility calculation. Log-scale ensures symmetry between gains and losses. | Both    |
| `volume_change`          | `(Vol_t - Vol_{t-1}) / Vol_{t-1}`  |  1 day  | (-1, ∞)        | Rate of change in trading volume. Spikes suggest institutional activity or news-driven moves.                          | ML      |
| `quarterly_return`       | `(Close_t / Close_{t-60}) - 1`     | 60 days | (-0.5, +1.0)   | Medium-term momentum. Captures quarterly performance trend.                                                            | ML      |
| `momentum_5d`            | `(Close_t / Close_{t-5}) - 1`      | 5 days  | (-0.1, +0.1)   | Short-term momentum. Positive = stock has been rising over the week.                                                   | ML      |
| `momentum_10d`           | `(Close_t / Close_{t-10}) - 1`     | 10 days | (-0.15, +0.15) | Intermediate momentum. Two-week price trend.                                                                           | ML      |
| `relative_momentum`      | `mom_5d - rolling_21d_avg(mom_5d)` | 21 days | (-0.1, +0.1)   | Self-relative momentum: is the stock moving faster than its own historical average?                                    | ML      |
| `mom_3d`                 | `(Close_t / Close_{t-3}) - 1`      | 3 days  | (-0.05, +0.05) | Very short-term momentum for intraday signal confirmation.                                                             | TA      |
| `mom_5d`                 | `(Close_t / Close_{t-5}) - 1`      | 5 days  | (-0.1, +0.1)   | Used in directional confidence filter to validate confluence signals.                                                  | TA      |
| `typical_daily_move_pct` | 60-day P60 of `abs(ΔClose)/Close`  | 60 days | (0.1%, 3%)     | "How much does this stock typically move?" Used to calibrate realistic targets.                                        | TA      |

---

## 2. Momentum Oscillators

Identify overbought/oversold conditions and momentum shifts.

| Feature          | Formula                            | Period  | Range        | Significance                                                                                                                                            | Used In |
| ---------------- | ---------------------------------- | :-----: | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `rsi_14`         | Wilder's RSI                       | 14 days | 0–100        | **Primary momentum oscillator.** <30 = oversold (buy signal), >70 = overbought (sell signal). The most weighted TA indicator in confluence scoring.     | Both    |
| `rsi_6`          | Wilder's RSI (fast)                | 6 days  | 0–100        | Short-period RSI catches early momentum shifts. More sensitive to recent price action.                                                                  | Both    |
| `rsi_delta`      | `RSI_14_t - RSI_14_{t-1}`          |  1 day  | (-20, +20)   | RSI rate of change. Positive delta = momentum accelerating upward. Helps confirm trend direction.                                                       | ML      |
| `rsi_velocity`   | `RSI_14_t - RSI_14_{t-5}`          | 5 days  | (-40, +40)   | Medium-term RSI momentum. Large positive values = strong bullish acceleration.                                                                          | ML      |
| `stoch_k`        | `100 × (C - L₁₄) / (H₁₄ - L₁₄)`    | 14 days | 0–100        | **Stochastic %K.** Measures where close is relative to 14-day high-low range. <20 = oversold, >80 = overbought.                                         | Both    |
| `stoch_d`        | 3-day SMA of `stoch_k`             | 3 days  | 0–100        | Smoothed Stochastic. K crossing above D = bullish signal.                                                                                               | Both    |
| `stoch_cross`    | K > D AND K_prev ≤ D_prev          |    —    | 0 or 1       | Binary bullish crossover signal. Marks the exact moment of Stochastic crossover.                                                                        | ML      |
| `williams_r`     | `-100 × (H₁₄ - C) / (H₁₄ - L₁₄)`   | 14 days | -100 to 0    | **Williams %R.** Similar to Stochastic but inverted. <-80 = oversold, >-20 = overbought. Used alongside Stochastic for dual confirmation in TA scoring. | Both    |
| `cci`            | `(TP - SMA₂₀(TP)) / (0.015 × MAD)` | 20 days | (-300, +300) | **Commodity Channel Index.** Measures deviation from statistical mean. >+100 = overbought trend, <-100 = oversold trend.                                | Both    |
| `mfi_14`         | Volume-weighted RSI                | 14 days | 0–100        | **Money Flow Index.** RSI with volume. <20 = oversold with low buying, >80 = overbought with heavy buying.                                              | ML      |
| `mfi_oversold`   | `MFI < 20`                         |    —    | 0 or 1       | Binary flag for MFI oversold state.                                                                                                                     | ML      |
| `mfi_overbought` | `MFI > 80`                         |    —    | 0 or 1       | Binary flag for MFI overbought state.                                                                                                                   | ML      |

---

## 3. Trend Indicators

Determine direction and strength of the prevailing trend.

| Feature             | Formula                         |  Period  | Range       | Significance                                                                                                                                  | Used In |
| ------------------- | ------------------------------- | :------: | ----------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `macd`              | EMA(12) - EMA(26)               |  12/26   | (-∞, +∞)    | **MACD line.** Positive = short-term momentum > long-term → bullish. Absolute value shows momentum intensity.                                 | Both    |
| `macd_signal`       | EMA(9) of MACD                  |  9 days  | (-∞, +∞)    | Smoothed MACD. MACD crossing above signal = buy.                                                                                              | Both    |
| `macd_hist`         | MACD - Signal                   |    —     | (-∞, +∞)    | **MACD Histogram.** Distance between MACD and signal. Growing histogram = accelerating momentum. Shrinking = decelerating.                    | Both    |
| `macd_momentum`     | `MACD_hist_t - MACD_hist_{t-1}` |  1 day   | (-∞, +∞)    | Rate of change of histogram. Positive = momentum accelerating in current direction. **Second-highest weighted factor in TA scoring.**         | Both    |
| `adx`               | Average Directional Index       | 14 days  | 0–100       | **Trend strength** (not direction). >25 = trending, >40 = strong trend, <20 = range-bound. Used with `regime_sma` to confirm trend alignment. | Both    |
| `trend_strength`    | `ADX / 100`                     | 14 days  | 0–1.0       | Normalized ADX. Used in TA scoring: >0.25 combined with correct SMA regime gives maximum trend alignment points.                              | Both    |
| `regime_sma`        | `SMA(20) > SMA(50) ? 1 : 0`     |  20/50   | 0 or 1      | **Binary trend regime.** 1 = bullish (SMA20 above SMA50), 0 = bearish. Foundation of trend alignment scoring.                                 | Both    |
| `trend_persistence` | Consecutive same-direction days | 10 days  | (-10, +10)  | Counts streak of up days (+) or down days (-). Long streaks suggest strong conviction.                                                        | ML      |
| `sma_10_20_cross`   | `SMA(10) > SMA(20) ? 1 : -1`    |  10/20   | -1 or 1     | Short-term trend cross. Fastest SMA cross to detect early trend changes.                                                                      | ML      |
| `sma_20_50_cross`   | `SMA(20) > SMA(50) ? 1 : -1`    |  20/50   | -1 or 1     | Medium-term trend cross. Same as `regime_sma` but encoded as ±1.                                                                              | ML      |
| `sma_50_200_cross`  | `SMA(50) > SMA(200) ? 1 : -1`   |  50/200  | -1 or 1     | **Golden/Death Cross.** Most significant long-term trend signal. Golden Cross (1) = major bullish, Death Cross (-1) = major bearish.          | ML      |
| `price_vs_52w_high` | `Close / 252d_high`             | 252 days | 0–1.0       | Proximity to 52-week high. Values near 1.0 = at highs (momentum strength), low values = beaten down.                                          | ML      |
| `parabolic_sar`     | Parabolic Stop and Reverse      |    —     | Price-level | Trailing stop calculation. When price crosses SAR, it signals trend reversal.                                                                 | ML      |

---

## 4. Volatility Features

Measure how much the price is fluctuating.

| Feature            | Formula                           | Period  | Range         | Significance                                                                                                                                                  | Used In |
| ------------------ | --------------------------------- | :-----: | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `atr_14`           | True Range rolling mean           | 14 days | > 0           | **Average True Range.** Primary measure of daily price movement in ₹ terms. Used to set targets (0.3x–1.0x ATR) and stops (0.5x–1.5x ATR).                    | Both    |
| `atr_28`           | True Range rolling mean           | 28 days | > 0           | Longer-period ATR for smoother volatility estimate. Less reactive to single-day spikes.                                                                       | ML      |
| `natr`             | `(ATR₁₄ / Close) × 100`           | 14 days | (0.5%, 5%)    | **Normalized ATR.** Percentage-based, allowing comparison across stocks at different price levels. A stock with NATR 1% typically moves ₹10 on a ₹1000 stock. | Both    |
| `volatility_7`     | StdDev(log_return)                | 7 days  | (0, 0.1)      | Short-term realized volatility. High values = recent turbulence.                                                                                              | ML      |
| `volatility_21`    | StdDev(log_return)                | 21 days | (0, 0.1)      | Monthly volatility. Standard risk measure. Used in Z-score normalization.                                                                                     | ML      |
| `volatility_60`    | StdDev(log_return)                | 60 days | (0, 0.1)      | Quarterly volatility. Stable measure for long-term regime classification.                                                                                     | ML      |
| `vol_expansion`    | `volatility_7 / volatility_21`    |    —    | (0, 3+)       | **Volatility expansion ratio.** >1 = volatility expanding (breakout risk), <1 = compression (quiet period before potential move).                             | ML      |
| `vol_acceleration` | `Δ(vol_expansion)`                |  1 day  | (-1, +1)      | Rate of change of expansion. Positive = volatility accelerating = breakout imminent.                                                                          | ML      |
| `vol_ref`          | `rolling_10d_std(log_return) + ε` | 10 days | > 0           | Reference volatility for Z-score target normalization. Converts Z-scores back to expected ₹ returns.                                                          | ML      |
| `bb_width`         | `(BB_upper - BB_lower) / SMA₂₀`   | 20 days | (0, 0.3)      | **Bollinger Band Width.** Bandwidth squeeze (low values) → imminent breakout. Wide bands → high volatility regime.                                            | Both    |
| `atr_change`       | % change in ATR over 21 days      | 21 days | (-50%, +100%) | Trend in volatility. Rising ATR = market becoming more volatile.                                                                                              | ML      |

---

## 5. Volume & Money Flow Features

Track buying/selling pressure and institutional activity.

| Feature               | Formula                                  | Period  | Range        | Significance                                                                                                                             | Used In |
| --------------------- | ---------------------------------------- | :-----: | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `volume_sma`          | SMA(Volume, 20)                          | 20 days | > 0          | Average trading volume. Baseline for comparing current volume.                                                                           | TA      |
| `volume_ratio`        | `Volume / SMA₂₀(Volume)`                 | 20 days | (0, 5+)      | **Relative volume.** >1.5x = above average (institutional interest), <0.5x = quiet day. Contributes up to 10 points in TA scoring.       | Both    |
| `relative_volume_10d` | `Volume / SMA₁₀(Volume)`                 | 10 days | (0, 5+)      | Shorter-period relative volume. More sensitive to recent activity shifts.                                                                | ML      |
| `obv`                 | Cumulative ±Volume based on direction    |    —    | (-∞, +∞)     | **On-Balance Volume.** Running total: adds volume on up days, subtracts on down days. Rising OBV = accumulation, falling = distribution. | ML      |
| `obv_sma`             | SMA(OBV, 20)                             | 20 days | (-∞, +∞)     | Smoothed OBV trend line.                                                                                                                 | ML      |
| `obv_divergence`      | `(OBV - OBV_SMA) / abs(OBV_SMA)`         | 20 days | (-1, +1)     | OBV deviation from its mean. Positive = current accumulation exceeding average. Price rising + OBV falling = bearish divergence.         | ML      |
| `vwap_20`             | Volume-weighted avg price                | 20 days | Price-level  | **Volume Weighted Average Price.** Institutional benchmark. Price above VWAP = bullish, below = institutions are net sellers.            | ML      |
| `vwap_10`             | Volume-weighted avg price                | 10 days | Price-level  | Short-period VWAP for tighter institutional tracking.                                                                                    | ML      |
| `price_vs_vwap`       | `(Close - VWAP₂₀) / VWAP₂₀`              | 20 days | (-0.1, +0.1) | Normalized distance from VWAP. Positive = price above institutional average cost.                                                        | ML      |
| `chaikin_osc`         | EMA(3) - EMA(10) of A/D Line             |  3/10   | (-∞, +∞)     | **Chaikin Oscillator.** Measures money flow acceleration. Positive = accumulation accelerating, negative = distribution.                 | ML      |
| `chaikin_signal`      | `chaikin_osc > 0 ? 1 : 0`                |    —    | 0 or 1       | Binary accumulation indicator.                                                                                                           | ML      |
| `pv_divergence`       | `price_momentum_5d - volume_momentum_5d` | 5 days  | (-1, +1)     | **Price-Volume Divergence.** If price is rising but volume is falling = weak rally → potential reversal.                                 | ML      |

---

## 6. Bollinger Band Features

Mean-reversion and volatility breakout signals.

| Feature           | Formula                         | Period  | Range       | Significance                                                                                                                                                                         | Used In |
| ----------------- | ------------------------------- | :-----: | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| `bb_upper`        | SMA₂₀ + 2σ                      | 20 days | Price-level | Upper band. Price touching upper band = overbought / trend strength depending on context.                                                                                            | TA      |
| `bb_lower`        | SMA₂₀ - 2σ                      | 20 days | Price-level | Lower band. Price touching lower band = oversold / capitulation.                                                                                                                     | TA      |
| `bb_zscore`       | `(Close - SMA₂₀) / σ₂₀`         | 20 days | (-3, +3)    | **Position within bands.** The primary BB feature. <-1.5 = below lower band (LONG candidate), >+1.5 = above upper band (SHORT candidate). Contributes up to 15 points in TA scoring. | Both    |
| `bb_width`        | `(Upper - Lower) / SMA₂₀`       | 20 days | (0, 0.3)    | Band width = volatility gauge. Narrow bands = **Squeeze** (pre-breakout). Wide = high volatility.                                                                                    | Both    |
| `zscore_velocity` | `BB_zscore_t - BB_zscore_{t-1}` |  1 day  | (-2, +2)    | Rate of change of BB position. Positive velocity + negative zscore = bouncing off lower band.                                                                                        | ML      |

---

## 7. Moving Average Features

Trend direction, momentum, and price support/resistance levels.

| Feature     | Formula                     |  Period  | Range       | Significance                                                                                                            | Used In |
| ----------- | --------------------------- | :------: | ----------- | ----------------------------------------------------------------------------------------------------------------------- | ------- |
| `sma_5`     | Simple Moving Average       |  5 days  | Price-level | Ultra-short trend. Used for immediate momentum reference.                                                               | TA      |
| `sma_20`    | Simple Moving Average       | 20 days  | Price-level | Short-term trend. Basis for Bollinger Bands. Key support in uptrends.                                                   | Both    |
| `sma_50`    | Simple Moving Average       | 50 days  | Price-level | Medium-term trend. SMA20 vs SMA50 cross defines the trend regime.                                                       | Both    |
| `sma_100`   | Simple Moving Average       | 100 days | Price-level | Intermediate trend. Institutional support level.                                                                        | Both    |
| `sma_200`   | Simple Moving Average       | 200 days | Price-level | Long-term trend. **The most important SMA.** Price above SMA200 = structural bull market.                               | Both    |
| `sma_20_z`  | Z-score of (Close - SMA₂₀)  | 20 days  | (-3, +3)    | How far price has deviated from its 20-day mean in standard deviation terms. Extreme values → mean-reversion potential. | ML      |
| `sma_50_z`  | Z-score of (Close - SMA₅₀)  | 50 days  | (-3, +3)    | Medium-term Z-score. High values = extended above/below medium-term average.                                            | ML      |
| `sma_100_z` | Z-score of (Close - SMA₁₀₀) | 100 days | (-3, +3)    | Quarterly deviation from mean trend.                                                                                    | ML      |
| `sma_200_z` | Z-score of (Close - SMA₂₀₀) | 200 days | (-3, +3)    | Annual deviation. Extreme values have historically strong mean-reversion characteristics.                               | ML      |

---

## 8. Fibonacci Retracement Features

Support/resistance levels based on golden ratio mathematics.

| Feature                   | Formula                               | Period  | Range       | Significance                                                                                                                                   | Used In |
| ------------------------- | ------------------------------------- | :-----: | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `fib_swing_high`          | Rolling 60-day high                   | 60 days | Price-level | Recent swing high point. Upper boundary of the Fibonacci range.                                                                                | Both    |
| `fib_swing_low`           | Rolling 60-day low                    | 60 days | Price-level | Recent swing low point. Lower boundary of the Fibonacci range.                                                                                 | Both    |
| `fib_0.0`                 | Swing low                             |    —    | Price-level | 0% retracement level = the low. Strong support.                                                                                                | Both    |
| `fib_23.6`                | Low + 0.236 × Range                   |    —    | Price-level | **23.6% level.** Shallow retracement in strong trends. First support/resistance zone.                                                          | Both    |
| `fib_38.2`                | Low + 0.382 × Range                   |    —    | Price-level | **38.2% level.** Common pullback zone. Healthy correction level.                                                                               | Both    |
| `fib_50.0`                | Low + 0.500 × Range                   |    —    | Price-level | **50% level.** Psychological midpoint. Not a true Fibonacci ratio but highly watched.                                                          | Both    |
| `fib_61.8`                | Low + 0.618 × Range                   |    —    | Price-level | **61.8% level (Golden Ratio).** The most significant Fibonacci level. Strong support/resistance.                                               | Both    |
| `fib_78.6`                | Low + 0.786 × Range                   |    —    | Price-level | **78.6% level.** Deep retracement. Last defense before full reversal.                                                                          | Both    |
| `fib_100.0`               | Swing high                            |    —    | Price-level | 100% retracement = the high. Major resistance on the way up.                                                                                   | Both    |
| `fib_distance_to_nearest` | `min(abs(Close - Fib_i)) / Close`     |    —    | (0, 0.1)    | Proximity to any Fibonacci level. **Close to a level (< 2%) gets 10 TA scoring points.** Levels act as magnets — price tends to react at them. | Both    |
| `fib_relative_position`   | `(Close - Low) / (High - Low)`        | 60 days | 0–1.0       | Where the price is within the Fibonacci range. 0 = at lows, 1 = at highs. Continuous feature for ML zone classification.                       | ML      |
| `fib_zone`                | Discrete zone (0–5) based on position |    —    | 0–5         | Categorical: which Fibonacci zone is the price in? 0 = below 23.6%, 5 = above 78.6%.                                                           | ML      |

---

## 9. Intraday & Candle Features

Capture within-day price dynamics and gap analysis.

| Feature            | Formula                            | Period | Range          | Significance                                                                                                                | Used In |
| ------------------ | ---------------------------------- | :----: | -------------- | --------------------------------------------------------------------------------------------------------------------------- | ------- |
| `overnight_return` | `ln(Open_t / Close_{t-1})`         | 1 day  | (-0.05, +0.05) | **Gap return.** Positive = gap up, negative = gap down. Large gaps often indicate news or institutional activity overnight. | ML      |
| `intraday_range`   | `(High - Low) / Close`             | 1 day  | (0, 0.1)       | Daily volatility as a percentage. Wide ranges = high intraday volatility.                                                   | ML      |
| `body_ratio`       | `(Close - Open) / (High - Low)`    | 1 day  | (-1, +1)       | Candle body proportion. +1 = full bullish candle (marubozu), -1 = full bearish, 0 = doji (indecision).                      | ML      |
| `upper_wick_ratio` | `(High - max(C,O)) / (High - Low)` | 1 day  | (0, 1)         | Upper wick proportion. Large upper wick = selling pressure at higher prices (bearish rejection).                            | ML      |
| `gap_direction`    | `sign(overnight_return)`           |   —    | -1, 0, 1       | Binary gap direction indicator.                                                                                             | ML      |
| `gap_streak`       | Consecutive same-direction gaps    |   —    | (-10, +10)     | Streak of gap-ups (+) or gap-downs (-). Extended streaks suggest institutional systematic buying/selling.                   | ML      |

---

## 10. Market Microstructure Features

Detect institutional activity and market quality.

| Feature           | Formula                            | Period  | Range    | Significance                                                                                                                                                                | Used In |
| ----------------- | ---------------------------------- | :-----: | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `illiquidity`     | `abs(return) / Volume` (Amihud)    |  1 day  | (0, ∞)   | **Amihud illiquidity ratio.** High = illiquid stock (price moves more per ₹ traded). Low liquidity stocks are riskier but may offer more alpha.                             | ML      |
| `kyle_lambda`     | `abs(return) / sqrt(Volume)`       |  1 day  | (0, ∞)   | **Kyle's Lambda (price impact).** Measures how much one unit of volume moves the price. High lambda = informed trading is likely moving prices.                             | ML      |
| `hl_spread`       | Rolling mean of `(H-L) / midprice` | 5 days  | (0, 0.1) | **High-Low spread estimator** (Corwin-Schultz proxy). Approximates bid-ask spread from OHLC data. Wide spread = low liquidity.                                              | ML      |
| `clv`             | `((C-L) - (H-C)) / (H-L)`          |  1 day  | (-1, +1) | **Close Location Value.** Where did the stock close within its daily range? +1 = at high (bullish), -1 = at low (bearish). Key for detecting institutional buying at close. | ML      |
| `volume_zscore`   | `(Vol - μ₂₀) / σ₂₀`                | 20 days | (-2, +5) | Volume Z-score. Values >2 = unusual volume spike, likely institutional activity or news.                                                                                    | ML      |
| `trade_intensity` | `Volume / Volatility`              | 10 days | (0, ∞)   | Volume normalized by volatility. High intensity = lots of trading with small moves = institutional accumulation/distribution (iceberg orders).                              | ML      |

---

## 11. Sentiment Features

News-derived sentiment indicators (computed when `sentiment_mean` is available in data).

| Feature                 | Formula                                 | Period  | Range        | Significance                                                                       | Used In |
| ----------------------- | --------------------------------------- | :-----: | ------------ | ---------------------------------------------------------------------------------- | ------- |
| `sentiment_mean`        | Average headline sentiment score        |  1 day  | (-1, +1)     | Daily news sentiment from TextBlob. +1 = very positive news, -1 = very negative.   | ML      |
| `sentiment_count`       | Number of news articles                 |  1 day  | (0, 50+)     | News volume. High count + extreme sentiment = strong signal.                       | ML      |
| `sentiment_rolling_5d`  | SMA₅(sentiment_mean)                    | 5 days  | (-1, +1)     | Short-term sentiment trend. Smooths out daily noise.                               | ML      |
| `sentiment_rolling_10d` | SMA₁₀(sentiment_mean)                   | 10 days | (-1, +1)     | Medium-term sentiment for swing trades.                                            | ML      |
| `sentiment_rolling_21d` | SMA₂₁(sentiment_mean)                   | 21 days | (-1, +1)     | Monthly sentiment baseline.                                                        | ML      |
| `sentiment_momentum`    | `sentiment_mean - sentiment_rolling_5d` |    —    | (-1, +1)     | Sentiment acceleration. Positive = news getting MORE positive than recent average. | ML      |
| `sentiment_spike`       | `z_score(count) > 2.0 ? 1 : 0`          | 21 days | 0 or 1       | Binary flag for unusual news volume. Spikes often precede large price moves.       | ML      |
| `sentiment_trend`       | `Δ₅(sentiment_rolling_10d)`             | 5 days  | (-0.5, +0.5) | Slope of 10-day sentiment. Positive = news sentiment improving over time.          | ML      |

---

## 12. Event Calendar Features

Market event flags for India-specific calendar effects.

| Feature              | Formula                            | Period | Range  | Significance                                                                                                                                           | Used In |
| -------------------- | ---------------------------------- | :----: | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| `is_expiry_week`     | Last 7 days of month               |   —    | 0 or 1 | **F&O Expiry Week.** Monthly derivatives expiry (last Thursday) causes elevated volatility, short covering rallies, and options-related gamma effects. | ML      |
| `is_earnings_season` | Apr/May, Jul/Aug, Oct/Nov, Jan/Feb |   —    | 0 or 1 | **Earnings Season.** Quarters when companies report results. Higher gap risk due to post-earnings surprises.                                           | ML      |
| `is_policy_month`    | Feb, Apr, Jun, Aug, Oct, Dec       |   —    | 0 or 1 | **RBI Policy Month.** Bi-monthly monetary policy decisions affect banking, rate-sensitive sectors significantly.                                       | ML      |

---

## 13. Cross-Asset & Macro Features

Broad market context and regime indicators.

| Feature       | Formula                             | Period  | Range        | Significance                                                                                                                 | Used In |
| ------------- | ----------------------------------- | :-----: | ------------ | ---------------------------------------------------------------------------------------------------------------------------- | ------- |
| `macro_proxy` | Stock's own rolling momentum        | 21 days | (-0.1, +0.1) | Proxy for market regime when NIFTY50 index data is unavailable. Positive = bullish macro environment.                        | ML      |
| `kc_upper`    | EMA₂₀ + 2 × ATR                     | 20 days | Price-level  | **Keltner Channel upper.** Combines trend (EMA) with volatility (ATR). More robust than Bollinger Bands for trend-following. | ML      |
| `kc_lower`    | EMA₂₀ - 2 × ATR                     | 20 days | Price-level  | Keltner lower channel. Price below = deeply oversold.                                                                        | ML      |
| `kc_width`    | `(KC_upper - KC_lower) / KC_middle` | 20 days | (0, 0.3)     | Channel width = current volatility state. Squeeze = potential breakout.                                                      | ML      |
| `price_vs_kc` | `(Close - KC_mid) / (KC_width/2)`   | 20 days | (-3, +3)     | Position within Keltner Channels. Similar to BB Z-score but ATR-based instead of std-based.                                  | ML      |

---

## 14. Cross-Sectional (Peer Comparison) Features

Compare a stock to all other stocks in the universe on the same date.

| Feature                 | Formula                                          |  Period  | Range        | Significance                                                                                                 | Used In |
| ----------------------- | ------------------------------------------------ | :------: | ------------ | ------------------------------------------------------------------------------------------------------------ | ------- |
| `return_rank_5d`        | Percentile rank of 5-day return                  |  5 days  | 0–1.0        | How does this stock's 5-day return compare to all peers? 1.0 = top performer.                                | ML      |
| `return_rank_10d`       | Percentile rank of 10-day return                 | 10 days  | 0–1.0        | 10-day relative performance ranking.                                                                         | ML      |
| `return_rank_21d`       | Percentile rank of 21-day return                 | 21 days  | 0–1.0        | Monthly relative performance. Key for **relative momentum strategies.**                                      | ML      |
| `return_zscore_Nd`      | Z-score of return vs peers                       | 5/10/21d | (-3, +3)     | Standard deviations from the universe average return. >2 = strong outperformer, <-2 = strong underperformer. | ML      |
| `return_quintile_Nd`    | Quintile (0–4) of returns                        | 5/10/21d | 0–4          | Which quintile of performers? 0 = bottom 20%, 4 = top 20%. Used for mean-reversion signals.                  | ML      |
| `volume_rank`           | Percentile rank of volume                        |  Daily   | 0–1.0        | Relative volume activity vs peers. High rank = more active than most stocks today.                           | ML      |
| `volatility_rank_7d`    | Percentile rank of 7-day vol                     |  7 days  | 0–1.0        | Risk rank vs peers. High = among the most volatile stocks.                                                   | ML      |
| `momentum_score`        | Weighted rank combo (21d×0.5 + 10d×0.3 + 5d×0.2) |    —     | 0–1.0        | Composite momentum score for ranking stocks.                                                                 | ML      |
| `momentum_decile`       | Decile (0–9) of momentum_score                   |    —     | 0–9          | Top decile (9) = strongest momentum. Used for momentum-based stock selection.                                | ML      |
| `mean_reversion_signal` | Bottom quintile in both 5d and 21d               |    —     | 0 or 1       | Flag for potential mean-reversion candidates. Stock is in bottom 20% over both short and medium horizons.    | ML      |
| `price_vs_market`       | `Close / avg(all_closes) - 1`                    |  Daily   | (-1, +1)     | Relative valuation vs universe average price.                                                                | ML      |
| `relative_strength_Nd`  | Pct change of price_vs_market                    |  10/21d  | (-0.2, +0.2) | Is the stock gaining or losing ground vs peers over time?                                                    | ML      |
| `market_corr_21d`       | Rolling correlation with market return           | 21 days  | (-1, +1)     | How closely does this stock move with the market? Low correlation = diversification benefit.                 | ML      |
| `beta_21d`              | `Cov(r_i, r_m) / Var(r_m)`                       | 21 days  | (0, 3)       | **Beta coefficient.** >1 = amplifies market moves, <1 = defensive stock.                                     | ML      |

---

## 15. Target Variables (Labels)

What the ML model learns to predict.

| Target                | Formula                               | Significance                                               |
| --------------------- | ------------------------------------- | ---------------------------------------------------------- |
| `target_h1`           | Z-score of smoothed 1-day log return  | Next-day direction and magnitude, normalized by volatility |
| `target_h3`           | Z-score of smoothed 3-day log return  | 3-day swing prediction                                     |
| `target_h5`           | Z-score of smoothed 5-day log return  | Weekly swing prediction                                    |
| `target_h7`           | Z-score of smoothed 7-day log return  | Extended swing prediction                                  |
| `target_h10`          | Z-score of smoothed 10-day log return | Two-week trend prediction                                  |
| `level_target_hN_res` | Z-score of future max high            | Predicted resistance level (max price in horizon)          |
| `level_target_hN_sup` | Z-score of future min low             | Predicted support level (min price in horizon)             |
| `level_target_hN_tgt` | Z-score of future close               | Predicted close at end of horizon                          |

### Target Normalization Pipeline

```
1. Raw Return  → log(Close_{t+h} / Close_t)
2. Smoothing   → rolling_mean(window = max(3, min(h, 7)))
3. Z-Score     → smoothed_return / rolling_10d_std(log_return)
4. Clipping    → [-5, +5] sigma
```

> **Why Z-scores?** Makes targets stationary across different volatility regimes. A +2σ move means "2 standard deviations" whether the stock is COALINDIA (low vol) or BALKRISIND (high vol), allowing the model to learn universal patterns.

---

## 16. TA Confluence Scoring Dimensions

The recommendation engine scores each stock 0–100 across 7 dimensions.

| #   | Dimension                    |     Max Points     | Indicators Used                | LONG Criteria (Full Score)                       | SHORT Criteria (Full Score)                        |
| --- | ---------------------------- | :----------------: | ------------------------------ | ------------------------------------------------ | -------------------------------------------------- |
| 1   | **RSI Zone**                 |         20         | `rsi_14`                       | RSI < 30 (Oversold)                              | RSI > 70 (Overbought)                              |
| 2   | **MACD Direction**           |         20         | `macd_hist`, `macd_momentum`   | Histogram > 0 AND momentum > 0 (accelerating up) | Histogram < 0 AND momentum < 0 (accelerating down) |
| 3   | **Bollinger Position**       |         15         | `bb_zscore`                    | Z-score < -1.5 (below lower band)                | Z-score > +1.5 (above upper band)                  |
| 4   | **Trend Alignment**          | 15 (or -5 penalty) | `regime_sma`, `trend_strength` | SMA20 > SMA50 AND ADX > 25                       | SMA20 < SMA50 AND ADX > 25                         |
| 5   | **Volume Confirmation**      |         10         | `volume_ratio`                 | Volume > 1.5× daily average                      | Volume > 1.5× daily average                        |
| 6   | **Stochastic + Williams %R** |         10         | `stoch_k`, `williams_r`        | Stoch < 20 AND W%R < -80                         | Stoch > 80 AND W%R > -20                           |
| 7   | **Fibonacci Proximity**      |         10         | `fib_distance_to_nearest`      | Price within 2% of any Fibonacci level           | Price within 2% of any Fibonacci level             |
|     | **TOTAL**                    |      **100**       |                                |                                                  |                                                    |

### Minimum Score Thresholds

| Horizon       | Min Score | Description                              |
| ------------- | :-------: | ---------------------------------------- |
| Intraday      |    65     | High conviction only — tight targets     |
| Swing 1-2 Day |    55     | Moderate conviction — more opportunities |
| Swing 3-5 Day |    45     | Lower bar — wider targets give more room |

### ATR-Based Target Calibration

| Horizon    |  Target  | Extended Target | Stop Loss | Typical R:R |
| ---------- | :------: | :-------------: | :-------: | :---------: |
| Intraday   | 0.3× ATR |    0.5× ATR     | 0.5× ATR  |    0.60     |
| Swing 1-2d | 0.5× ATR |    0.8× ATR     | 0.75× ATR |    0.67     |
| Swing 3-5d | 1.0× ATR |    1.5× ATR     | 1.5× ATR  |    0.67     |

---

## 17. Features Per Model Module

The ML model has five specialized encoders, each receiving a different subset of features. Inputs are time-series sequences (T timesteps) for the three temporal encoders, and point-in-time snapshots for the two context encoders.

### Architecture Summary

```
X_short  (T=7  days, 8  features)  → ResNLS Encoder     → hidden_dim
X_mid    (T=21 days, 12 features)  → BiGRU Encoder      → hidden_dim
X_long   (T=60 days, 8  features)  → BiLSTM Encoder     → hidden_dim
X_context      (18 features)       → Context Encoder    → hidden_dim
X_sentiment     (9 features)       → Sentiment Encoder  → hidden_dim
                                           ↓
                              Attention Fusion + Horizon Mixer
                                           ↓
                            Direction / Magnitude / Level Heads
```

---

### Module 1: ResNLS Encoder — Short-Term (7-day window)

**Architecture:** Residual Non-Local Scaling network. Captures immediate price momentum and intraday-scale signals. Each prediction head for H1 gets a 70% blend of this encoder's output.

| #   | Feature          | Category   | Why Short-Term?                                              |
| --- | ---------------- | ---------- | ------------------------------------------------------------ |
| 1   | `log_return`     | Price      | Most recent daily returns are the fastest signal             |
| 2   | `volume_change`  | Volume     | Volume spikes are immediate reaction signals                 |
| 3   | `rsi_6`          | Momentum   | Fast RSI reacts quickly to recent price swings               |
| 4   | `volatility_7`   | Volatility | 7-day realized vol for immediate risk assessment             |
| 5   | `bb_zscore`      | Bollinger  | Current deviation from mean — mean-reversion signal          |
| 6   | `rsi_delta`      | Momentum   | RSI rate of change — momentum acceleration/deceleration      |
| 7   | `obv_divergence` | Volume     | OBV vs its own avg — recent accumulation/distribution shifts |
| 8   | `price_vs_vwap`  | Volume     | Distance from institutional average price                    |

---

### Module 2: BiGRU Encoder — Medium-Term (21-day window)

**Architecture:** Bidirectional Gated Recurrent Unit. Captures swing-level trends and multi-day momentum patterns. H3 predictions get a 70% blend of this encoder's output.

| #   | Feature           | Category   | Why Medium-Term?                                        |
| --- | ----------------- | ---------- | ------------------------------------------------------- |
| 1   | `rsi_14`          | Momentum   | Standard RSI needs 14+ days of history to stabilize     |
| 2   | `macd`            | Trend      | MACD (12,26) EMA spread requires 26-day window          |
| 3   | `macd_signal`     | Trend      | 9-day EMA of MACD, needs medium-term context            |
| 4   | `macd_hist`       | Trend      | Histogram trend visible over weeks, not days            |
| 5   | `bb_width`        | Volatility | Band compression/expansion cycles are 2–3 week patterns |
| 6   | `sma_20_z`        | Moving Avg | Z-score relative to 20-day mean                         |
| 7   | `volatility_21`   | Volatility | Monthly realized volatility — the standard risk window  |
| 8   | `macd_momentum`   | Trend      | Histogram acceleration — swing trade timing signal      |
| 9   | `zscore_velocity` | Bollinger  | Rate of change in BB position — reversion speed         |
| 10  | `mfi_14`          | Volume     | Money Flow Index — 14-day volume-weighted momentum      |
| 11  | `stoch_k`         | Momentum   | 14-day high-low range based oscillator                  |
| 12  | `chaikin_osc`     | Volume     | Short EMA vs long EMA of A/D line — money flow cycles   |

---

### Module 3: BiLSTM Encoder — Long-Term (60-day window)

**Architecture:** Bidirectional Long Short-Term Memory. Captures structural trends, regime changes, and macro-level patterns. H10 predictions use 100% of this encoder's output.

| #   | Feature            | Category   | Why Long-Term?                                          |
| --- | ------------------ | ---------- | ------------------------------------------------------- |
| 1   | `adx`              | Trend      | ADX needs 60+ days to reflect true trend strength       |
| 2   | `sma_50_z`         | Moving Avg | Z-score vs 50-day SMA — medium-long mean reversion      |
| 3   | `volatility_60`    | Volatility | Quarterly realized volatility for regime classification |
| 4   | `trend_strength`   | Trend      | Normalized ADX — meaningful only over long windows      |
| 5   | `regime_sma`       | Trend      | SMA20 vs SMA50 crossover — structural regime signal     |
| 6   | `vol_acceleration` | Volatility | Rate of change of volatility expansion ratio            |
| 7   | `sma_100_z`        | Moving Avg | Z-score vs 100-day SMA — quarterly mean deviation       |
| 8   | `atr_14`           | Volatility | ATR in absolute ₹ terms — for target level calibration  |

---

### Module 4: Context Encoder — Point-in-Time Snapshot

**Architecture:** Multi-layer MLP (ContextEncoder). Receives a single snapshot of the _current state_ (not a time series). Provides market condition awareness to all horizon heads.

| #   | Feature                   | Category   | Role in Context                                            |
| --- | ------------------------- | ---------- | ---------------------------------------------------------- |
| 1   | `bb_zscore`               | Bollinger  | Are we currently extended from mean?                       |
| 2   | `rsi_14`                  | Momentum   | Current momentum zone (oversold/overbought)                |
| 3   | `trend_strength`          | Trend      | How strong is the current trend?                           |
| 4   | `regime_sma`              | Trend      | Bull or bear regime right now?                             |
| 5   | `volatility_21`           | Volatility | Current risk environment                                   |
| 6   | `vol_expansion`           | Volatility | Is volatility expanding (breakout) or contracting (range)? |
| 7   | `pv_divergence`           | Volume     | Price-volume alignment — is the move genuine?              |
| 8   | `sentiment_mean`          | Sentiment  | Today's news sentiment tone                                |
| 9   | `sentiment_momentum`      | Sentiment  | Is sentiment improving or deteriorating?                   |
| 10  | `sentiment_rolling_5d`    | Sentiment  | Short-term sentiment trend                                 |
| 11  | `rsi_delta`               | Momentum   | RSI direction (accelerating up or down?)                   |
| 12  | `macd_momentum`           | Trend      | MACD histogram acceleration                                |
| 13  | `fib_distance_to_nearest` | Fibonacci  | Proximity to nearest Fibonacci support/resistance          |
| 14  | `fib_relative_position`   | Fibonacci  | Where in the Fibonacci range is price?                     |
| 15  | `fib_zone`                | Fibonacci  | Which Fibonacci zone (0–5)?                                |
| 16  | `natr`                    | Volatility | Normalized ATR — how volatile is this stock right now?     |
| 17  | `williams_r`              | Momentum   | Current overbought/oversold reading                        |
| 18  | `cci`                     | Momentum   | Deviation from statistical average price                   |

---

### Module 5: Sentiment Encoder — Point-in-Time News Features

**Architecture:** Dedicated MLP (`SentimentEncoder`). Fixed input dimension of 9. Processes only news-derived features. Output is fused with all other encoders via attention. When no live news is available, zero-padded inputs are passed.

| #   | Feature                 | Description                                                   |
| --- | ----------------------- | ------------------------------------------------------------- |
| 1   | `sentiment_mean`        | Average sentiment score across today's news articles          |
| 2   | `sentiment_count`       | Number of news articles processed today                       |
| 3   | `sentiment_std`         | Dispersion of sentiment scores (conflicting news = high std)  |
| 4   | `sentiment_rolling_5d`  | 5-day smoothed sentiment average                              |
| 5   | `sentiment_momentum`    | `sentiment_mean - sentiment_rolling_5d` (improving/worsening) |
| 6   | `sentiment_spike`       | Binary: unusual news volume detected (z-score > 2)            |
| 7   | `sentiment_rolling_10d` | 10-day smoothed sentiment (swing trade context)               |
| 8   | `sentiment_rolling_21d` | Monthly sentiment baseline                                    |
| 9   | `sentiment_trend`       | Slope of 10-day sentiment rolling average                     |

---

### Horizon-to-Module Blending Weights

The model uses graduated feature routing so each prediction horizon gets encoder outputs blended by relevance:

| Horizon           | Short (ResNLS) | Mid (BiGRU) | Long (BiLSTM) | Use Case          |
| ----------------- | :------------: | :---------: | :-----------: | ----------------- |
| **H1** (next day) |    **70%**     |     30%     |      0%       | Intraday momentum |
| **H3** (3 days)   |      30%       |   **70%**   |      0%       | Short swing       |
| **H5** (5 days)   |       0%       |     50%     |    **50%**    | Weekly swing      |
| **H7** (7 days)   |       0%       |     30%     |    **70%**    | Extended swing    |
| **H10** (10 days) |       0%       |     0%      |   **100%**    | Trend following   |

> Context and Sentiment encoders contribute to **all horizons equally** via the attention fusion layer.

---

## Total Feature Count Summary

| Category             | Feature Count | Used By |
| -------------------- | :-----------: | ------- |
| Price & Returns      |       9       | Both    |
| Momentum Oscillators |      12       | Both    |
| Trend Indicators     |      13       | Both    |
| Volatility           |      11       | Both    |
| Volume & Money Flow  |      12       | Both    |
| Bollinger Bands      |       5       | Both    |
| Moving Averages      |       9       | Both    |
| Fibonacci            |      12       | Both    |
| Intraday & Candle    |       6       | ML only |
| Microstructure       |       6       | ML only |
| Sentiment            |       8       | ML only |
| Event Calendar       |       3       | ML only |
| Cross-Asset & Macro  |       5       | ML only |
| Cross-Sectional      |      14+      | ML only |
| **TOTAL**            |   **~125+**   |         |

> ⚠️ Not all features contribute equally. Feature importance analysis shows RSI, MACD momentum, BB z-score, trend strength, and volatility features are the most predictive across all horizons.
