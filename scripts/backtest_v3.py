#!/usr/bin/env python3
"""
StockXpert V3 — capital-aware portfolio backtest.

Uses a trained V3 checkpoint to generate daily predictions, then simulates
entries/exits with configured capital, position limits, ATR targets/stops,
and brokerage — same engine as ``backtest_historical.py``.

Usage:
    PYTHONPATH=src python scripts/backtest_v3.py \\
      --run-dir runs/20260522_181848_v3_nifty100_from_nifty500 \\
      --data-dir data/nifty500 \\
      --start 2025-01-01 --end 2025-12-31

    # One profile only
    PYTHONPATH=src python scripts/backtest_v3.py --run-dir runs/... --profile moderate

    # Custom capital / thresholds
    PYTHONPATH=src python scripts/backtest_v3.py --run-dir runs/... \\
      --capital 500000 --max-positions 5 --min-p-up 0.60 --horizon 5,7

    # Production filters (regime + rolling top-15 stocks)
    PYTHONPATH=src python scripts/backtest_v3.py --run-dir runs/... --production --profile moderate
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import pickle
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

import train_v3
from backtest_historical import (
    PROFILES,
    HitRateEvaluator,
    PortfolioBacktester,
    Prediction,
    ProfileConfig,
    print_hit_rate_report,
    print_profile_results,
    save_markdown_report,
)
from historical_data_loader import HistoricalDataLoader
from stockxpert.dataset.scaling import ScalerGroup
from stockxpert.features.builder import FeatureBuilder
from stockxpert.utils import ensure_dir, get_device, set_seed
from stockxpert.v3.config import load_v3_config
from stockxpert.v3.dataset.make_samples import STOCK_TRAIT_COLUMNS, compute_stock_traits
from stockxpert.v3.models.stockxpert import StockXpertModelV3

console = Console()
logger = logging.getLogger("backtest_v3")


class V3ModelInference:
    """Load a V3 checkpoint and generate per-day predictions."""

    def __init__(self, run_dir: Path, config_path: Path | None = None):
        self.run_dir = Path(run_dir)
        cfg_path = config_path or self.run_dir / "config.yaml"
        self.cfg = load_v3_config(cfg_path)
        self.device = get_device(getattr(self.cfg.run, "device", "auto"))
        self.model = self._load_model()
        self.scaler_group = ScalerGroup.load(self.run_dir / "artifacts" / "scalers.pkl")
        self.calibrators = self._load_calibrators()
        self.symbol_map = {symbol: idx for idx, symbol in enumerate(self.cfg.data.symbols)}
        self.stock_traits: Dict[str, np.ndarray] = {}
        logger.info(
            "V3 model loaded: %d symbols, horizons=%s, device=%s",
            len(self.cfg.data.symbols),
            self.cfg.features.horizons,
            self.device,
        )

    def _load_model(self) -> StockXpertModelV3:
        ckpt_path = self.run_dir / "checkpoints" / "model_final.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sample = None  # dims come from config
        model = StockXpertModelV3(
            short_dim=len(self.cfg.features.short_features),
            mid_dim=len(self.cfg.features.mid_features),
            long_dim=len(self.cfg.features.long_features),
            context_dim=len(self.cfg.features.context_features),
            num_horizons=len(self.cfg.features.horizons),
            hidden_dim=self.cfg.model.hidden_dim,
            stock_embed_dim=self.cfg.model.stock_embed_dim,
            stock_traits_dim=self.cfg.model.stock_traits_dim,
            attn_heads=self.cfg.model.attn_heads,
            dropout=self.cfg.model.dropout,
            use_regime_head=getattr(self.cfg.model, "use_regime_head", False),
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def _load_calibrators(self) -> dict:
        for name in ("calibrators.pkl", "calibrators_all_horizons.pkl"):
            path = self.run_dir / "artifacts" / name
            if path.exists():
                with path.open("rb") as f:
                    return pickle.load(f)
        logger.warning("No V3 calibrators found; using raw probabilities.")
        return {}

    def set_stock_traits(self, feature_data: Dict[str, pd.DataFrame]) -> None:
        self.stock_traits = {
            symbol: compute_stock_traits(df, STOCK_TRAIT_COLUMNS, self.cfg.model.stock_traits_dim)
            for symbol, df in feature_data.items()
        }

    def predict_for_date(self, feature_data: Dict[str, pd.DataFrame], target_date: str) -> List[Prediction]:
        predictions: list[Prediction] = []
        with torch.no_grad():
            for symbol, df in feature_data.items():
                if symbol not in self.symbol_map:
                    continue
                date_mask = df.index.strftime("%Y-%m-%d") == target_date
                if not date_mask.any():
                    continue
                t = df.index.get_indexer([df.index[date_mask][-1]])[0]
                if t < self.cfg.features.windows.long:
                    continue
                try:
                    predictions.extend(self._predict_single(df, t, symbol))
                except Exception as exc:
                    logger.debug("Prediction failed for %s: %s", symbol, exc)
        return predictions

    def _predict_single(self, df: pd.DataFrame, t: int, symbol: str) -> List[Prediction]:
        cfg = self.cfg

        def get_win(feat_list, win_len):
            start = t - win_len + 1
            return df.iloc[start : t + 1][feat_list].values.astype(np.float32)

        x_short = get_win(cfg.features.short_features, cfg.features.windows.short)
        x_mid = get_win(cfg.features.mid_features, cfg.features.windows.mid)
        x_long = get_win(cfg.features.long_features, cfg.features.windows.long)
        x_context = df.iloc[t][cfg.features.context_features].values.astype(np.float32)

        if all(col in df.columns for col in train_v3.SENTIMENT_FEATURES):
            x_sentiment = df.iloc[t][train_v3.SENTIMENT_FEATURES].values.astype(np.float32)
        else:
            x_sentiment = np.zeros(len(train_v3.SENTIMENT_FEATURES), dtype=np.float32)

        x_s_scaled = self.scaler_group.scaler_short.transform(x_short)
        x_m_scaled = self.scaler_group.scaler_mid.transform(x_mid)
        x_l_scaled = self.scaler_group.scaler_long.transform(x_long)
        x_c_scaled = self.scaler_group.scaler_context.transform(x_context.reshape(1, -1))[0]

        if getattr(self.scaler_group, "scaler_sentiment", None) is not None:
            try:
                x_sent_scaled = self.scaler_group.scaler_sentiment.transform(x_sentiment.reshape(1, -1))[0]
            except Exception:
                x_sent_scaled = x_sentiment
        else:
            x_sent_scaled = x_sentiment

        traits = self.stock_traits.get(
            symbol,
            np.zeros(cfg.model.stock_traits_dim, dtype=np.float32),
        )

        t_short = torch.tensor(x_s_scaled).unsqueeze(0).to(self.device)
        t_mid = torch.tensor(x_m_scaled).unsqueeze(0).to(self.device)
        t_long = torch.tensor(x_l_scaled).unsqueeze(0).to(self.device)
        t_context = torch.tensor(x_c_scaled).unsqueeze(0).to(self.device)
        t_sent = torch.tensor(x_sent_scaled).unsqueeze(0).to(self.device)
        t_traits = torch.tensor(traits).unsqueeze(0).to(self.device)

        outputs = self.model(t_short, t_mid, t_long, t_context, t_sent, t_traits)
        direction_logits = outputs[0]
        magnitude_pred = outputs[1]

        raw_probs = torch.sigmoid(direction_logits).cpu().numpy()
        logits = direction_logits.cpu().numpy()
        if self.calibrators:
            cal_probs = train_v3.apply_calibrators(
                raw_probs, logits, self.calibrators, cfg.features.horizons
            )
        else:
            cal_probs = raw_probs

        current_price = float(df.iloc[t]["Close"])
        vol_ref = float(df.iloc[t].get("vol_ref", 0.015))
        atr = float(df.iloc[t]["atr_14"]) if "atr_14" in df.columns else current_price * 0.02
        date_str = df.index[t].strftime("%Y-%m-%d")

        preds: list[Prediction] = []
        for h_idx, horizon in enumerate(cfg.features.horizons):
            p_up = float(cal_probs[0, h_idx])
            mag_z = float(magnitude_pred[0, h_idx].item())
            conf = float(max(p_up, 1.0 - p_up))

            abs_dlog = abs(mag_z) * vol_ref
            abs_ret = np.exp(abs_dlog) - 1.0
            expected_ret = abs_ret if p_up > 0.5 else -abs_ret

            preds.append(
                Prediction(
                    date=date_str,
                    symbol=symbol,
                    horizon=horizon,
                    p_up=p_up,
                    predicted_direction="UP" if p_up > 0.5 else "DOWN",
                    predicted_magnitude=expected_ret * 100,
                    confidence=conf,
                    current_price=current_price,
                    vol_ref=vol_ref,
                    atr=atr,
                )
            )
        return preds


def build_v3_feature_data(cfg, data_dir: str) -> Dict[str, pd.DataFrame]:
    """Build indicator frames using the same V3 training pipeline."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=32),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Loading OHLCV", total=6)
        progress.advance(task)
        price_data = train_v3.load_price_data(cfg, data_dir)
        progress.update(task, description="Attaching sentiment")
        progress.advance(task)
        price_data = train_v3.attach_sentiment(price_data, cfg)
        progress.update(task, description="Market context")
        progress.advance(task)
        price_data = train_v3.add_market_context(price_data)
        if getattr(cfg.model, "use_macro_features", True):
            progress.update(task, description="Macro features")
            progress.advance(task)
            price_data = train_v3.add_yfinance_macro_context(price_data, cfg)
        else:
            progress.advance(task)
        progress.update(task, description="Stock traits")
        progress.advance(task)
        price_data = train_v3.add_static_stock_traits(price_data, {})
        progress.update(task, description="Technical features")
        progress.advance(task)
        feature_data = FeatureBuilder(horizons=cfg.features.horizons).build_features(
            price_data, is_inference=True
        )
        progress.update(task, description="Done")
        progress.advance(task)
    return feature_data


def print_portfolio_table(results: list[dict]) -> None:
    table = Table(title="Capital-aware portfolio backtest", box=box.ROUNDED)
    table.add_column("Profile", style="cyan")
    table.add_column("Capital", justify="right")
    table.add_column("Final", justify="right")
    table.add_column("Return", justify="right", style="green")
    table.add_column("Sharpe", justify="right")
    table.add_column("MaxDD", justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Win%", justify="right")

    for r in results:
        table.add_row(
            r["profile"],
            f"₹{r['initial_capital']:,.0f}",
            f"₹{r['final_value']:,.0f}",
            f"{r['total_return_pct']:+.2f}%",
            f"{r['sharpe_ratio']:.2f}",
            f"{r['max_drawdown_pct']:.1f}%",
            str(r["total_trades"]),
            f"{r['win_rate']:.1f}%",
        )
    console.print()
    console.print(table)


ROLLING_WINDOW_DEFAULT = 63
ROLLING_MIN_PERIODS = 21
INDIA_DELIVERY_COSTS = {
    "slippage_pct": 0.05,
    "stt_pct": 0.10,
    "stamp_pct": 0.015,
    "gst_pct": 18.0,
    "exchange_pct": 0.00345,
}


def build_rolling_ranks(
    predictions: List[Prediction],
    horizon: int,
    rolling_window: int = ROLLING_WINDOW_DEFAULT,
    min_periods: int = ROLLING_MIN_PERIODS,
) -> Dict[str, pd.Series]:
    """Per-symbol rolling direction accuracy (no look-ahead)."""
    rows = [
        {
            "date": pd.Timestamp(p.date),
            "symbol": p.symbol,
            "correct": int(p.is_correct),
        }
        for p in predictions
        if p.horizon == horizon and p.actual_direction
    ]
    if not rows:
        return {}

    frame = pd.DataFrame(rows).sort_values(["symbol", "date"])
    ranks: dict[str, pd.Series] = {}
    for symbol, group in frame.groupby("symbol"):
        series = group.set_index("date").sort_index()["correct"]
        ranks[symbol] = series.rolling(rolling_window, min_periods=min_periods).mean()
    return ranks


def get_top_stocks(
    rolling_ranks: Dict[str, pd.Series],
    as_of_date: str,
    n: int,
) -> List[str]:
    """Top-N symbols by rolling accuracy using data strictly before ``as_of_date``."""
    dt = pd.Timestamp(as_of_date)
    scores: dict[str, float] = {}
    for symbol, series in rolling_ranks.items():
        prior = series[series.index < dt]
        if len(prior) and not pd.isna(prior.iloc[-1]):
            scores[symbol] = float(prior.iloc[-1])
    if not scores:
        return []
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [symbol for symbol, _ in ranked[:n]]


def compute_regime_bullish(
    trading_dates: List[pd.Timestamp],
    daily_data: Dict[str, pd.DataFrame],
    symbols: List[str] | None = None,
) -> Dict[str, bool]:
    """Market regime: bullish when median 21-day return across universe is positive."""
    universe = symbols or list(daily_data.keys())
    regime: dict[str, bool] = {}
    for trade_date in trading_dates:
        date_str = trade_date.strftime("%Y-%m-%d")
        returns: list[float] = []
        for symbol in universe:
            frame = daily_data.get(symbol)
            if frame is None:
                continue
            mask = frame.index.strftime("%Y-%m-%d") == date_str
            if not mask.any():
                continue
            idx = frame.index.get_loc(frame.index[mask][-1])
            if idx >= 21:
                ret21 = frame.iloc[idx]["Close"] / frame.iloc[idx - 21]["Close"] - 1
                returns.append(float(ret21))
        regime[date_str] = float(np.median(returns)) > 0 if returns else True
    return regime


def compute_regime_scores(
    trading_dates: List[pd.Timestamp],
    daily_data: Dict[str, pd.DataFrame],
    symbols: List[str] | None = None,
) -> Dict[str, float]:
    """Median 21-day cross-stock return by date; positive means broad uptrend."""
    universe = symbols or list(daily_data.keys())
    scores: dict[str, float] = {}
    for trade_date in trading_dates:
        date_str = trade_date.strftime("%Y-%m-%d")
        returns: list[float] = []
        for symbol in universe:
            frame = daily_data.get(symbol)
            if frame is None:
                continue
            mask = frame.index.strftime("%Y-%m-%d") == date_str
            if not mask.any():
                continue
            idx = frame.index.get_loc(frame.index[mask][-1])
            if idx >= 21:
                ret21 = frame.iloc[idx]["Close"] / frame.iloc[idx - 21]["Close"] - 1
                returns.append(float(ret21))
        scores[date_str] = float(np.median(returns)) if returns else 0.0
    return scores


def compute_robust_regime_states(
    trading_dates: List[pd.Timestamp],
    daily_data: Dict[str, pd.DataFrame],
    symbols: List[str] | None = None,
    neutral_band: float = 0.01,
) -> Dict[str, str]:
    """Causal robust regime using only closes available on or before each date."""
    universe = symbols or list(daily_data.keys())
    states: dict[str, str] = {}
    for trade_date in trading_dates:
        date_str = trade_date.strftime("%Y-%m-%d")
        ret21: list[float] = []
        ret63: list[float] = []
        positive_21d = 0
        for symbol in universe:
            frame = daily_data.get(symbol)
            if frame is None:
                continue
            mask = frame.index.strftime("%Y-%m-%d") == date_str
            if not mask.any():
                continue
            idx = frame.index.get_loc(frame.index[mask][-1])
            if idx >= 21:
                r21 = frame.iloc[idx]["Close"] / frame.iloc[idx - 21]["Close"] - 1
                ret21.append(float(r21))
                positive_21d += int(r21 > 0)
            if idx >= 63:
                ret63.append(float(frame.iloc[idx]["Close"] / frame.iloc[idx - 63]["Close"] - 1))

        if not ret21:
            states[date_str] = "neutral"
            continue

        median21 = float(np.median(ret21))
        median63 = float(np.median(ret63)) if ret63 else 0.0
        breadth21 = positive_21d / len(ret21)
        medium_not_weak = median63 >= -neutral_band and breadth21 >= 0.45

        if median21 > 0 and medium_not_weak:
            states[date_str] = "bull"
        elif median21 <= -neutral_band or median63 < -neutral_band or breadth21 < 0.35:
            states[date_str] = "bear"
        else:
            states[date_str] = "neutral"
    return states


def classify_regime(score: float, neutral_band: float) -> str:
    if score < -neutral_band:
        return "bear"
    if score > neutral_band:
        return "bull"
    return "neutral"


def graded_profile(
    profile: ProfileConfig,
    regime: str,
    *,
    neutral_exposure: float,
    bear_exposure: float,
    neutral_threshold_shift: float,
    bear_threshold_shift: float,
) -> tuple[ProfileConfig, float]:
    """Return entry-only profile and size multiplier for a graded regime state."""
    if regime == "bull":
        return profile, 1.0

    exposure = neutral_exposure if regime == "neutral" else bear_exposure
    shift = neutral_threshold_shift if regime == "neutral" else bear_threshold_shift
    max_positions = 0 if exposure <= 0 else max(1, int(np.ceil(profile.max_positions * exposure)))
    adjusted = dataclasses.replace(
        profile,
        max_positions=max_positions,
        min_p_up=min(0.99, profile.min_p_up + shift),
        max_p_up=max(0.01, profile.max_p_up - shift),
        description=f"{profile.description} | {regime} graded",
    )
    return adjusted, exposure


def process_day_with_entry_profile(
    backtester: PortfolioBacktester,
    date_str: str,
    predictions: List[Prediction],
    daily_data: Dict[str, pd.DataFrame],
    next_date: str | None,
    entry_profile: ProfileConfig,
    entry_size_scale: float,
) -> None:
    """Process exits with the base profile, then entries with a temporary profile."""
    base_profile = backtester.profile
    base_entry_size_scale = backtester.entry_size_scale

    backtester._evaluate_exits(date_str, daily_data)
    backtester._check_horizon_expiry(date_str, daily_data)

    if next_date and entry_size_scale > 0 and entry_profile.max_positions > 0:
        backtester.profile = entry_profile
        backtester.entry_size_scale = entry_size_scale
        try:
            backtester._enter_trades(predictions, next_date, daily_data)
        finally:
            backtester.profile = base_profile
            backtester.entry_size_scale = base_entry_size_scale

    portfolio_value = backtester._get_portfolio_value(date_str, daily_data)
    backtester.daily_equity.append((date_str, portfolio_value))


def simulate_portfolio(
    profile: ProfileConfig,
    loader: HistoricalDataLoader,
    trading_dates: List[pd.Timestamp],
    preds_by_date: Dict[str, List[Prediction]],
    daily_data: Dict[str, pd.DataFrame],
    *,
    regime_filter: bool = False,
    regime_policy: str = "off",
    dynamic_stocks: int = 0,
    rolling_ranks: Dict[str, pd.Series] | None = None,
    regime_bullish: Dict[str, bool] | None = None,
    regime_scores: Dict[str, float] | None = None,
    regime_states: Dict[str, str] | None = None,
    regime_neutral_band: float = 0.01,
    neutral_exposure: float = 0.75,
    bear_exposure: float = 0.0,
    neutral_threshold_shift: float = 0.03,
    bear_threshold_shift: float = 0.12,
) -> Tuple[PortfolioBacktester, dict]:
    """Run portfolio simulation with optional regime and dynamic stock filters."""
    backtester = PortfolioBacktester(profile, loader)
    current_top: list[str] = []
    last_rebalance_month = ""
    skipped_regime_days = 0
    regime_counts = {"bull": 0, "neutral": 0, "bear": 0}
    selection_log: list[dict] = []
    if regime_filter and regime_policy == "off":
        regime_policy = "binary"

    for i, trade_date in enumerate(trading_dates):
        date_str = trade_date.strftime("%Y-%m-%d")
        month_key = date_str[:7]

        if dynamic_stocks > 0 and rolling_ranks is not None:
            if month_key != last_rebalance_month:
                current_top = get_top_stocks(rolling_ranks, date_str, dynamic_stocks)
                last_rebalance_month = month_key
                selection_log.append(
                    {
                        "month": month_key,
                        "as_of_date": date_str,
                        "top_stocks": ",".join(current_top),
                        "count": len(current_top),
                    }
                )

        day_preds = preds_by_date.get(date_str, [])
        if dynamic_stocks > 0 and current_top:
            day_preds = [p for p in day_preds if p.symbol in current_top]

        next_date = trading_dates[i + 1].strftime("%Y-%m-%d") if i + 1 < len(trading_dates) else None

        if regime_policy == "binary" and regime_bullish is not None and not regime_bullish.get(date_str, True):
            backtester._evaluate_exits(date_str, daily_data)
            backtester._check_horizon_expiry(date_str, daily_data)
            portfolio_value = backtester._get_portfolio_value(date_str, daily_data)
            backtester.daily_equity.append((date_str, portfolio_value))
            skipped_regime_days += 1
            continue

        if regime_policy in {"graded", "robust"} and (regime_scores is not None or regime_states is not None):
            if regime_policy == "robust" and regime_states is not None:
                regime = regime_states.get(date_str, "neutral")
            else:
                regime = classify_regime(regime_scores.get(date_str, 0.0), regime_neutral_band)
            regime_counts[regime] += 1
            entry_profile, entry_size_scale = graded_profile(
                profile,
                regime,
                neutral_exposure=neutral_exposure,
                bear_exposure=bear_exposure,
                neutral_threshold_shift=neutral_threshold_shift,
                bear_threshold_shift=bear_threshold_shift,
            )
            process_day_with_entry_profile(
                backtester,
                date_str,
                day_preds,
                daily_data,
                next_date,
                entry_profile,
                entry_size_scale,
            )
        else:
            backtester.process_day(date_str, day_preds, daily_data, next_date)

    if backtester.open_trades:
        last_date = trading_dates[-1].strftime("%Y-%m-%d")
        for trade in list(backtester.open_trades):
            close_price = loader.get_eod_price(trade.symbol, last_date)
            if close_price:
                backtester._close_trade(trade, close_price, "Backtest End", last_date)
        backtester.open_trades = []
        if backtester.daily_equity:
            backtester.daily_equity[-1] = (last_date, backtester._get_portfolio_value(last_date, daily_data))

    meta = {
        "regime_filter": regime_policy != "off",
        "regime_policy": regime_policy,
        "dynamic_stocks": dynamic_stocks,
        "regime_skipped_days": skipped_regime_days,
        "regime_counts": regime_counts,
        "stock_selection_log": selection_log,
    }
    return backtester, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="StockXpert V3 capital-aware portfolio backtest")
    parser.add_argument("--run-dir", required=True, help="Trained V3 run directory")
    parser.add_argument("--config", default=None, help="Config YAML (default: <run-dir>/config.yaml)")
    parser.add_argument("--data-dir", default="data/nifty500")
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--max-stocks", type=int, default=None)
    parser.add_argument("--hit-rate-only", action="store_true", help="Skip portfolio simulation")
    parser.add_argument("--profile", choices=list(PROFILES.keys()), default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--capital", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--min-p-up", type=float, default=None)
    parser.add_argument("--max-p-up", type=float, default=None)
    parser.add_argument("--horizon", default=None, help="Comma-separated horizons, e.g. 5,7,10")
    parser.add_argument(
        "--regime-filter",
        action="store_true",
        help="Skip new entries when median 21-day cross-stock return is negative",
    )
    parser.add_argument(
        "--regime-policy",
        choices=["off", "binary", "graded", "robust"],
        default="off",
        help="Regime handling: off, binary skip-bear entries, graded exposure, or robust causal breadth/trend policy",
    )
    parser.add_argument(
        "--regime-neutral-band",
        type=float,
        default=0.01,
        help="Neutral band for graded regime, as decimal median 21d return (default: 0.01 = +/-1%%)",
    )
    parser.add_argument("--regime-neutral-exposure", type=float, default=0.75)
    parser.add_argument("--regime-bear-exposure", type=float, default=0.0)
    parser.add_argument("--regime-neutral-threshold-shift", type=float, default=0.03)
    parser.add_argument("--regime-bear-threshold-shift", type=float, default=0.12)
    parser.add_argument(
        "--dynamic-stocks",
        type=int,
        default=0,
        help="Trade only top-N stocks by rolling hit-rate (0 = all symbols)",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=ROLLING_WINDOW_DEFAULT,
        help="Rolling window for dynamic stock ranking (trading days)",
    )
    parser.add_argument(
        "--rank-horizon",
        type=int,
        default=1,
        help="Prediction horizon used to rank stocks for dynamic selection",
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Enable production filters: regime filter + top-15 dynamic stocks",
    )
    parser.add_argument("--verbose", action="store_true")
    # A3: Transaction cost overrides (applied to all profiles)
    parser.add_argument("--brokerage-pct", type=float, default=-1.0, help="Brokerage per side %% (default: use profile default 0.03%%; pass 0.0 for zero-cost baseline)")
    parser.add_argument("--slippage-pct", type=float, default=0.0, help="Slippage per side %% of trade value")
    parser.add_argument("--stt-pct", type=float, default=0.0, help="STT %% (delivery equity: 0.10)")
    parser.add_argument("--stamp-pct", type=float, default=0.0, help="Stamp duty %% (buy side: 0.015)")
    parser.add_argument("--gst-pct", type=float, default=0.0, help="GST on brokerage %% (typical: 18)")
    parser.add_argument("--exchange-pct", type=float, default=0.0, help="NSE/SEBI exchange fee %% (typical: 0.00345)")
    parser.add_argument("--stop-atr-mult", type=float, default=None, help="Override stop ATR multiplier for custom profile")
    parser.add_argument("--target-atr-mult", type=float, default=None, help="Override target ATR multiplier for custom profile")
    parser.add_argument(
        "--allow-shorts",
        action="store_true",
        help="Allow SHORT entries when P(Up) is below the profile short threshold",
    )
    parser.add_argument(
        "--allow-multi-horizon-per-symbol",
        action="store_true",
        help="Allow multiple same-symbol entries from different horizons on the same day",
    )
    parser.add_argument(
        "--full-cost-india-delivery",
        action="store_true",
        help="Use canonical Indian cash-equity delivery costs: slippage 0.05%%/side, STT 0.10%%, stamp 0.015%% buy-side, GST 18%%, exchange 0.00345%%",
    )
    args = parser.parse_args()

    if args.full_cost_india_delivery:
        for key, value in INDIA_DELIVERY_COSTS.items():
            if getattr(args, key) == 0.0:
                setattr(args, key, value)

    if args.production:
        if args.regime_policy == "off":
            args.regime_policy = "binary"
        args.regime_filter = args.regime_policy != "off"
        if args.dynamic_stocks == 0:
            args.dynamic_stocks = 15
    elif args.regime_filter and args.regime_policy == "off":
        args.regime_policy = "binary"
    else:
        args.regime_filter = args.regime_policy != "off"

    if not args.verbose:
        logging.basicConfig(level=logging.WARNING)
        warnings.filterwarnings("ignore", category=FutureWarning)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run_dir = Path(args.run_dir)
    cfg_path = Path(args.config) if args.config else run_dir / "config.yaml"
    cfg = load_v3_config(cfg_path)
    set_seed(cfg.run.seed)

    filter_bits = []
    if args.regime_policy != "off":
        filter_bits.append(f"regime {args.regime_policy}")
    if args.dynamic_stocks > 0:
        filter_bits.append(f"top-{args.dynamic_stocks} rolling stocks")
    filter_line = ", ".join(filter_bits) if filter_bits else "no regime/stock filters"

    console.print(
        Panel(
            "\n".join(
                [
                    f"[bold]Run[/bold]     {run_dir.name}",
                    f"[bold]Window[/bold]  {args.start} → {args.end}",
                    f"[bold]Mode[/bold]    Capital-aware portfolio simulation (V3)",
                    f"[bold]Filters[/bold] {filter_line}",
                ]
            ),
            title="StockXpert V3 Backtest",
            border_style="blue",
        )
    )

    loader = HistoricalDataLoader(args.data_dir)
    inference = V3ModelInference(run_dir, cfg_path)

    symbols = list(cfg.data.symbols)
    if args.max_stocks:
        symbols = symbols[: args.max_stocks]
    available = loader.get_available_model_symbols(symbols)
    if not available:
        raise RuntimeError("No symbols with local data found.")

    if args.max_stocks:
        cfg.data.symbols = available[: args.max_stocks]
        available = cfg.data.symbols

    # Extend cfg.data.end to args.end so feature data covers the full backtest window.
    # load_price_data clips to cfg.data.end; without this, predictions past the training
    # config end date are silently dropped.
    if args.end > cfg.data.end:
        cfg.data.end = args.end

    feature_data = build_v3_feature_data(cfg, args.data_dir)
    feature_data = {sym: df for sym, df in feature_data.items() if sym in available}
    inference.set_stock_traits(feature_data)

    daily_data = {
        sym: df[["Open", "High", "Low", "Close", "Volume"]].copy()
        for sym, df in feature_data.items()
        if all(col in df.columns for col in ("Open", "High", "Low", "Close"))
    }

    trading_dates = loader.get_trading_dates(available[0], start=args.start, end=args.end)
    if not trading_dates:
        raise RuntimeError(f"No trading dates between {args.start} and {args.end}")

    prediction_dates = list(trading_dates)
    if args.dynamic_stocks > 0 or args.production:
        warmup_days = max(args.rolling_window + ROLLING_MIN_PERIODS, ROLLING_MIN_PERIODS)
        warmup_start = (pd.Timestamp(args.start) - pd.offsets.BDay(warmup_days)).strftime("%Y-%m-%d")
        warmup_dates = loader.get_trading_dates(available[0], start=warmup_start, end=args.start)
        if warmup_dates:
            # Exclude backtest start duplicates; keep strictly prior dates for ranking warmup.
            prior = [d for d in warmup_dates if d.strftime("%Y-%m-%d") < args.start]
            prediction_dates = prior + trading_dates
            if prior:
                console.print(
                    f"[dim]Dynamic stock filter: generating {len(prior)} warmup prediction days "
                    f"before {args.start}[/dim]"
                )

    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / f"v3_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(out_dir)

    # ── Part 1: generate predictions + hit rate ──
    all_predictions: list[Prediction] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        pred_task = progress.add_task("Generating predictions", total=len(prediction_dates))
        for trade_date in prediction_dates:
            date_str = trade_date.strftime("%Y-%m-%d")
            all_predictions.extend(inference.predict_for_date(feature_data, date_str))
            progress.advance(pred_task)

    evaluator = HitRateEvaluator(loader)
    all_predictions = evaluator.fill_actuals(all_predictions, daily_data)
    metrics = evaluator.compute_metrics(all_predictions)
    if metrics:
        print_hit_rate_report(metrics, out_dir)

    pred_rows = [
        {
            "date": p.date,
            "symbol": p.symbol,
            "horizon": p.horizon,
            "p_up": p.p_up,
            "predicted_dir": p.predicted_direction,
            "confidence": p.confidence,
            "current_price": p.current_price,
            "actual_return_pct": p.actual_return_pct,
            "actual_dir": p.actual_direction,
            "is_correct": p.is_correct,
        }
        for p in all_predictions
    ]
    pd.DataFrame(pred_rows).to_csv(out_dir / "predictions.csv", index=False)

    all_results: list[dict] = []
    if not args.hit_rate_only:
        if args.profile:
            profiles_to_run = {args.profile: PROFILES[args.profile]}
        elif args.capital or args.max_positions or args.min_p_up:
            custom_horizons = [int(h.strip()) for h in args.horizon.split(",")] if args.horizon else list(cfg.features.horizons)
            profiles_to_run = {
                "custom": ProfileConfig(
                    name="Custom",
                    capital=args.capital or 500_000,
                    max_positions=args.max_positions or 5,
                    min_p_up=args.min_p_up or 0.58,
                    max_p_up=args.max_p_up or 0.42,
                    horizons=custom_horizons,
                    stop_atr_mult=args.stop_atr_mult or 1.5,
                    target_atr_mult=args.target_atr_mult or 2.0,
                    description="CLI overrides",
                )
            }
        else:
            profiles_to_run = PROFILES

        profiles_to_run = {
            key: dataclasses.replace(
                profile,
                allow_shorts=args.allow_shorts,
                one_position_per_symbol=not args.allow_multi_horizon_per_symbol,
            )
            for key, profile in profiles_to_run.items()
        }

        preds_by_date: dict[str, list[Prediction]] = {}
        for pred in all_predictions:
            preds_by_date.setdefault(pred.date, []).append(pred)

        rolling_ranks = None
        if args.dynamic_stocks > 0:
            rolling_ranks = build_rolling_ranks(
                all_predictions,
                horizon=args.rank_horizon,
                rolling_window=args.rolling_window,
            )
            if not rolling_ranks:
                console.print(
                    "[yellow]Warning:[/yellow] dynamic stock filter enabled but no rolling ranks "
                    f"for horizon {args.rank_horizon}; trading all symbols."
                )

        regime_bullish = None
        regime_scores = None
        regime_states = None
        if args.regime_policy == "binary":
            regime_bullish = compute_regime_bullish(trading_dates, daily_data, available)
        elif args.regime_policy == "graded":
            regime_scores = compute_regime_scores(trading_dates, daily_data, available)
        elif args.regime_policy == "robust":
            regime_states = compute_robust_regime_states(
                trading_dates, daily_data, available, neutral_band=args.regime_neutral_band
            )

        # Apply cost overrides to every profile
        cost_overrides = {k: v for k, v in {
            "slippage_pct": args.slippage_pct,
            "stt_pct": args.stt_pct,
            "stamp_pct": args.stamp_pct,
            "gst_pct": args.gst_pct,
            "exchange_pct": args.exchange_pct,
        }.items() if v != 0.0}
        # brokerage_pct: -1.0 means "use profile default"; any other value (including 0.0) overrides
        if args.brokerage_pct >= 0.0:
            cost_overrides["brokerage_pct"] = args.brokerage_pct
        if cost_overrides:
            profiles_to_run = {k: dataclasses.replace(p, **cost_overrides) for k, p in profiles_to_run.items()}

        for profile_name, profile in profiles_to_run.items():
            console.print(f"[bold]Simulating profile[/bold] {profile.name} — {profile.description}")
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[bold green]{{task.description}} ({profile.name})"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                sim_task = progress.add_task("Portfolio simulation", total=len(trading_dates))
                backtester, sim_meta = simulate_portfolio(
                    profile,
                    loader,
                    trading_dates,
                    preds_by_date,
                    daily_data,
                    regime_filter=args.regime_filter,
                    regime_policy=args.regime_policy,
                    dynamic_stocks=args.dynamic_stocks if rolling_ranks else 0,
                    rolling_ranks=rolling_ranks,
                    regime_bullish=regime_bullish,
                    regime_scores=regime_scores,
                    regime_states=regime_states,
                    regime_neutral_band=args.regime_neutral_band,
                    neutral_exposure=args.regime_neutral_exposure,
                    bear_exposure=args.regime_bear_exposure,
                    neutral_threshold_shift=args.regime_neutral_threshold_shift,
                    bear_threshold_shift=args.regime_bear_threshold_shift,
                )
                progress.update(sim_task, completed=len(trading_dates))

            results = backtester.get_results()
            results["regime_filter"] = sim_meta["regime_filter"]
            results["regime_policy"] = sim_meta["regime_policy"]
            results["dynamic_stocks"] = sim_meta["dynamic_stocks"]
            results["regime_skipped_days"] = sim_meta["regime_skipped_days"]
            results["regime_counts"] = sim_meta["regime_counts"]
            all_results.append(results)

            if sim_meta["stock_selection_log"]:
                pd.DataFrame(sim_meta["stock_selection_log"]).to_csv(
                    out_dir / f"stock_selection_{profile_name}.csv", index=False
                )

            if backtester.closed_trades:
                pd.DataFrame(
                    [
                        {
                            "Symbol": t.symbol,
                            "Side": t.side,
                            "Horizon": t.horizon,
                            "P_Up": t.p_up,
                            "Entry_Date": t.entry_date,
                            "Entry_Price": t.entry_price,
                            "Exit_Date": t.exit_date,
                            "Exit_Price": t.exit_price,
                            "Exit_Reason": t.exit_reason,
                            "Entry_Cost": t.entry_cost,
                            "Exit_Cost": t.exit_cost,
                            "Gross_PnL": t.gross_pnl,
                            "Net_PnL": t.net_pnl,
                            "Accounting_Version": t.accounting_version,
                            "PnL": t.pnl,
                            "PnL_Pct": t.pnl_pct,
                        }
                        for t in backtester.closed_trades
                    ]
                ).to_csv(out_dir / f"trade_log_{profile_name}.csv", index=False)

            if backtester.daily_equity:
                pd.DataFrame(backtester.daily_equity, columns=["Date", "Portfolio_Value"]).to_csv(
                    out_dir / f"equity_curve_{profile_name}.csv", index=False
                )

        print_profile_results(all_results, out_dir)
        print_portfolio_table(all_results)

    with (out_dir / "summary.json").open("w") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "start": args.start,
                "end": args.end,
                "predictions": len(all_predictions),
                "regime_filter": args.regime_filter,
                "regime_policy": args.regime_policy,
                "regime_neutral_band": args.regime_neutral_band,
                "regime_neutral_exposure": args.regime_neutral_exposure,
                "regime_bear_exposure": args.regime_bear_exposure,
                "regime_neutral_threshold_shift": args.regime_neutral_threshold_shift,
                "regime_bear_threshold_shift": args.regime_bear_threshold_shift,
                "dynamic_stocks": args.dynamic_stocks,
                "rolling_window": args.rolling_window,
                "rank_horizon": args.rank_horizon,
                "full_cost_india_delivery": args.full_cost_india_delivery,
                "brokerage_pct_override": args.brokerage_pct if args.brokerage_pct >= 0.0 else None,
                "profiles": all_results,
            },
            f,
            indent=2,
        )

    if metrics:
        save_markdown_report(metrics, all_results, out_dir, f"{args.start} to {args.end}")

    console.print(f"\n[bold green]Done[/bold green] — outputs in [link={out_dir}]{out_dir}[/link]")


if __name__ == "__main__":
    main()
