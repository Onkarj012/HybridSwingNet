#!/usr/bin/env python3
"""End-to-end V3 training pipeline for StockXpert."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from historical_data_loader import HistoricalDataLoader
from stockxpert.logging_utils import setup_logging
from stockxpert.mdata.csv_sentiment import CSVSentimentClient
from stockxpert.mdata.prices import download_prices
from stockxpert.mdata.sentiment import merge_sentiment_with_prices
from stockxpert.mdata.trading_calendar import TradingCalendar
from stockxpert.features.builder import FeatureBuilder
from stockxpert.dataset.scaling import ScalerGroup
from stockxpert.reporting.metrics import compute_and_save_metrics
from stockxpert.training.checkpointing import save_checkpoint
from stockxpert.utils import Timer, ensure_dir, get_device, set_seed
from stockxpert.v3.config import load_v3_config
from stockxpert.v3.dataset.dataloaders import StockXpertV3Dataset
from stockxpert.v3.dataset.make_samples import SampleBuilder
from stockxpert.v3.dataset.splits import cross_sectional_split, time_split
from stockxpert.v3.live.model_registry import ModelRegistry, ModelRegistryRecord, current_git_commit, sha256_json
from stockxpert.v3.models.calibration import CalibrationEvaluator, IsotonicCalibrator, TemperatureScaling
from stockxpert.v3.models.losses import StockXpertV3Loss
from stockxpert.v3.models.stockxpert import StockXpertModelV3
from stockxpert.v3.reporting.v3_evaluator import V3Evaluator
from stockxpert.v3.training.eval_loop import evaluate
from stockxpert.v3.training.train_loop import Trainer

logger = logging.getLogger("stockxpert.train_v3")

SENTIMENT_FEATURES = [
    "sentiment_mean",
    "sentiment_count",
    "sentiment_std",
    "sentiment_rolling_5d",
    "sentiment_momentum",
    "sentiment_spike",
    "sentiment_rolling_10d",
    "sentiment_rolling_21d",
    "sentiment_trend",
]

YFINANCE_MACRO_TICKERS = {
    "macro_nifty": "^NSEI",
    "macro_banknifty": "^NSEBANK",
    "macro_usdinr": "INR=X",
    "macro_crude": "CL=F",
    "macro_sp500": "^GSPC",
}


def load_price_data(cfg, data_dir: str | None):
    if data_dir:
        loader = HistoricalDataLoader(data_dir)
        symbols = loader.get_available_model_symbols(cfg.data.symbols)
        if not symbols:
            raise RuntimeError(f"No symbols from config found under {data_dir}")
        price_data = loader.get_daily_data_batch(symbols, start=cfg.data.start, end=cfg.data.end)
        if not price_data:
            raise RuntimeError("Failed to load any daily price data from local minute CSVs")
        return normalize_price_frames(price_data)

    cache_dir = Path(cfg.data.cache_dir) if cfg.data.cache_dir else project_root / "runs" / "cache" / "prices"
    ensure_dir(cache_dir)
    return download_prices(
        cfg.data.symbols,
        cfg.data.start,
        cfg.data.end,
        cache_dir,
        auto_adjust=cfg.data.auto_adjust,
    )


def normalize_price_frames(price_data):
    """Match the yfinance frame shape expected by sentiment/feature utilities."""
    normalized = {}
    for symbol, df in price_data.items():
        frame = df.copy()
        if "Date" in frame.columns:
            frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None)
            frame = frame.set_index("Date")
        else:
            frame.index = pd.to_datetime(frame.index)
            if frame.index.tz is not None:
                frame.index = frame.index.tz_localize(None)
            frame.index.name = "Date"
        price_cols = ["Open", "High", "Low", "Close"]
        frame[price_cols] = frame[price_cols].apply(pd.to_numeric, errors="coerce")
        if "Volume" in frame.columns:
            frame["Volume"] = pd.to_numeric(frame["Volume"], errors="coerce").fillna(0)

        before = len(frame)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(subset=price_cols)
        frame = frame[(frame[price_cols] > 0).all(axis=1)]
        dropped = before - len(frame)
        if dropped:
            logger.info("Dropped %d invalid OHLC rows for %s before feature engineering", dropped, symbol)
        normalized[symbol] = frame
    return normalized


def add_market_context(price_data):
    """Add point-in-time universe market context used by the V3 context encoder."""
    close = pd.DataFrame({symbol: df["Close"] for symbol, df in price_data.items()}).sort_index()
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)

    market_return_1d = returns.mean(axis=1).fillna(0.0)
    market_context = pd.DataFrame(index=close.index)
    market_context["market_return_1d"] = market_return_1d
    market_context["market_return_5d"] = market_return_1d.rolling(5, min_periods=1).sum().fillna(0.0)
    market_context["market_vol_21d"] = market_return_1d.rolling(21, min_periods=2).std().fillna(0.0)
    market_context["market_trend_60d"] = market_return_1d.rolling(60, min_periods=5).mean().fillna(0.0)
    market_context["market_breadth_5d"] = (returns > 0).mean(axis=1).rolling(5, min_periods=1).mean().fillna(0.5)

    enriched = {}
    for symbol, df in price_data.items():
        frame = df.join(market_context, how="left")
        context_cols = list(market_context.columns)
        frame[context_cols] = frame[context_cols].ffill().bfill().fillna(0.0)
        enriched[symbol] = frame
    return enriched


def fetch_yfinance_macro_context(index, start: str, end: str):
    """Fetch daily macro market proxies from yfinance.

    Minute data remains local; these are daily exogenous context series suitable
    for end-of-day swing decisions.
    """
    try:
        import yfinance as yf
    except Exception as exc:
        logger.warning("yfinance unavailable; macro context disabled: %s", exc)
        return pd.DataFrame(index=index)

    macro = pd.DataFrame(index=index)
    for prefix, ticker in YFINANCE_MACRO_TICKERS.items():
        expected_cols = [f"{prefix}_ret_1d", f"{prefix}_ret_5d", f"{prefix}_vol_21d"]
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if data.empty or "Close" not in data:
                logger.warning("No yfinance macro data for %s (%s)", prefix, ticker)
                for col in expected_cols:
                    macro[col] = 0.0
                continue
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.index = pd.to_datetime(close.index).tz_localize(None)
            close = close.reindex(index).ffill().bfill()
            ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            macro[f"{prefix}_ret_1d"] = ret
            macro[f"{prefix}_ret_5d"] = ret.rolling(5, min_periods=1).sum().fillna(0.0)
            macro[f"{prefix}_vol_21d"] = ret.rolling(21, min_periods=2).std().fillna(0.0)
        except Exception as exc:
            logger.warning("Failed to fetch macro ticker %s (%s): %s", ticker, prefix, exc)
            for col in expected_cols:
                macro[col] = 0.0
    return macro


def add_yfinance_macro_context(price_data, cfg):
    index = pd.DataFrame({symbol: df["Close"] for symbol, df in price_data.items()}).sort_index().index
    macro = fetch_yfinance_macro_context(index, cfg.data.start, cfg.data.end)
    if macro.empty:
        return price_data

    enriched = {}
    for symbol, df in price_data.items():
        frame = df.join(macro, how="left")
        frame[macro.columns] = frame[macro.columns].ffill().bfill().fillna(0.0)
        enriched[symbol] = frame
    return enriched


def load_sector_map(path: str | None):
    if not path:
        return {}
    sector_path = Path(path)
    if not sector_path.exists():
        logger.warning("Sector map not found at %s; sector traits default to zero", sector_path)
        return {}
    df = pd.read_csv(sector_path)
    symbol_col = "symbol" if "symbol" in df.columns else df.columns[0]
    sector_col = "sector" if "sector" in df.columns else df.columns[1]
    return dict(zip(df[symbol_col].astype(str), df[sector_col].astype(str)))


def add_static_stock_traits(price_data, sector_map: dict[str, str] | None = None):
    """Add repeated per-symbol stock trait columns consumed by StockXpert V3."""
    sector_map = sector_map or {}
    liquidity = {}
    volatility = {}
    beta = {}

    market_return = pd.DataFrame(
        {symbol: df["Close"].pct_change() for symbol, df in price_data.items()}
    ).mean(axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    market_var = float(market_return.var()) or 1e-8

    for symbol, df in price_data.items():
        returns = df["Close"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        dollar_volume = (df["Close"] * df.get("Volume", 0)).replace([np.inf, -np.inf], np.nan)
        liquidity[symbol] = float(dollar_volume.median()) if not dollar_volume.empty else 0.0
        volatility[symbol] = float(returns.rolling(21, min_periods=5).std().median() or 0.0)
        aligned = pd.concat([returns, market_return], axis=1).dropna()
        beta[symbol] = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / market_var) if len(aligned) > 2 else 1.0

    liquidity_rank = pd.Series(liquidity).rank(pct=True).fillna(0.5)
    volatility_rank = pd.Series(volatility).rank(pct=True).fillna(0.5)
    sectors = sorted({sector for sector in sector_map.values() if sector})
    sector_ids = {sector: idx + 1 for idx, sector in enumerate(sectors)}

    enriched = {}
    for symbol, df in price_data.items():
        frame = df.copy()
        sector_id = sector_ids.get(sector_map.get(symbol, ""), 0)
        frame["sector_enc_0"] = float((sector_id >> 0) & 1)
        frame["sector_enc_1"] = float((sector_id >> 1) & 1)
        frame["sector_enc_2"] = float((sector_id >> 2) & 1)
        frame["mcap_rank"] = float(liquidity_rank.get(symbol, 0.5))
        frame["beta"] = float(np.clip(beta.get(symbol, 1.0), -3.0, 3.0))
        frame["vol_pct"] = float(volatility_rank.get(symbol, 0.5))
        frame["liquidity_tier"] = float(liquidity_rank.get(symbol, 0.5))
        enriched[symbol] = frame
    return enriched


def attach_sentiment(price_data, cfg):
    if cfg.sentiment.mode != "csv_sentiment":
        for symbol in price_data:
            price_data[symbol]["sentiment_mean"] = 0.0
            price_data[symbol]["sentiment_count"] = 0
        return price_data

    csv_path = Path(cfg.sentiment.csv_path)
    if not csv_path.exists():
        logger.warning("Sentiment CSV not found at %s; using neutral sentiment", csv_path)
        for symbol in price_data:
            price_data[symbol]["sentiment_mean"] = 0.0
            price_data[symbol]["sentiment_count"] = 0
        return price_data

    csv_client = CSVSentimentClient(str(csv_path))
    calendar = TradingCalendar(price_data)
    start_dt = datetime.strptime(cfg.data.start, "%Y-%m-%d")
    end_dt = datetime.strptime(cfg.data.end, "%Y-%m-%d")
    sentiment_df = csv_client.fetch_for_symbols(list(price_data.keys()), start_dt, end_dt, calendar)
    if sentiment_df.empty:
        logger.warning("No sentiment rows loaded; using neutral sentiment")
        for symbol in price_data:
            price_data[symbol]["sentiment_mean"] = 0.0
            price_data[symbol]["sentiment_count"] = 0
        return price_data

    return merge_sentiment_with_prices(price_data, sentiment_df)


def split_samples(all_samples, cfg):
    if cfg.splits.split_mode == "cross_sectional":
        return cross_sectional_split(
            all_samples,
            cfg.data.symbols,
            cfg.splits.train_end,
            cfg.splits.val_end,
            cfg.splits.test_end,
            val_stocks=cfg.splits.val_stocks,
            test_stocks=cfg.splits.test_stocks,
            blind_holdout_stocks=cfg.splits.blind_holdout_stocks,
            embargo_days=cfg.splits.embargo_days,
            seed=cfg.splits.seed,
        )

    train, val, test = time_split(
        all_samples,
        cfg.splits.train_end,
        cfg.splits.val_end,
        cfg.splits.test_end,
        embargo_days=cfg.splits.embargo_days,
    )
    return {"train": train, "val": val, "test": test, "blind_holdout": []}


def collect_predictions(model, dataloader, device):
    model.eval()
    probs = []
    labels = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            stock_traits = inputs["stock_traits"].to(device)
            outputs = model(
                inputs["X_short"].to(device),
                inputs["X_mid"].to(device),
                inputs["X_long"].to(device),
                inputs["X_context"].to(device),
                inputs["X_sentiment"].to(device),
                stock_traits,
            )
            direction_logits = outputs[0]
            mag_targets = targets["magnitude"]
            batch_probs = torch.sigmoid(direction_logits)[:, 0].cpu().numpy()
            batch_labels = (mag_targets[:, 0] > 0).cpu().numpy().astype(int)
            probs.extend(batch_probs.tolist())
            labels.extend(batch_labels.tolist())
    return np.asarray(probs), np.asarray(labels)


def collect_horizon_predictions(model, dataloader, device):
    model.eval()
    probs = []
    labels = []
    logits = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(
                inputs["X_short"].to(device),
                inputs["X_mid"].to(device),
                inputs["X_long"].to(device),
                inputs["X_context"].to(device),
                inputs["X_sentiment"].to(device),
                inputs["stock_traits"].to(device),
            )
            direction_logits = outputs[0]
            mag_targets = targets["magnitude"]
            logits.append(direction_logits.cpu().numpy())
            probs.append(torch.sigmoid(direction_logits).cpu().numpy())
            labels.append((mag_targets > 0).cpu().numpy().astype(int))
    return np.concatenate(probs, axis=0), np.concatenate(labels, axis=0), np.concatenate(logits, axis=0)


def fit_calibrators(model, val_loader, device, method: str, horizons: list[int]):
    probs, labels, logits = collect_horizon_predictions(model, val_loader, device)
    calibrators = {}
    calibration_rows = []
    methods = ["isotonic", "temperature"] if method == "best" else [method]
    for idx, horizon in enumerate(horizons):
        candidates = []
        for candidate_method in methods:
            if candidate_method == "temperature":
                calibrator = TemperatureScaling().fit(logits[:, idx], labels[:, idx])
                cal_probs = calibrator.transform(logits[:, idx])
            else:
                calibrator = IsotonicCalibrator().fit(probs[:, idx], labels[:, idx])
                cal_probs = calibrator.transform(probs[:, idx])
            cal_probs = np.clip(cal_probs, 1e-6, 1.0 - 1e-6)
            ece = CalibrationEvaluator().ece(cal_probs, labels[:, idx])
            nll = float(
                -np.mean(labels[:, idx] * np.log(cal_probs) + (1 - labels[:, idx]) * np.log(1 - cal_probs))
            )
            candidates.append((ece, nll, candidate_method, calibrator))
            calibration_rows.append(
                {"horizon": horizon, "method": candidate_method, "ece": ece, "nll": nll}
            )
        best_ece, best_nll, best_method, best_calibrator = sorted(candidates, key=lambda item: (item[0], item[1]))[0]
        calibrators[f"h{horizon}"] = best_calibrator
        calibration_rows.append(
            {"horizon": horizon, "method": "selected", "selected_method": best_method, "ece": best_ece, "nll": best_nll}
        )
    return calibrators, pd.DataFrame(calibration_rows)


def apply_calibrators(probs: np.ndarray, logits: np.ndarray, calibrators: dict, horizons: list[int]) -> np.ndarray:
    calibrated = np.zeros_like(probs, dtype=float)
    for idx, horizon in enumerate(horizons):
        calibrator = calibrators.get(f"h{horizon}")
        if calibrator is None:
            calibrated[:, idx] = probs[:, idx]
        elif isinstance(calibrator, TemperatureScaling):
            calibrated[:, idx] = calibrator.transform(logits[:, idx])
        else:
            calibrated[:, idx] = calibrator.transform(probs[:, idx])
    return calibrated


def confidence_threshold_metrics(probs: np.ndarray, labels: np.ndarray, horizons: list[int], thresholds=None) -> pd.DataFrame:
    thresholds = thresholds or [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    records = []
    confidence = np.maximum(probs, 1.0 - probs)
    predictions = (probs >= 0.5).astype(int)
    for idx, horizon in enumerate(horizons):
        correct = predictions[:, idx] == labels[:, idx]
        for threshold in thresholds:
            mask = confidence[:, idx] >= threshold
            records.append(
                {
                    "horizon": horizon,
                    "threshold": threshold,
                    "coverage": float(mask.mean()) if len(mask) else 0.0,
                    "n": int(mask.sum()),
                    "accuracy": float(correct[mask].mean()) if mask.any() else 0.0,
                    "avg_confidence": float(confidence[mask, idx].mean()) if mask.any() else 0.0,
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StockXpert V3 end-to-end")
    parser.add_argument("--config", default="configs/v3_nifty100_from_nifty500.yaml")
    parser.add_argument("--run-dir", type=str, default=None, help="Optional fixed run directory")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/nifty500",
        help="Local minute-bar directory (default: data/nifty500). Pass empty string to use yfinance cache instead.",
    )
    args = parser.parse_args()

    cfg = load_v3_config(args.config)
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(cfg.run.out_dir) / f"{timestamp}_{cfg.run.name}"
    ensure_dir(run_dir)

    setup_logging(run_dir)
    logger.info("V3 training started: %s", run_dir.name)

    set_seed(cfg.run.seed)
    device = get_device(getattr(cfg.run, "device", "auto"))
    logger.info("Using device: %s", device)

    data_dir = args.data_dir or None

    with Timer("Data Load"):
        price_data = load_price_data(cfg, data_dir)
        logger.info("Loaded price data for %d symbols", len(price_data))
        price_data = attach_sentiment(price_data, cfg)
        price_data = add_market_context(price_data)
        if getattr(cfg.model, "use_macro_features", True):
            price_data = add_yfinance_macro_context(price_data, cfg)
        sector_map = load_sector_map(getattr(cfg.data, "sector_map_path", ""))
        price_data = add_static_stock_traits(price_data, sector_map)

    with Timer("Feature Engineering"):
        feature_builder = FeatureBuilder(horizons=cfg.features.horizons)
        feature_data = feature_builder.build_features(price_data)
        logger.info("Features computed for %d symbols", len(feature_data))

    with Timer("Dataset Construction"):
        symbol_map = {symbol: idx for idx, symbol in enumerate(cfg.data.symbols)}
        sample_builder = SampleBuilder(
            window_short=cfg.features.windows.short,
            window_mid=cfg.features.windows.mid,
            window_long=cfg.features.windows.long,
            short_features=cfg.features.short_features,
            mid_features=cfg.features.mid_features,
            long_features=cfg.features.long_features,
            context_features=cfg.features.context_features,
            sentiment_features=SENTIMENT_FEATURES,
            horizons=cfg.features.horizons,
        )
        all_samples = sample_builder.build_samples(feature_data, symbol_map)
        logger.info("Built %d samples", len(all_samples))

    with Timer("Split"):
        split = split_samples(all_samples, cfg)
        train_samples = split["train"]
        val_samples = split["val"]
        test_samples = split["test"]
        blind_samples = split["blind_holdout"]
        logger.info(
            "Split sizes: train=%d val=%d test=%d blind=%d",
            len(train_samples),
            len(val_samples),
            len(test_samples),
            len(blind_samples),
        )

    split_meta = {
        "train_stocks": sorted({s.symbol for s in train_samples}),
        "val_stocks": sorted({s.symbol for s in val_samples}),
        "test_stocks": sorted({s.symbol for s in test_samples}),
        "blind_stocks": sorted({s.symbol for s in blind_samples}),
    }
    with (run_dir / "split_stocks.json").open("w") as f:
        json.dump(split_meta, f, indent=2)

    with Timer("Scaling"):
        scaler = ScalerGroup()
        scaler.fit(train_samples)
        train_scaled = scaler.transform(train_samples)
        val_scaled = scaler.transform(val_samples)
        test_scaled = scaler.transform(test_samples)
        blind_scaled = scaler.transform(blind_samples) if blind_samples else []

    train_loader = DataLoader(
        StockXpertV3Dataset(train_scaled, cfg.model.stock_traits_dim),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        StockXpertV3Dataset(val_scaled, cfg.model.stock_traits_dim),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        StockXpertV3Dataset(test_scaled, cfg.model.stock_traits_dim),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    blind_loader = (
        DataLoader(
            StockXpertV3Dataset(blind_scaled, cfg.model.stock_traits_dim),
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        if blind_scaled
        else None
    )

    sample = train_scaled[0]
    model = StockXpertModelV3(
        short_dim=sample.X_short.shape[-1],
        mid_dim=sample.X_mid.shape[-1],
        long_dim=sample.X_long.shape[-1],
        context_dim=sample.X_context.shape[-1],
        num_horizons=len(cfg.features.horizons),
        hidden_dim=cfg.model.hidden_dim,
        stock_embed_dim=cfg.model.stock_embed_dim,
        stock_traits_dim=cfg.model.stock_traits_dim,
        attn_heads=cfg.model.attn_heads,
        dropout=cfg.model.dropout,
        use_regime_head=cfg.model.use_regime_head,
    ).to(device)

    criterion = StockXpertV3Loss(
        direction_weight=cfg.train.loss_weights.direction,
        magnitude_weight=cfg.train.loss_weights.magnitude,
        regime_weight=cfg.train.loss_weights.regime,
        confidence_weight=cfg.train.loss_weights.confidence,
        aux_trend_weight=cfg.train.loss_weights.aux_trend,
        consistency_weight=cfg.train.loss_weights.consistency,
        legacy_loss_mode=cfg.train.loss_weights.legacy_loss_mode,
        horizon_weights=cfg.train.horizon_weights or None,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        grad_clip=cfg.train.grad_clip,
        early_stopping_patience=cfg.train.early_stopping_patience,
        use_amp=cfg.train.use_amp,
    )

    with Timer("Training"):
        history = trainer.train(train_loader, val_loader, cfg.train.epochs)
        history.to_csv(run_dir / "history.csv", index=False)

    with Timer("Evaluation"):
        test_metrics = evaluate(model, test_loader, device, cfg.features.horizons)
        blind_metrics = (
            evaluate(model, blind_loader, device, cfg.features.horizons) if blind_loader is not None else None
        )
        compute_and_save_metrics(test_metrics["metrics_per_horizon"], run_dir / "reports")
        with (run_dir / "reports" / "test_metrics.json").open("w") as f:
            json.dump(test_metrics, f, indent=2)
        if blind_metrics is not None:
            with (run_dir / "reports" / "blind_metrics.json").open("w") as f:
                json.dump(blind_metrics, f, indent=2)

    calibrators = {}
    if cfg.calibration.calibrate_on_val:
        with Timer("Calibration"):
            calibrators, calibration_report = fit_calibrators(
                model, val_loader, device, cfg.calibration.method, cfg.features.horizons
            )
            calibration_report.to_csv(run_dir / "reports" / "calibration_selection.csv", index=False)

    with Timer("Confidence Thresholds"):
        test_probs, test_labels, test_logits = collect_horizon_predictions(model, test_loader, device)
        test_confidence_probs = apply_calibrators(test_probs, test_logits, calibrators, cfg.features.horizons) if calibrators else test_probs
        ensure_dir(run_dir / "reports")
        confidence_threshold_metrics(test_confidence_probs, test_labels, cfg.features.horizons).to_csv(
            run_dir / "reports" / "confidence_threshold_metrics.csv", index=False
        )
        if blind_loader is not None:
            blind_probs, blind_labels, blind_logits = collect_horizon_predictions(model, blind_loader, device)
            blind_confidence_probs = (
                apply_calibrators(blind_probs, blind_logits, calibrators, cfg.features.horizons)
                if calibrators
                else blind_probs
            )
            confidence_threshold_metrics(blind_confidence_probs, blind_labels, cfg.features.horizons).to_csv(
                run_dir / "reports" / "blind_confidence_threshold_metrics.csv", index=False
            )

    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "artifacts")
    model_path = run_dir / "checkpoints" / "model_final.pt"
    save_checkpoint(model=model, optimizer=optimizer, epoch=len(history), loss=trainer.best_val_loss, path=model_path)
    scaler.save(run_dir / "artifacts" / "scalers.pkl")
    if calibrators:
        with (run_dir / "artifacts" / "calibrators.pkl").open("wb") as f:
            pickle.dump(calibrators, f)

    shutil.copy(args.config, run_dir / "config.yaml")
    with (run_dir / "config_resolved.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "run": cfg.run.__dict__,
                "data": cfg.data.__dict__,
                "splits": cfg.splits.__dict__,
                "model": cfg.model.__dict__,
                "train": {
                    **cfg.train.__dict__,
                    "loss_weights": cfg.train.loss_weights.__dict__,
                },
            },
            f,
            sort_keys=False,
        )

    evaluator = V3Evaluator()
    val_probs, val_labels = collect_predictions(model, val_loader, device)
    val_eval = evaluator.evaluate(val_probs, val_labels)
    with (run_dir / "reports" / "val_statistical_eval.json").open("w") as f:
        json.dump(
            {"metrics": val_eval.metrics, "statistical_tests": val_eval.statistical_tests},
            f,
            indent=2,
        )

    metrics_summary = {
        "val_accuracy_h1": val_eval.metrics.get("accuracy", 0.0),
        "test_accuracy_h1": test_metrics["metrics_per_horizon"][0]["direction_accuracy"],
    }
    if blind_metrics is not None:
        metrics_summary["blind_accuracy_h1"] = blind_metrics["metrics_per_horizon"][0]["direction_accuracy"]

    registry = ModelRegistry(run_dir.parent / "model_registry")
    version_id = f"v3-{run_dir.name}"
    registry.register(
        ModelRegistryRecord(
            version_id=version_id,
            git_commit=current_git_commit(),
            config_hash=sha256_json(cfg),
            dataset_hash=sha256_json({"symbols": cfg.data.symbols, "data_dir": data_dir}),
            train_stocks=split_meta["train_stocks"],
            val_stocks=split_meta["val_stocks"],
            test_stocks=split_meta["test_stocks"],
            blind_stocks=split_meta["blind_stocks"],
            metrics=metrics_summary,
            artifacts={
                "model_checkpoint": str(model_path),
                "scalers": str(run_dir / "artifacts" / "scalers.pkl"),
                "calibrators": str(run_dir / "artifacts" / "calibrators.pkl"),
            },
        )
    )

    logger.info("V3 training complete")
    logger.info("Run directory: %s", run_dir)
    logger.info("Model checkpoint: %s", model_path)


if __name__ == "__main__":
    main()
