#!/usr/bin/env python3
"""Generate V3 confidence-threshold metrics for an existing checkpoint."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

import train_v3
from stockxpert.dataset.scaling import ScalerGroup
from stockxpert.features.builder import FeatureBuilder
from stockxpert.utils import ensure_dir, get_device, set_seed
from stockxpert.v3.config import load_v3_config
from stockxpert.v3.dataset.dataloaders import StockXpertV3Dataset
from stockxpert.v3.dataset.make_samples import SampleBuilder
from stockxpert.v3.models.stockxpert import StockXpertModelV3


def build_loaders(cfg, data_dir: str):
    price_data = train_v3.load_price_data(cfg, data_dir)
    price_data = train_v3.attach_sentiment(price_data, cfg)
    price_data = train_v3.add_market_context(price_data)
    if getattr(cfg.model, "use_macro_features", True):
        price_data = train_v3.add_yfinance_macro_context(price_data, cfg)
    price_data = train_v3.add_static_stock_traits(price_data, {})

    feature_data = FeatureBuilder(horizons=cfg.features.horizons).build_features(price_data)
    symbol_map = {symbol: idx for idx, symbol in enumerate(cfg.data.symbols)}
    sample_builder = SampleBuilder(
        window_short=cfg.features.windows.short,
        window_mid=cfg.features.windows.mid,
        window_long=cfg.features.windows.long,
        short_features=cfg.features.short_features,
        mid_features=cfg.features.mid_features,
        long_features=cfg.features.long_features,
        context_features=cfg.features.context_features,
        sentiment_features=train_v3.SENTIMENT_FEATURES,
        horizons=cfg.features.horizons,
    )
    samples = sample_builder.build_samples(feature_data, symbol_map)
    split = train_v3.split_samples(samples, cfg)

    scaler = ScalerGroup()
    scaler.fit(split["train"])
    scaled = {key: scaler.transform(value) if value else [] for key, value in split.items()}

    loaders = {}
    for key in ["val", "test", "blind_holdout"]:
        if scaled[key]:
            loaders[key] = DataLoader(
                StockXpertV3Dataset(scaled[key], cfg.model.stock_traits_dim),
                batch_size=cfg.train.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
    return loaders, scaled["train"][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default=None, help="Defaults to <run-dir>/config.yaml")
    parser.add_argument("--data-dir", default="data/nifty500")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg = load_v3_config(args.config or run_dir / "config.yaml")
    set_seed(cfg.run.seed)
    device = get_device(getattr(cfg.run, "device", "auto"))

    loaders, sample = build_loaders(cfg, args.data_dir)
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

    checkpoint = torch.load(run_dir / "checkpoints" / "model_final.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    calibrators = train_v3.fit_calibrators(model, loaders["val"], device, cfg.calibration.method, cfg.features.horizons)
    ensure_dir(run_dir / "artifacts")
    with (run_dir / "artifacts" / "calibrators_all_horizons.pkl").open("wb") as f:
        pickle.dump(calibrators, f)

    ensure_dir(run_dir / "reports")
    for split_name, loader in [("test", loaders["test"]), ("blind", loaders.get("blind_holdout"))]:
        if loader is None:
            continue
        probs, labels, logits = train_v3.collect_horizon_predictions(model, loader, device)
        calibrated = train_v3.apply_calibrators(probs, logits, calibrators, cfg.features.horizons)
        report = train_v3.confidence_threshold_metrics(calibrated, labels, cfg.features.horizons)
        out = run_dir / "reports" / f"{split_name}_confidence_threshold_metrics.csv"
        report.to_csv(out, index=False)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

