"""Configuration for the V3 implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from stockxpert.config import (
    DataConfig,
    FeaturesConfig,
    GDELTConfig,
    ReportingConfig,
    RunConfig,
    SentimentConfig,
    StrategyConfig,
    WindowsConfig,
)


@dataclass
class V3SplitsConfig:
    train_end: str = "2023-06-30"
    val_end: str = "2024-03-31"
    test_end: str = "2024-12-31"
    split_mode: str = "cross_sectional"
    val_stocks: int = 50
    test_stocks: int = 50
    blind_holdout_stocks: int = 50
    embargo_days: int = 10
    seed: int = 42


@dataclass
class V3ModelConfig:
    type: str = "stockxpert_v3"
    hidden_dim: int = 96
    stock_embed_dim: int = 16
    stock_traits_dim: int = 7
    attn_heads: int = 4
    dropout: float = 0.15
    use_stock_traits: bool = True
    use_macro_features: bool = True
    use_regime_head: bool = False


@dataclass
class V3LossWeightsConfig:
    direction: float = 1.0
    magnitude: float = 0.4
    regime: float = 0.2
    variance: float = 0.0
    zone: float = 0.0
    confidence: float = 0.0
    aux_trend: float = 0.0
    level: float = 0.0
    consistency: float = 0.0
    magnitude_symmetry: float = 0.0
    legacy_loss_mode: bool = False


@dataclass
class V3TrainConfig:
    epochs: int = 50
    batch_size: int = 256
    lr: float = 0.0003
    weight_decay: float = 0.001
    grad_clip: float = 1.0
    early_stopping_patience: int = 10
    use_amp: bool = True
    loss_weights: V3LossWeightsConfig = field(default_factory=V3LossWeightsConfig)
    horizon_weights: list[float] = field(default_factory=list)


@dataclass
class CalibrationConfig:
    method: str = "isotonic"
    calibrate_on_val: bool = True


@dataclass
class BacktestConfig:
    sector_cap_pct: float = 0.30
    drawdown_pause_pct: float = 0.15
    max_adv_pct: float = 0.05


@dataclass
class LiveConfig:
    paper_trading_threshold: float = 0.60
    accuracy_floor: float = 0.52
    psi_threshold: float = 0.20
    ece_drift_threshold: float = 0.05


@dataclass
class V3Config:
    run: RunConfig
    data: DataConfig = field(default_factory=DataConfig)
    splits: V3SplitsConfig = field(default_factory=V3SplitsConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: V3ModelConfig = field(default_factory=V3ModelConfig)
    train: V3TrainConfig = field(default_factory=V3TrainConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    live: LiveConfig = field(default_factory=LiveConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)


def _sentiment_config(data: dict[str, Any]) -> SentimentConfig:
    gdelt_cfg = GDELTConfig(**data.get("gdelt", {}))
    return SentimentConfig(
        mode=data.get("mode", "none"),
        language=data.get("language", "English"),
        tz=data.get("tz", "Asia/Kolkata"),
        market_close_time=data.get("market_close_time", "15:30:00"),
        gdelt=gdelt_cfg,
        csv_path=data.get("csv_path", "data/sentiment/final_news_sentiment_analysis.csv"),
        cache_dir=data.get("cache_dir", ""),
    )


def _features_config(data: dict[str, Any]) -> FeaturesConfig:
    return FeaturesConfig(
        horizons=data.get("horizons", [1, 3, 5, 7, 10]),
        windows=WindowsConfig(**data.get("windows", {})),
        short_features=data.get("short_features", []),
        mid_features=data.get("mid_features", []),
        long_features=data.get("long_features", []),
        context_features=data.get("context_features", []),
        cache_dir=data.get("cache_dir", ""),
    )


def load_v3_config(config_path: str | Path) -> V3Config:
    """Load a V3 YAML config without changing the legacy config loader."""
    path = Path(config_path)
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}

    train_data = data.get("train", {})
    loss_weights = V3LossWeightsConfig(**data.get("loss_weights", train_data.get("loss_weights", {})))
    train = V3TrainConfig(
        epochs=train_data.get("epochs", 50),
        batch_size=train_data.get("batch_size", 256),
        lr=train_data.get("lr", 0.0003),
        weight_decay=train_data.get("weight_decay", 0.001),
        grad_clip=train_data.get("grad_clip", 1.0),
        early_stopping_patience=train_data.get("early_stopping_patience", train_data.get("early_stopping", 10)),
        use_amp=train_data.get("use_amp", True),
        loss_weights=loss_weights,
        horizon_weights=train_data.get("horizon_weights", []),
    )

    reporting_data = data.get("reporting", {})
    run_data = {"out_dir": "runs", **data.get("run", {})}
    return V3Config(
        run=RunConfig(**run_data),
        data=DataConfig(**data.get("data", {})),
        splits=V3SplitsConfig(**data.get("splits", {})),
        sentiment=_sentiment_config(data.get("sentiment", {})),
        features=_features_config(data.get("features", {})),
        model=V3ModelConfig(**data.get("model", {})),
        train=train,
        calibration=CalibrationConfig(**data.get("calibration", {})),
        backtest=BacktestConfig(**data.get("backtest", {})),
        live=LiveConfig(**data.get("live", {})),
        reporting=ReportingConfig(
            make_plots=reporting_data.get("make_plots", True),
            equity_curve=reporting_data.get("equity_curve", True),
            strategy=StrategyConfig(**reporting_data.get("strategy", {})),
        ),
    )

