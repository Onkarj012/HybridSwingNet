"""Configuration for the isolated V4 trading stack."""

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
class V4ExecutionProfileConfig:
    entry: str = "next_open"
    cost_model: str = "india_delivery"
    allow_shorts: bool = False


@dataclass
class V4TargetConfig:
    mode: str = "cost_adjusted_triple_barrier"
    execution_profile: V4ExecutionProfileConfig = field(default_factory=V4ExecutionProfileConfig)
    flat_cost_buffer_bps: float = 20.0
    target_atr: float = 2.5
    stop_atr: float = 1.5
    horizon: int = 7


@dataclass
class V4DataConfig(DataConfig):
    snapshot_id: str = "unset"
    as_of_end: str = ""


@dataclass
class V4FeaturesConfig(FeaturesConfig):
    point_in_time_stock_traits: bool = True


@dataclass
class V4ModelConfig:
    type: str = "stockxpert_v4"
    hidden_dim: int = 96
    stock_embed_dim: int = 16
    stock_traits_dim: int = 7
    attn_heads: int = 4
    dropout: float = 0.15
    use_macro_features: bool = True


@dataclass
class V4LossWeightsConfig:
    p_up: float = 0.5
    action: float = 1.0
    long_return: float = 0.6
    short_return: float = 0.6
    downside: float = 0.4
    confidence: float = 0.2
    aux_trend: float = 0.1


@dataclass
class V4TrainConfig:
    epochs: int = 50
    batch_size: int = 256
    lr: float = 0.0003
    weight_decay: float = 0.001
    grad_clip: float = 1.0
    early_stopping_patience: int = 10
    use_amp: bool = True
    validation_scheme: str = "nested_walk_forward"
    loss_weights: V4LossWeightsConfig = field(default_factory=V4LossWeightsConfig)


@dataclass
class V4CalibrationConfig:
    scheme: str = "horizon_regime_oof"
    method: str = "isotonic"
    min_samples: int = 100
    fallback_method: str = "global"


@dataclass
class V4PolicyConfig:
    selection_objective: str = "expected_after_cost_utility"
    min_expected_return_bps: float = 0.0
    max_positions: int = 5
    allow_shorts: bool = False
    reject_one_month_winners: bool = True
    reject_one_side_winners: bool = True


@dataclass
class V4SplitsConfig:
    train_end: str = "2023-12-31"
    val_end: str = "2024-12-31"
    test_end: str = "2025-12-31"
    embargo_days: int = 10
    seed: int = 42


@dataclass
class V4Config:
    run: RunConfig
    data: V4DataConfig = field(default_factory=V4DataConfig)
    splits: V4SplitsConfig = field(default_factory=V4SplitsConfig)
    target: V4TargetConfig = field(default_factory=V4TargetConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    features: V4FeaturesConfig = field(default_factory=V4FeaturesConfig)
    model: V4ModelConfig = field(default_factory=V4ModelConfig)
    train: V4TrainConfig = field(default_factory=V4TrainConfig)
    calibration: V4CalibrationConfig = field(default_factory=V4CalibrationConfig)
    policy: V4PolicyConfig = field(default_factory=V4PolicyConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)


def _sentiment_config(data: dict[str, Any]) -> SentimentConfig:
    return SentimentConfig(
        mode=data.get("mode", "none"),
        language=data.get("language", "English"),
        tz=data.get("tz", "Asia/Kolkata"),
        market_close_time=data.get("market_close_time", "15:30:00"),
        gdelt=GDELTConfig(**data.get("gdelt", {})),
        csv_path=data.get("csv_path", "data/sentiment/final_news_sentiment_analysis.csv"),
        cache_dir=data.get("cache_dir", ""),
    )


def _features_config(data: dict[str, Any]) -> V4FeaturesConfig:
    return V4FeaturesConfig(
        horizons=data.get("horizons", [3, 5, 7, 10]),
        windows=WindowsConfig(**data.get("windows", {})),
        short_features=data.get("short_features", []),
        mid_features=data.get("mid_features", []),
        long_features=data.get("long_features", []),
        context_features=data.get("context_features", []),
        cache_dir=data.get("cache_dir", ""),
        point_in_time_stock_traits=data.get("point_in_time_stock_traits", True),
    )


def load_v4_config(config_path: str | Path) -> V4Config:
    """Load V4 YAML without touching legacy or V3 config behavior."""
    path = Path(config_path)
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}

    target_data = data.get("target", {})
    train_data = data.get("train", {})
    reporting_data = data.get("reporting", {})
    run_data = {"out_dir": "runs", **data.get("run", {})}

    return V4Config(
        run=RunConfig(**run_data),
        data=V4DataConfig(**data.get("data", {})),
        splits=V4SplitsConfig(**data.get("splits", {})),
        target=V4TargetConfig(
            mode=target_data.get("mode", "cost_adjusted_triple_barrier"),
            execution_profile=V4ExecutionProfileConfig(**target_data.get("execution_profile", {})),
            flat_cost_buffer_bps=target_data.get("flat_cost_buffer_bps", 20.0),
            target_atr=target_data.get("target_atr", 2.5),
            stop_atr=target_data.get("stop_atr", 1.5),
            horizon=target_data.get("horizon", 7),
        ),
        sentiment=_sentiment_config(data.get("sentiment", {})),
        features=_features_config(data.get("features", {})),
        model=V4ModelConfig(**data.get("model", {})),
        train=V4TrainConfig(
            epochs=train_data.get("epochs", 50),
            batch_size=train_data.get("batch_size", 256),
            lr=train_data.get("lr", 0.0003),
            weight_decay=train_data.get("weight_decay", 0.001),
            grad_clip=train_data.get("grad_clip", 1.0),
            early_stopping_patience=train_data.get("early_stopping_patience", 10),
            use_amp=train_data.get("use_amp", True),
            validation_scheme=train_data.get("validation_scheme", "nested_walk_forward"),
            loss_weights=V4LossWeightsConfig(**train_data.get("loss_weights", {})),
        ),
        calibration=V4CalibrationConfig(**data.get("calibration", {})),
        policy=V4PolicyConfig(**data.get("policy", {})),
        reporting=ReportingConfig(
            make_plots=reporting_data.get("make_plots", True),
            equity_curve=reporting_data.get("equity_curve", True),
            strategy=StrategyConfig(**reporting_data.get("strategy", {})),
        ),
    )
