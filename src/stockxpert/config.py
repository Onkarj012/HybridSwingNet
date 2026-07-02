"""
Configuration loading and validation for StockXpert.
"""

import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path


@dataclass
class RunConfig:
    name: str
    out_dir: str
    seed: int = 42
    device: str = "cuda"


@dataclass
class DataConfig:
    symbols: List[str] = field(default_factory=list)
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    auto_adjust: bool = True
    cache_dir: str = ""


@dataclass
class SplitsConfig:
    train_end: str = "2022-12-31"
    val_end: str = "2023-12-31"
    test_end: str = "2024-12-31"


@dataclass
class GDELTConfig:
    max_records_per_query: int = 250
    sleep_seconds: float = 1.0
    query_map: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class SentimentConfig:
    mode: str = "none"  # none | gdelt_finbert | csv_sentiment
    language: str = "English"
    tz: str = "Asia/Kolkata"
    market_close_time: str = "15:30:00"
    gdelt: GDELTConfig = field(default_factory=GDELTConfig)
    csv_path: str = "data/sentiment/final_news_sentiment_analysis.csv"
    cache_dir: str = ""


@dataclass
class WindowsConfig:
    short: int = 7
    mid: int = 21
    long: int = 60


@dataclass
class FeaturesConfig:
    horizons: List[int] = field(default_factory=lambda: [1, 3, 5, 7, 10])
    windows: WindowsConfig = field(default_factory=WindowsConfig)
    short_features: List[str] = field(default_factory=list)
    mid_features: List[str] = field(default_factory=list)
    long_features: List[str] = field(default_factory=list)
    context_features: List[str] = field(default_factory=list)
    cache_dir: str = ""


@dataclass
class ModelConfig:
    type: str = "stockxpert"
    hidden_dim: int = 64
    stock_embed_dim: int = 16
    attn_heads: int = 4
    dropout: float = 0.15


@dataclass
class LossWeightsConfig:
    direction: float = 0.5
    magnitude: float = 0.4
    variance: float = 0.1
    zone: float = 0.3
    confidence: float = 0.2
    aux_trend: float = 0.5
    magnitude_symmetry: float = 0.1


@dataclass
class DirectionLossConfig:
    """Phase 2: Direction-aware loss parameters."""
    gamma_pos: float = 1.5
    gamma_neg: float = 1.5
    downside_weight: float = 1.0


@dataclass
class OversamplingConfig:
    enabled: bool = False
    alpha: float = 2.0
    min_weight: float = 1.0
    max_weight: float = 20.0


@dataclass
class LRSchedulerConfig:
    type: str = "ReduceLROnPlateau"
    patience: int = 3
    factor: float = 0.5


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 0.0002
    weight_decay: float = 0.0001
    grad_clip: float = 1.0
    early_stopping_patience: int = 15
    loss_weights: LossWeightsConfig = field(default_factory=LossWeightsConfig)
    oversampling: OversamplingConfig = field(default_factory=OversamplingConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    horizon_weights: List[float] = field(default_factory=list)  # Phase 2
    direction_loss: DirectionLossConfig = field(default_factory=DirectionLossConfig)  # Phase 2


@dataclass
class StrategyConfig:
    horizon: int = 1
    p_up_threshold: float = 0.55
    transaction_cost_bps: float = 10


@dataclass
class ReportingConfig:
    make_plots: bool = True
    equity_curve: bool = True
    strategy: StrategyConfig = field(default_factory=StrategyConfig)


@dataclass
class Config:
    """Main configuration container."""
    run: RunConfig
    data: DataConfig
    splits: SplitsConfig
    sentiment: SentimentConfig
    features: FeaturesConfig
    model: ModelConfig
    train: TrainConfig
    reporting: ReportingConfig


def load_config(config_path: str) -> Config:
    """Load and parse YAML configuration file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Parse nested configurations
    run_cfg = RunConfig(**data.get('run', {}))
    data_cfg = DataConfig(**data.get('data', {}))
    splits_cfg = SplitsConfig(**data.get('splits', {}))
    
    # Sentiment config with nested GDELT
    sentiment_data = data.get('sentiment', {})
    gdelt_cfg = GDELTConfig(**sentiment_data.get('gdelt', {}))
    sentiment_cfg = SentimentConfig(
        mode=sentiment_data.get('mode', 'none'),
        language=sentiment_data.get('language', 'English'),
        tz=sentiment_data.get('tz', 'Asia/Kolkata'),
        market_close_time=sentiment_data.get('market_close_time', '15:30:00'),
        gdelt=gdelt_cfg,
        csv_path=sentiment_data.get('csv_path', 'final_news_sentiment_analysis.csv'),
        cache_dir=sentiment_data.get('cache_dir', "")
    )
    
    # Features config with nested windows
    features_data = data.get('features', {})
    windows_cfg = WindowsConfig(**features_data.get('windows', {}))
    features_cfg = FeaturesConfig(
        horizons=features_data.get('horizons', [1, 3, 5, 7, 10]),
        windows=windows_cfg,
        short_features=features_data.get('short_features', []),
        mid_features=features_data.get('mid_features', []),
        long_features=features_data.get('long_features', []),
        context_features=features_data.get('context_features', []),
        cache_dir=features_data.get('cache_dir', "")
    )
    
    model_cfg = ModelConfig(**data.get('model', {}))
    
    # Train config with nested loss weights, oversampling, and scheduler
    train_data = data.get('train', {})
    
    loss_weights_cfg = LossWeightsConfig(**train_data.get('loss_weights', {}))
    oversampling_cfg = OversamplingConfig(**train_data.get('oversampling', {}))
    lr_scheduler_cfg = LRSchedulerConfig(**train_data.get('lr_scheduler', {}))
    
    # Phase 2: Direction loss config
    direction_loss_cfg = DirectionLossConfig(**train_data.get('direction_loss', {}))
    
    train_cfg = TrainConfig(
        epochs=train_data.get('epochs', 50),
        batch_size=train_data.get('batch_size', 128),
        lr=train_data.get('lr', 0.0002),
        weight_decay=train_data.get('weight_decay', 0.0001),
        grad_clip=train_data.get('grad_clip', 1.0),
        early_stopping_patience=train_data.get('early_stopping_patience', 15),
        loss_weights=loss_weights_cfg,
        oversampling=oversampling_cfg,
        lr_scheduler=lr_scheduler_cfg,
        horizon_weights=train_data.get('horizon_weights', []),
        direction_loss=direction_loss_cfg
    )
    
    # Reporting config with nested strategy
    reporting_data = data.get('reporting', {})
    strategy_cfg = StrategyConfig(**reporting_data.get('strategy', {}))
    reporting_cfg = ReportingConfig(
        make_plots=reporting_data.get('make_plots', True),
        equity_curve=reporting_data.get('equity_curve', True),
        strategy=strategy_cfg
    )
    
    return Config(
        run=run_cfg,
        data=data_cfg,
        splits=splits_cfg,
        sentiment=sentiment_cfg,
        features=features_cfg,
        model=model_cfg,
        train=train_cfg,
        reporting=reporting_cfg
    )
