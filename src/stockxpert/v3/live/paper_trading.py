"""V3 paper-trading pipeline primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class PaperTradeConfig:
    threshold: float = 0.60
    max_positions: int = 10
    sector_cap_pct: float = 0.30
    drawdown_pause_pct: float = 0.15


@dataclass
class PaperTradingPipeline:
    """Formal daily pipeline state for V3 paper trading.

    Data refresh, feature building, inference, and execution are injected as
    callables by scripts so this module stays reusable and testable.
    """

    config: PaperTradeConfig = field(default_factory=PaperTradeConfig)
    log_dir: Path = Path("runs/paper_trading_v3")

    def __post_init__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def generate_signals(self, predictions: pd.DataFrame) -> pd.DataFrame:
        required = {"symbol", "horizon", "prob_up", "pred_magnitude"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"Missing prediction columns: {sorted(missing)}")
        signals = predictions[predictions["prob_up"] >= self.config.threshold].copy()
        return signals.sort_values(["prob_up", "pred_magnitude"], ascending=False)

    def apply_risk_management(self, signals: pd.DataFrame, equity: float, current_drawdown: float = 0.0) -> pd.DataFrame:
        if current_drawdown >= self.config.drawdown_pause_pct:
            return signals.iloc[0:0].copy()
        selected = []
        sector_counts: dict[str, int] = {}
        max_sector_positions = max(1, int(self.config.max_positions * self.config.sector_cap_pct))
        for _, row in signals.iterrows():
            sector = row.get("sector", "unknown")
            if sector_counts.get(sector, 0) >= max_sector_positions:
                continue
            selected.append(row)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if len(selected) >= self.config.max_positions:
                break
        return pd.DataFrame(selected)

    def write_orders(self, orders: pd.DataFrame, trade_date: str) -> Path:
        path = self.log_dir / f"orders_{trade_date}.csv"
        orders.to_csv(path, index=False)
        return path

    def record_outcomes(self, orders: pd.DataFrame, outcomes: pd.DataFrame, trade_date: str) -> Path:
        merged = orders.merge(outcomes, on=["symbol", "horizon"], how="left", suffixes=("", "_actual"))
        path = self.log_dir / f"outcomes_{trade_date}.csv"
        merged.to_csv(path, index=False)
        return path

