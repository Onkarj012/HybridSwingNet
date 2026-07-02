"""V4 sample construction.

This module wraps the legacy tensor sample with trading-target metadata needed
by the V4 losses and policy replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from stockxpert.dataset.make_samples import Sample, SampleBuilder as BaseSampleBuilder
from stockxpert.v3.dataset.make_samples import STOCK_TRAIT_COLUMNS, compute_stock_traits
from stockxpert.v4.dataset.targets import TradeAction

logger = logging.getLogger("stockxpert.v4.dataset.make_samples")


@dataclass
class V4Sample:
    base: Sample
    raw_return: float
    market_excess_return: float
    net_return_long: float
    net_return_short: float
    trade_action: TradeAction
    mfe: float
    mae: float
    cost_bps: float
    sample_weight: float
    stock_traits: np.ndarray

    def __getattr__(self, name: str):
        return getattr(self.base, name)


class SampleBuilder(BaseSampleBuilder):
    """Build V4 samples from frames that already have V4 target columns."""

    required_target_columns = {
        "raw_return",
        "market_excess_return",
        "net_return_long",
        "net_return_short",
        "trade_action",
        "mfe",
        "mae",
        "cost_bps",
        "sample_weight",
    }

    def __init__(self, *args, stock_trait_columns: List[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stock_trait_columns = stock_trait_columns or STOCK_TRAIT_COLUMNS

    def build_samples(self, feature_data: Dict[str, pd.DataFrame], symbol_map: Dict[str, int]) -> List[V4Sample]:
        samples: list[V4Sample] = []
        total_symbols = len(feature_data)
        for idx, (symbol, df) in enumerate(feature_data.items(), start=1):
            logger.info("V4 samples for %s (%d/%d)", symbol, idx, total_symbols)
            missing = self.required_target_columns.difference(df.columns)
            if missing:
                raise ValueError(f"{symbol} is missing V4 target columns: {sorted(missing)}")
            traits = compute_stock_traits(df, self.stock_trait_columns, len(self.stock_trait_columns))
            before = len(samples)
            for base in super()._build_samples_for_symbol(df, symbol, symbol_map[symbol]):
                row = df.loc[base.date]
                if pd.isna(row["trade_action"]):
                    continue
                samples.append(
                    V4Sample(
                        base=base,
                        raw_return=float(row["raw_return"]),
                        market_excess_return=float(row["market_excess_return"]),
                        net_return_long=float(row["net_return_long"]),
                        net_return_short=float(row["net_return_short"]),
                        trade_action=TradeAction(int(row["trade_action"])),
                        mfe=float(row["mfe"]),
                        mae=float(row["mae"]),
                        cost_bps=float(row["cost_bps"]),
                        sample_weight=float(row["sample_weight"]),
                        stock_traits=traits,
                    )
                )
            logger.info("  %s: %d V4 samples", symbol, len(samples) - before)
        return samples
