"""V3 sample helpers for attaching static stock traits."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from stockxpert.dataset.make_samples import Sample, SampleBuilder as V1SampleBuilder


STOCK_TRAIT_COLUMNS = [
    "sector_enc_0",
    "sector_enc_1",
    "sector_enc_2",
    "mcap_rank",
    "beta",
    "vol_pct",
    "liquidity_tier",
]


def compute_stock_traits(
    df: pd.DataFrame,
    trait_columns: List[str] | None = None,
    fallback_dim: int = 7,
) -> np.ndarray:
    """Compute one static trait vector for a stock dataframe."""
    columns = trait_columns or STOCK_TRAIT_COLUMNS
    if all(col in df.columns for col in columns):
        values = df[columns].replace([np.inf, -np.inf], np.nan).median(numeric_only=True)
        return values.fillna(0.0).to_numpy(dtype=np.float32)
    return np.zeros(fallback_dim, dtype=np.float32)


class SampleBuilder(V1SampleBuilder):
    """V1 sample builder plus non-invasive ``stock_traits`` attributes."""

    def __init__(self, *args, stock_trait_columns: List[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stock_trait_columns = stock_trait_columns or STOCK_TRAIT_COLUMNS

    def build_samples(self, feature_data: Dict[str, pd.DataFrame], symbol_map: Dict[str, int]) -> List[Sample]:
        samples = super().build_samples(feature_data, symbol_map)
        traits_by_symbol = {
            symbol: compute_stock_traits(df, self.stock_trait_columns, len(self.stock_trait_columns))
            for symbol, df in feature_data.items()
        }
        for sample in samples:
            setattr(sample, "stock_traits", traits_by_symbol.get(sample.symbol, np.zeros(len(self.stock_trait_columns), dtype=np.float32)))
        return samples

