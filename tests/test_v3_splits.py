import numpy as np
import pandas as pd
import pytest

from stockxpert.dataset.make_samples import Sample
from stockxpert.v3.dataset.splits import assert_no_leakage, cross_sectional_split, time_split


def _sample(symbol: str, date: str) -> Sample:
    return Sample(
        X_short=np.zeros((1, 1), dtype=np.float32),
        X_mid=np.zeros((1, 1), dtype=np.float32),
        X_long=np.zeros((1, 1), dtype=np.float32),
        X_context=np.zeros(1, dtype=np.float32),
        X_sentiment=np.zeros(9, dtype=np.float32),
        y=np.zeros(5, dtype=np.float32),
        y_zone=np.zeros(5, dtype=np.int64),
        y_levels=np.zeros((5, 3), dtype=np.float32),
        y_trend=0.0,
        date=pd.Timestamp(date),
        symbol=symbol,
        stock_idx=0,
        close_t=1.0,
        vol_ref=1.0,
    )


def test_time_split_enforces_embargo():
    samples = [_sample("A", date) for date in ["2023-01-01", "2023-01-05", "2023-01-10", "2023-01-20"]]
    train, val, test = time_split(samples, "2023-01-05", "2023-01-20", "2023-02-01", embargo_days=10)

    assert [s.date.day for s in train] == [1, 5]
    assert [s.date.day for s in val] == [20]
    assert test == []


def test_cross_sectional_split_has_disjoint_symbols():
    symbols = [f"S{i}" for i in range(8)]
    samples = [
        _sample(symbol, date)
        for symbol in symbols
        for date in ["2023-01-01", "2023-01-20", "2023-02-10", "2023-03-10"]
    ]

    split = cross_sectional_split(
        samples,
        symbols,
        "2023-01-05",
        "2023-02-15",
        "2023-03-31",
        val_stocks=2,
        test_stocks=2,
        blind_holdout_stocks=2,
        embargo_days=10,
        seed=1,
    )

    assert_no_leakage(split["train"], split["val"], split["test"], split["blind_holdout"])
    assert all(split.values())


def test_no_leakage_assertion_fails_for_overlap():
    train = [_sample("A", "2023-01-01")]
    val = [_sample("A", "2023-02-01")]
    with pytest.raises(AssertionError):
        assert_no_leakage(train, val, [])

