"""V3 split utilities with embargo and cross-sectional holdouts."""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from typing import Dict, List, Tuple

import pandas as pd

from stockxpert.dataset.make_samples import Sample

logger = logging.getLogger("stockxpert.v3.dataset.splits")


def _sample_date(sample: Sample) -> pd.Timestamp:
    return pd.to_datetime(sample.date, utc=True).tz_localize(None)


def time_split(
    samples: List[Sample],
    train_end: str,
    val_end: str,
    test_end: str,
    embargo_days: int = 10,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """Temporal split with purge gaps after train and validation periods."""
    train_end_dt = pd.to_datetime(train_end).tz_localize(None)
    val_end_dt = pd.to_datetime(val_end).tz_localize(None)
    test_end_dt = pd.to_datetime(test_end).tz_localize(None)
    val_start_dt = train_end_dt + pd.Timedelta(days=embargo_days)
    test_start_dt = val_end_dt + pd.Timedelta(days=embargo_days)

    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    test_samples: list[Sample] = []

    for sample in samples:
        sample_date = _sample_date(sample)
        if sample_date <= train_end_dt:
            train_samples.append(sample)
        elif val_start_dt <= sample_date <= val_end_dt:
            val_samples.append(sample)
        elif test_start_dt <= sample_date <= test_end_dt:
            test_samples.append(sample)

    logger.info(
        "V3 temporal split: train=%d val=%d test=%d embargo_days=%d",
        len(train_samples),
        len(val_samples),
        len(test_samples),
        embargo_days,
    )
    return train_samples, val_samples, test_samples


def cross_sectional_split(
    samples: List[Sample],
    all_symbols: Sequence[str],
    train_end: str,
    val_end: str,
    test_end: str,
    val_stocks: int = 50,
    test_stocks: int = 50,
    blind_holdout_stocks: int = 50,
    embargo_days: int = 10,
    seed: int = 42,
) -> Dict[str, List[Sample]]:
    """Split by disjoint stock sets and temporal windows.

    Held-out validation/test/blind stocks never appear in the training set.
    Blind holdout uses the full date range through ``test_end`` after the
    initial train embargo to measure unseen-stock generalization.
    """
    symbols = list(dict.fromkeys(all_symbols))
    rng = random.Random(seed)
    rng.shuffle(symbols)

    blind_symbols = set(symbols[:blind_holdout_stocks])
    val_symbols = set(symbols[blind_holdout_stocks : blind_holdout_stocks + val_stocks])
    test_start = blind_holdout_stocks + val_stocks
    test_symbols = set(symbols[test_start : test_start + test_stocks])
    train_symbols = set(symbols[test_start + test_stocks :])

    by_symbol = {
        "train": [s for s in samples if s.symbol in train_symbols],
        "val": [s for s in samples if s.symbol in val_symbols],
        "test": [s for s in samples if s.symbol in test_symbols],
        "blind_holdout": [s for s in samples if s.symbol in blind_symbols],
    }

    train, _, _ = time_split(by_symbol["train"], train_end, val_end, test_end, embargo_days)
    _, val, _ = time_split(by_symbol["val"], train_end, val_end, test_end, embargo_days)
    _, _, test = time_split(by_symbol["test"], train_end, val_end, test_end, embargo_days)

    train_end_dt = pd.to_datetime(train_end).tz_localize(None)
    test_end_dt = pd.to_datetime(test_end).tz_localize(None)
    blind_start_dt = train_end_dt + pd.Timedelta(days=embargo_days)
    blind = [
        s
        for s in by_symbol["blind_holdout"]
        if blind_start_dt <= _sample_date(s) <= test_end_dt
    ]

    assert_no_leakage(train, val, test, blind)
    return {
        "train": train,
        "val": val,
        "test": test,
        "blind_holdout": blind,
    }


def assert_no_leakage(
    train: List[Sample],
    val: List[Sample],
    test: List[Sample],
    blind: List[Sample] | None = None,
) -> None:
    """Assert that stock universes are disjoint across split partitions."""
    train_syms = {s.symbol for s in train}
    val_syms = {s.symbol for s in val}
    test_syms = {s.symbol for s in test}

    assert train_syms.isdisjoint(val_syms), "LEAKAGE: stocks in both train and val"
    assert train_syms.isdisjoint(test_syms), "LEAKAGE: stocks in both train and test"
    assert val_syms.isdisjoint(test_syms), "LEAKAGE: stocks in both val and test"

    if blind is not None:
        blind_syms = {s.symbol for s in blind}
        assert train_syms.isdisjoint(blind_syms), "LEAKAGE: stocks in both train and blind"
        assert val_syms.isdisjoint(blind_syms), "LEAKAGE: stocks in both val and blind"
        assert test_syms.isdisjoint(blind_syms), "LEAKAGE: stocks in both test and blind"

