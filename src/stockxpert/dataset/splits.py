"""
Time-based data splitting.
"""

import pandas as pd
from typing import List, Tuple
from .make_samples import Sample
import logging

logger = logging.getLogger("stockxpert.dataset.splits")


def time_split(
    samples: List[Sample],
    train_end: str,
    val_end: str,
    test_end: str
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    """
    Split samples into train/val/test based on dates.
    
    Args:
        samples: List of Sample objects
        train_end: Train end date (YYYY-MM-DD), inclusive
        val_end: Validation end date (YYYY-MM-DD), inclusive
        test_end: Test end date (YYYY-MM-DD), inclusive
    
    Returns:
        (train_samples, val_samples, test_samples)
    """
    train_end_dt = pd.to_datetime(train_end).tz_localize(None)
    val_end_dt = pd.to_datetime(val_end).tz_localize(None)
    test_end_dt = pd.to_datetime(test_end).tz_localize(None)
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    for sample in samples:
        sample_date = pd.to_datetime(sample.date, utc=True).tz_localize(None)
        
        if sample_date <= train_end_dt:
            train_samples.append(sample)
        elif sample_date <= val_end_dt:
            val_samples.append(sample)
        elif sample_date <= test_end_dt:
            test_samples.append(sample)
        # Samples beyond test_end are dropped
    
    logger.info(f"Split summary:")
    logger.info(f"  Train: {len(train_samples)} samples (up to {train_end})")
    logger.info(f"  Val:   {len(val_samples)} samples ({train_end} to {val_end})")
    logger.info(f"  Test:  {len(test_samples)} samples ({val_end} to {test_end})")
    
    return train_samples, val_samples, test_samples
