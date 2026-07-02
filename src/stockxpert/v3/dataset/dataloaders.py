"""V3 dataloaders that pass static stock traits instead of stock indices."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from stockxpert.dataset.dataloaders import compute_sample_weights
from stockxpert.dataset.make_samples import Sample


DEFAULT_STOCK_TRAITS_DIM = 7


def stock_traits_from_sample(sample: Sample, dim: int = DEFAULT_STOCK_TRAITS_DIM) -> np.ndarray:
    """Return sample stock traits, falling back to zeros for legacy samples."""
    traits = getattr(sample, "stock_traits", None)
    if traits is None:
        return np.zeros(dim, dtype=np.float32)
    arr = np.asarray(traits, dtype=np.float32)
    if arr.shape[0] != dim:
        raise ValueError(f"Expected stock_traits dim {dim}, got {arr.shape[0]} for {sample.symbol}")
    return arr


class StockXpertV3Dataset(Dataset):
    def __init__(self, samples: List[Sample], stock_traits_dim: int = DEFAULT_STOCK_TRAITS_DIM):
        self.samples = samples
        self.stock_traits_dim = stock_traits_dim

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[dict, dict]:
        sample = self.samples[idx]
        inputs = {
            "X_short": torch.from_numpy(sample.X_short),
            "X_mid": torch.from_numpy(sample.X_mid),
            "X_long": torch.from_numpy(sample.X_long),
            "X_context": torch.from_numpy(sample.X_context),
            "X_sentiment": torch.from_numpy(sample.X_sentiment),
            "stock_traits": torch.from_numpy(stock_traits_from_sample(sample, self.stock_traits_dim)),
        }
        targets = {
            "magnitude": torch.from_numpy(sample.y),
            "zone": torch.from_numpy(sample.y_zone),
            "trend": torch.tensor([sample.y_trend], dtype=torch.float32),
            "levels": torch.from_numpy(sample.y_levels),
        }
        return inputs, targets


def create_dataloaders(
    train_samples: List[Sample],
    val_samples: List[Sample],
    test_samples: List[Sample],
    batch_size: int = 128,
    num_workers: int = 0,
    use_oversampling: bool = False,
    oversampling_cfg: dict | None = None,
    stock_traits_dim: int = DEFAULT_STOCK_TRAITS_DIM,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = StockXpertV3Dataset(train_samples, stock_traits_dim)
    val_dataset = StockXpertV3Dataset(val_samples, stock_traits_dim)
    test_dataset = StockXpertV3Dataset(test_samples, stock_traits_dim)

    if use_oversampling:
        oversampling_cfg = oversampling_cfg or {"alpha": 2.0, "min_weight": 1.0, "max_weight": 20.0}
        weights = compute_sample_weights(
            train_samples,
            alpha=oversampling_cfg.get("alpha", 2.0),
            min_w=oversampling_cfg.get("min_weight", 1.0),
            max_w=oversampling_cfg.get("max_weight", 20.0),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(train_samples), replacement=True),
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


StockXpertDataset = StockXpertV3Dataset

