"""
PyTorch DataLoaders for StockXpert samples.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from .make_samples import Sample


class StockXpertDataset(Dataset):
    """
    PyTorch Dataset for StockXpert samples.
    """
    
    def __init__(self, samples: List[Sample]):
        """
        Args:
            samples: List of Sample objects
        """
        self.samples = samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[dict, torch.Tensor]:
        """
        Returns:
            (inputs_dict, targets)
            
            inputs_dict:
                'X_short': (win_short, n_short_feat)
                'X_mid': (win_mid, n_mid_feat)
                'X_long': (win_long, n_long_feat)
                'X_context': (n_context_feat,)
                'stock_idx': int
            targets: (n_horizons,)
        """
        sample = self.samples[idx]
        
        inputs = {
            'X_short': torch.from_numpy(sample.X_short),
            'X_mid': torch.from_numpy(sample.X_mid),
            'X_long': torch.from_numpy(sample.X_long),
            'X_context': torch.from_numpy(sample.X_context),
            'X_sentiment': torch.from_numpy(sample.X_sentiment),
            'stock_idx': torch.tensor(sample.stock_idx, dtype=torch.long)
        }
        
        targets = {
            'magnitude': torch.from_numpy(sample.y),
            'zone': torch.from_numpy(sample.y_zone),
            'trend': torch.tensor([sample.y_trend], dtype=torch.float32),
            'levels': torch.from_numpy(sample.y_levels)
        }
        
        return inputs, targets


import numpy as np
from torch.utils.data import WeightedRandomSampler


def compute_sample_weights(samples: List[Sample], alpha: float = 2.0, min_w: float = 1.0, max_w: float = 20.0) -> np.ndarray:
    """
    Compute weights for tail oversampling based on target magnitude.
    Assuming target index 0 (1-day horizon) or 2 (5-day) is representative.
    Using index 0 (h=1) as default.
    """
    # Use first horizon as proxy for magnitude
    mags = np.array([abs(s.y[0]) for s in samples])
    med = np.median(mags) + 1e-8
    scaled = mags / med
    
    weights = min_w + (scaled ** alpha)
    weights = np.clip(weights, min_w, max_w)
    
    return weights


def create_dataloaders(
    train_samples: List[Sample],
    val_samples: List[Sample],
    test_samples: List[Sample],
    batch_size: int = 128,
    num_workers: int = 0,
    use_oversampling: bool = False,
    oversampling_cfg: dict = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders.
    """
    train_dataset = StockXpertDataset(train_samples)
    val_dataset = StockXpertDataset(val_samples)
    test_dataset = StockXpertDataset(test_samples)
    
    if use_oversampling:
        if oversampling_cfg is None:
            oversampling_cfg = {'alpha': 2.0, 'min_weight': 1.0, 'max_weight': 20.0}
            
        weights = compute_sample_weights(
            train_samples,
            alpha=oversampling_cfg.get('alpha', 2.0),
            min_w=oversampling_cfg.get('min_weight', 1.0),
            max_w=oversampling_cfg.get('max_weight', 20.0)
        )
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_samples),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
