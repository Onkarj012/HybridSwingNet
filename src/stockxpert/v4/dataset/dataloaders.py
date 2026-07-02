"""V4 torch datasets."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from stockxpert.v4.dataset.make_samples import V4Sample


class StockXpertV4Dataset(Dataset):
    def __init__(self, samples: List[V4Sample]):
        self.samples = samples

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
            "stock_traits": torch.from_numpy(sample.stock_traits),
        }
        targets = {
            "p_up": torch.tensor(float(sample.net_return_long > sample.net_return_short), dtype=torch.float32),
            "action": torch.tensor(int(sample.trade_action), dtype=torch.long),
            "net_return_long": torch.tensor(sample.net_return_long, dtype=torch.float32),
            "net_return_short": torch.tensor(sample.net_return_short, dtype=torch.float32),
            "downside": torch.tensor(abs(min(sample.mae, 0.0)), dtype=torch.float32),
            "confidence_correct": torch.tensor(float(sample.trade_action != 0), dtype=torch.float32),
            "trend": torch.tensor([sample.y_trend], dtype=torch.float32),
            "sample_weight": torch.tensor(sample.sample_weight, dtype=torch.float32),
        }
        return inputs, targets


def create_dataloaders(
    train_samples: List[V4Sample],
    val_samples: List[V4Sample],
    test_samples: List[V4Sample],
    batch_size: int = 256,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    return (
        DataLoader(StockXpertV4Dataset(train_samples), batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(StockXpertV4Dataset(val_samples), batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(StockXpertV4Dataset(test_samples), batch_size=batch_size, shuffle=False, num_workers=num_workers),
    )
