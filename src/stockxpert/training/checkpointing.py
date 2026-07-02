"""
Checkpoint management for saving/loading models.
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging

logger = logging.getLogger("stockxpert.training.checkpointing")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Path
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        path: Path to save checkpoint
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'loss': loss
    }
    
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    path: Path,
    device: torch.device
) -> int:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        path: Path to checkpoint
        device: Device to load to
    
    Returns:
        Epoch number from checkpoint
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    logger.info(f"Checkpoint loaded from {path} (epoch {epoch}, loss {loss:.4f})")
    
    return epoch
