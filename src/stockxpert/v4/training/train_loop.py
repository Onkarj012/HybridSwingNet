"""V4 stock-traits-aware training loop."""

from __future__ import annotations

import copy
import logging

import pandas as pd
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger("stockxpert.v4.training.train_loop")


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        device,
        grad_clip: float = 1.0,
        early_stopping_patience: int = 5,
        use_amp: bool = True,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        self.early_stopping_patience = early_stopping_patience
        self.use_amp = bool(use_amp and device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.history: list[dict] = []

    def _forward_batch(self, inputs: dict, targets: dict):
        inputs_device = {key: value.to(self.device) for key, value in inputs.items()}
        targets_device = {key: value.to(self.device) for key, value in targets.items()}
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(
                inputs_device["X_short"],
                inputs_device["X_mid"],
                inputs_device["X_long"],
                inputs_device["X_context"],
                inputs_device["X_sentiment"],
                inputs_device["stock_traits"],
            )
            loss, metrics = self.criterion(outputs, targets_device)
        return loss, metrics

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        self.model.train()
        losses: dict[str, list[float]] = {}
        for inputs, targets in train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss, metrics = self._forward_batch(inputs, targets)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for key, value in metrics.items():
                losses.setdefault(key, []).append(value)
        return {key: sum(values) / max(len(values), 1) for key, values in losses.items()}

    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        losses: dict[str, list[float]] = {}
        with torch.no_grad():
            for inputs, targets in val_loader:
                _, metrics = self._forward_batch(inputs, targets)
                for key, value in metrics.items():
                    losses.setdefault(key, []).append(value)
        return {key: sum(values) / max(len(values), 1) for key, values in losses.items()}

    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> pd.DataFrame:
        best_state = copy.deepcopy(self.model.state_dict())
        for epoch in range(1, epochs + 1):
            logger.info("V4 epoch %d/%d", epoch, epochs)
            train_losses = self.train_epoch(train_loader)
            val_losses = self.validate(val_loader)
            row = {
                "epoch": epoch,
                **{f"train_{key}": value for key, value in train_losses.items()},
                **{f"val_{key}": value for key, value in val_losses.items()},
            }
            self.history.append(row)
            val_total = val_losses.get("total", float("inf"))
            logger.info("V4 epoch %d train_total=%.4f val_total=%.4f", epoch, train_losses.get("total", 0.0), val_total)
            if val_total < self.best_val_loss:
                self.best_val_loss = val_total
                best_state = copy.deepcopy(self.model.state_dict())
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info("V4 early stopping at epoch %d", epoch)
                    break
        self.model.load_state_dict(best_state)
        return pd.DataFrame(self.history)
