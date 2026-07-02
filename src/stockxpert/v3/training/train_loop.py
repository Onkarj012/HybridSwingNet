"""V3 trainer with AMP and stock_traits inputs."""

from __future__ import annotations

import logging
import copy
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from stockxpert.training.train_loop import Trainer as V1Trainer

logger = logging.getLogger("stockxpert.v3.training.train_loop")


class Trainer(V1Trainer):
    def __init__(self, *args, use_amp: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _stock_input(self, inputs: dict) -> torch.Tensor:
        return inputs["stock_traits"].to(self.device)

    def _forward_batch(self, inputs: dict, targets: dict):
        stock_traits = self._stock_input(inputs)
        X_short = inputs["X_short"].to(self.device)
        X_mid = inputs["X_mid"].to(self.device)
        X_long = inputs["X_long"].to(self.device)
        X_context = inputs["X_context"].to(self.device)
        X_sentiment = inputs["X_sentiment"].to(self.device)

        if isinstance(targets, dict):
            targets_device = {k: v.to(self.device) for k, v in targets.items()}
        else:
            targets_device = targets.to(self.device)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(
                X_short, X_mid, X_long, X_context, X_sentiment, stock_traits
            )
            if len(outputs) == 7:
                direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred, _ = outputs
            else:
                direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred = outputs
            loss, loss_dict = self.criterion(
                direction_logits,
                magnitude_pred,
                confidence,
                target_zone_logits,
                aux_trend_logit,
                level_pred,
                targets_device,
            )
        return loss, loss_dict

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        epoch_losses = {
            "total": [],
            "direction": [],
            "magnitude": [],
            "regime": [],
            "confidence": [],
            "aux_trend": [],
            "consistency": [],
        }

        for inputs, targets in train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = self._forward_batch(inputs, targets)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key].append(loss_dict[key])

        return {k: sum(v) / max(len(v), 1) for k, v in epoch_losses.items()}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        epoch_losses = {
            "total": [],
            "direction": [],
            "magnitude": [],
            "regime": [],
            "confidence": [],
            "aux_trend": [],
            "consistency": [],
        }

        with torch.no_grad():
            for inputs, targets in val_loader:
                _, loss_dict = self._forward_batch(inputs, targets)
                for key in epoch_losses:
                    if key in loss_dict:
                        epoch_losses[key].append(loss_dict[key])

        return {k: sum(v) / max(len(v), 1) for k, v in epoch_losses.items()}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> pd.DataFrame:
        """V3 training loop.

        The V1 parent performs an SWA batch-norm pass with ``stock_idx`` after
        training. V3 has no stock IDs, so keep the loop local and stock-traits
        aware from start to finish.
        """
        logger.info("Starting V3 training for %d epochs", epochs)
        logger.info("Train batches per epoch: %d | Val batches: %d", len(train_loader), len(val_loader))
        best_state_dict = copy.deepcopy(self.model.state_dict())
        for epoch in range(1, epochs + 1):
            logger.info("Epoch %d/%d started", epoch, epochs)
            train_losses = self.train_epoch(train_loader)
            val_losses = self.validate(val_loader)

            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.scheduler.step(val_losses["total"])

            self.history.append(
                {
                    "epoch": epoch,
                    "train_total": train_losses["total"],
                    "train_direction": train_losses["direction"],
                    "train_magnitude": train_losses["magnitude"],
                    "val_total": val_losses["total"],
                    "val_direction": val_losses["direction"],
                    "val_magnitude": val_losses["magnitude"],
                    "train_confidence": train_losses["confidence"],
                    "val_confidence": val_losses["confidence"],
                    "train_aux_trend": train_losses["aux_trend"],
                    "val_aux_trend": val_losses["aux_trend"],
                    "train_consistency": train_losses["consistency"],
                    "val_consistency": val_losses["consistency"],
                }
            )

            logger.info(
                "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | "
                "Train Dir: %.4f | Val Dir: %.4f",
                epoch,
                epochs,
                train_losses["total"],
                val_losses["total"],
                train_losses["direction"],
                val_losses["direction"],
            )

            if val_losses["total"] < self.best_val_loss:
                self.best_val_loss = val_losses["total"]
                best_state_dict = copy.deepcopy(self.model.state_dict())
                self.epochs_without_improvement = 0
                logger.info("New best V3 val loss: %.4f", self.best_val_loss)
            else:
                self.epochs_without_improvement += 1
                logger.info(
                    "No V3 val improvement for %d epochs (best %.4f)",
                    self.epochs_without_improvement,
                    self.best_val_loss,
                )
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info("V3 early stopping at epoch %d", epoch)
                    break

        self.model.load_state_dict(best_state_dict)
        logger.info("Restored best V3 validation checkpoint before final evaluation")
        return pd.DataFrame(self.history)
