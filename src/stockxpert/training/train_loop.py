"""
Training loop with early stopping.

Includes:
- Standard loss-based early stopping
- Financial early stopping (based on direction accuracy, simulated returns)
- Comprehensive validation metrics (IC, hit rate, calibration)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field

logger = logging.getLogger("stockxpert.training.train_loop")


@dataclass
class FinancialMetrics:
    """Financial validation metrics for early stopping."""
    direction_accuracy: float = 0.5
    direction_up_accuracy: float = 0.5
    direction_down_accuracy: float = 0.5
    hit_rate: float = 0.5  # Same as direction accuracy but for positive returns
    information_coefficient: float = 0.0  # Rank correlation between pred and actual
    calibration_error: float = 1.0  # Lower is better
    simulated_return: float = 0.0  # Simulated strategy return
    sharpe_ratio: float = 0.0
    confidence_auc: float = 0.5  # How well confidence predicts accuracy


class FinancialEarlyStopping:
    """
    Early stopping based on financial metrics rather than just loss.
    
    Stops training when:
    - Direction accuracy fails to improve for N epochs
    - OR simulated returns degrade significantly
    - OR combined financial score doesn't improve
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max',  # 'max' for accuracy, 'min' for error
        metric: str = 'direction_accuracy',  # or 'combined_score'
        warmup_epochs: int = 5
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.metric = metric
        self.warmup_epochs = warmup_epochs
        
        self.best_score = -float('inf') if mode == 'max' else float('inf')
        self.epochs_without_improvement = 0
        self.epoch_count = 0
        
    def step(self, metrics: FinancialMetrics) -> bool:
        """
        Check if training should stop.
        
        Args:
            metrics: FinancialMetrics for current epoch
        
        Returns:
            True if training should stop
        """
        self.epoch_count += 1
        
        # Skip during warmup
        if self.epoch_count <= self.warmup_epochs:
            return False
        
        # Get score
        if self.metric == 'combined_score':
            score = self._compute_combined_score(metrics)
        else:
            score = getattr(metrics, self.metric, 0.5)
        
        # Check improvement
        improved = False
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            if self.mode == 'max':
                self.best_score = score
            else:
                self.best_score = score
            self.epochs_without_improvement = 0
            logger.info(f"  → Financial early stopping: New best {self.metric} = {score:.4f}")
        else:
            self.epochs_without_improvement += 1
            logger.info(
                f"  → Financial early stopping: No improvement for "
                f"{self.epochs_without_improvement} epochs (best: {self.best_score:.4f})"
            )
        
        return self.epochs_without_improvement >= self.patience
    
    def _compute_combined_score(self, metrics: FinancialMetrics) -> float:
        """Compute combined financial score."""
        # Weighted combination of metrics
        score = (
            0.4 * metrics.direction_accuracy +
            0.2 * max(0, metrics.information_coefficient) +
            0.2 * max(0, metrics.sharpe_ratio / 2) +  # Normalize sharpe
            0.1 * metrics.confidence_auc +
            0.1 * max(0, min(1, 0.5 + metrics.simulated_return))  # Normalize return
        )
        return score


def compute_comprehensive_metrics(
    direction_logits: torch.Tensor,
    magnitude_pred: torch.Tensor,
    confidence: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    horizons: List[int] = [1, 3, 5, 7, 10]
) -> FinancialMetrics:
    """
    Compute comprehensive validation metrics.
    
    Args:
        direction_logits: (B, H) direction predictions
        magnitude_pred: (B, H) magnitude predictions
        confidence: (B, H) confidence scores
        targets: Dict with 'magnitude' (B, H) targets
        horizons: List of prediction horizons
    
    Returns:
        FinancialMetrics with all computed metrics
    """
    metrics = FinancialMetrics()
    
    with torch.no_grad():
        # Get direction probabilities
        direction_probs = torch.sigmoid(direction_logits)
        direction_pred = (direction_probs > 0.5).float()
        
        # Get actual directions from magnitude targets
        mag_targets = targets['magnitude']
        actual_direction = (mag_targets > 0).float()
        
        # Direction accuracy (per horizon, then average)
        direction_correct = (direction_pred == actual_direction).float()
        metrics.direction_accuracy = direction_correct.mean().item()

        # Phase 5: Track separate UP and DOWN accuracy to verify unbiased predictions
        up_mask = (actual_direction == 1).float()
        down_mask = (actual_direction == 0).float()
        
        if up_mask.sum() > 0:
            metrics.direction_up_accuracy = (direction_correct * up_mask).sum().item() / up_mask.sum().item()
        if down_mask.sum() > 0:
            metrics.direction_down_accuracy = (direction_correct * down_mask).sum().item() / down_mask.sum().item()
        
        # Hit rate (accuracy for non-zero returns)
        significant_mask = torch.abs(mag_targets) > 0.001
        if significant_mask.sum() > 0:
            metrics.hit_rate = direction_correct[significant_mask].mean().item()
        
        # Information Coefficient (rank correlation)
        # Correlation between predicted magnitude and actual magnitude
        pred_flat = magnitude_pred.flatten().cpu().numpy()
        actual_flat = mag_targets.flatten().cpu().numpy()
        
        try:
            # Spearman rank correlation
            from scipy.stats import spearmanr
            ic, _ = spearmanr(pred_flat, actual_flat)
            metrics.information_coefficient = ic if not np.isnan(ic) else 0.0
        except:
            metrics.information_coefficient = 0.0
        
        # Calibration error (for confidence)
        # How well does confidence predict accuracy?
        conf_flat = confidence.flatten().cpu().numpy()
        correct_flat = direction_correct.flatten().cpu().numpy()
        
        try:
            # Bin by confidence and compute calibration
            bins = np.linspace(0, 1, 11)
            bin_indices = np.digitize(conf_flat, bins)
            calibration_errors = []
            for i in range(1, 11):
                mask = bin_indices == i
                if mask.sum() > 10:
                    avg_conf = conf_flat[mask].mean()
                    avg_acc = correct_flat[mask].mean()
                    calibration_errors.append(abs(avg_conf - avg_acc))
            metrics.calibration_error = np.mean(calibration_errors) if calibration_errors else 1.0
        except:
            metrics.calibration_error = 1.0
        
        # Simulated return (simple strategy: long if P(up) > 0.5, weighted by confidence)
        # Focus on H1 horizon
        h1_probs = direction_probs[:, 0].cpu().numpy()
        h1_conf = confidence[:, 0].cpu().numpy()
        h1_actual = mag_targets[:, 0].cpu().numpy()
        
        # Position: sign based on direction, size based on confidence
        positions = np.where(h1_probs > 0.5, 1, -1) * h1_conf
        returns = positions * h1_actual
        metrics.simulated_return = returns.mean()
        
        # Sharpe ratio of simulated returns
        if len(returns) > 1 and returns.std() > 0:
            metrics.sharpe_ratio = returns.mean() / returns.std()
        
        # Confidence AUC (how well confidence separates correct vs incorrect)
        try:
            from sklearn.metrics import roc_auc_score
            conf_for_auc = confidence.flatten().cpu().numpy()
            correct_for_auc = direction_correct.flatten().cpu().numpy()
            if len(np.unique(correct_for_auc)) > 1:
                metrics.confidence_auc = roc_auc_score(correct_for_auc, conf_for_auc)
        except:
            metrics.confidence_auc = 0.5
    
    return metrics


class Trainer:
    """
    Training loop manager with early stopping.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        grad_clip: float = 1.0,
        early_stopping_patience: int = 15
    ):
        """
        Args:
            model: StockXpertModel
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            grad_clip: Gradient clipping value
            early_stopping_patience: Epochs without val improvement before stopping
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip
        self.early_stopping_patience = early_stopping_patience
        
        self.history = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )
        
        # SWA (Stochastic Weight Averaging)
        from torch.optim.swa_utils import AveragedModel, SWALR
        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=0.0001)
        self.swa_start_epoch = 10
        
        # Curriculum Learning state
        self.curriculum_threshold = 0.0 # Default: No filtering
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch with optional Mixup augmentation.
        """
        self.model.train()
        
        epoch_losses = {'total': [], 'direction': [], 'magnitude': [], 'confidence': [], 'zone': [], 'aux_trend': [], 'level': []}
        
        # Mixup parameters
        mixup_alpha = 0.2
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move to device
            X_short = inputs['X_short'].to(self.device)
            X_mid = inputs['X_mid'].to(self.device)
            X_long = inputs['X_long'].to(self.device)
            X_context = inputs['X_context'].to(self.device)
            X_sentiment = inputs['X_sentiment'].to(self.device)
            stock_idx = inputs['stock_idx'].to(self.device)
            
            # Move targets to device
            if isinstance(targets, dict):
                mag_targets = targets['magnitude'].to(self.device)
                zone_targets = targets['zone'].to(self.device)
                trend_targets = targets['trend'].to(self.device)
            else:
                mag_targets = targets.to(self.device)
                zone_targets = None
                trend_targets = None
            
            # Mixup implementation
            if mixup_alpha > 0 and self.model.training:
                # ... mixup code ...
                pass 

            # Curriculum Learning: Filter easy/hard samples based on epoch
            # If curriculum_threshold is set, we only train on samples with |target_h1| > threshold
            if self.curriculum_threshold > 0.0:
                 # Phase 5: Check ALL horizons have clear signal, not just H1
                 # min(|target_h1|, |target_h3|, ...) > threshold
                 min_mag = torch.abs(mag_targets).min(dim=1).values  # (B,)
                 mask = min_mag > self.curriculum_threshold
                 
                 if mask.sum() == 0:
                     continue
                 
                 # Apply mask
                 X_short = X_short[mask]
                 X_mid = X_mid[mask]
                 X_long = X_long[mask]
                 X_context = X_context[mask]
                 X_sentiment = X_sentiment[mask]
                 stock_idx = stock_idx[mask]
                 mag_targets = mag_targets[mask]
                 if zone_targets is not None:
                     zone_targets = zone_targets[mask]
                 if trend_targets is not None:
                     trend_targets = trend_targets[mask]
            
            # Mixup logic (simplified re-insert)
            if mixup_alpha > 0 and self.model.training:
                lam = torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item()
                index = torch.randperm(X_short.size(0)).to(self.device)
                
                # Mix inputs
                X_short = lam * X_short + (1 - lam) * X_short[index]
                X_mid = lam * X_mid + (1 - lam) * X_mid[index]
                X_long = lam * X_long + (1 - lam) * X_long[index]
                X_context = lam * X_context + (1 - lam) * X_context[index]
                X_sentiment = lam * X_sentiment + (1 - lam) * X_sentiment[index]
                
                # Mix targets
                mag_targets = lam * mag_targets + (1 - lam) * mag_targets[index]
                if trend_targets is not None:
                    trend_targets = lam * trend_targets + (1 - lam) * trend_targets[index]
                # For classification (zone), we skip mixup or use one-hot. 
                # Keeping it simple: skip zone mixup (just uses primary)
            
            # Construct combined targets for criterion
            if isinstance(targets, dict):
                 targets_mixed = {'magnitude': mag_targets, 'zone': zone_targets, 'trend': trend_targets}
            else:
                 targets_mixed = mag_targets
            
            # Forward
            outputs = self.model(
                X_short, X_mid, X_long, X_context, X_sentiment, stock_idx
            )
            # Handle both 6 and 7 return values (with or without attention_dict)
            if len(outputs) == 7:
                direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred, attention_dict = outputs
            else:
                direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred = outputs
            
            # Loss
            loss, loss_dict = self.criterion(direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred, targets_mixed)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Record losses
            epoch_losses['total'].append(loss_dict['total'])
            epoch_losses['direction'].append(loss_dict['direction'])
            epoch_losses['magnitude'].append(loss_dict['magnitude'])
            if 'confidence' in loss_dict: epoch_losses['confidence'].append(loss_dict['confidence'])
            if 'zone' in loss_dict: epoch_losses['zone'].append(loss_dict['zone'])
            if 'aux_trend' in loss_dict: epoch_losses['aux_trend'].append(loss_dict['aux_trend'])
            if 'level' in loss_dict: epoch_losses['level'].append(loss_dict['level'])
        
        # Average losses
        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        
        return avg_losses
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate on validation set.
        
        Returns:
            Dictionary of average losses
        """
        self.model.eval()
        
        epoch_losses = {'total': [], 'direction': [], 'magnitude': [], 'confidence': [], 'zone': [], 'aux_trend': [], 'level': []}
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move to device
                X_short = inputs['X_short'].to(self.device)
                X_mid = inputs['X_mid'].to(self.device)
                X_long = inputs['X_long'].to(self.device)
                X_context = inputs['X_context'].to(self.device)
                X_sentiment = inputs['X_sentiment'].to(self.device)
                stock_idx = inputs['stock_idx'].to(self.device)
                
                # Move targets
                if isinstance(targets, dict):
                    targets = {k: v.to(self.device) for k, v in targets.items()}
                else:
                    targets = targets.to(self.device)
                
                # Forward
                outputs = self.model(
                    X_short, X_mid, X_long, X_context, X_sentiment, stock_idx
                )
                # Handle both 6 and 7 return values (with or without attention_dict)
                if len(outputs) == 7:
                    direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred, _ = outputs
                else:
                    direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred = outputs
                
                # Loss
                loss, loss_dict = self.criterion(direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred, targets)
                
                # Record losses
                epoch_losses['total'].append(loss_dict['total'])
                epoch_losses['direction'].append(loss_dict['direction'])
                epoch_losses['magnitude'].append(loss_dict['magnitude'])
                if 'confidence' in loss_dict: epoch_losses['confidence'].append(loss_dict['confidence'])
                if 'zone' in loss_dict: epoch_losses['zone'].append(loss_dict['zone'])
                if 'aux_trend' in loss_dict: epoch_losses['aux_trend'].append(loss_dict['aux_trend'])
                if 'level' in loss_dict: epoch_losses['level'].append(loss_dict['level'])
        
        # Average losses
        avg_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items()}
        
        
        return avg_losses
        
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int
    ) -> pd.DataFrame:
        """
        Full training loop with early stopping.
        
        Returns:
            DataFrame with training history
        """
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Early stopping patience: {self.early_stopping_patience}")
        
        for epoch in range(1, epochs + 1):
            # Curriculum Schedule
            # Epochs 1-5: Train only on significant moves (|z| > 0.5) -> "Easy" signal
            # Epochs 6+: Train on everything
            import time
            if epoch <= 5:
                self.curriculum_threshold = 0.5
                logger.info(f"  → Curriculum: High-Signal Mode (Threshold > {self.curriculum_threshold})")
            else:
                self.curriculum_threshold = 0.0
                if epoch == 6:
                     logger.info("  → Curriculum: Full Dataset Mode Enabled")

            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_losses = self.validate(val_loader)
            
            # Step scheduler
            if epoch > self.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
                lr = self.swa_scheduler.get_last_lr()[0]
                logger.info(f"  → SWA Update (LR: {lr:.6f})")
            else:
                self.scheduler.step(val_losses['total'])
            
            # Log
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_losses['total']:.4f} | "
                f"Val Loss: {val_losses['total']:.4f}"
            )
            
            # Record history
            self.history.append({
                'epoch': epoch,
                'train_total': train_losses['total'],
                'train_direction': train_losses['direction'],
                'train_magnitude': train_losses['magnitude'],
                'val_total': val_losses['total'],
                'val_direction': val_losses['direction'],
                'val_magnitude': val_losses['magnitude']
            })
            
            # Early stopping check
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.epochs_without_improvement = 0
                logger.info(f"  → New best val loss: {self.best_val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1
                logger.info(
                    f"  → No improvement for {self.epochs_without_improvement} epochs "
                    f"(best: {self.best_val_loss:.4f})"
                )
                
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Update SWA BN
        if epochs >= self.swa_start_epoch:
             logger.info("Updating SWA Batch Norm statistics...")
             # Custom update_bn to handle dict input
             self.swa_model.train()
             with torch.no_grad():
                 for i, (inputs, _) in enumerate(train_loader):
                     if i > 500: break # Limit to subset for speed
                     
                     # Extract tensors
                     X_short = inputs['X_short'].to(self.device)
                     X_mid = inputs['X_mid'].to(self.device)
                     X_long = inputs['X_long'].to(self.device)
                     X_context = inputs['X_context'].to(self.device)
                     X_sentiment = inputs['X_sentiment'].to(self.device)
                     stock_idx = inputs['stock_idx'].to(self.device)
                     
                     # Forward pass updates BN stats
                     _ = self.swa_model(X_short, X_mid, X_long, X_context, X_sentiment, stock_idx)
                     
             # Replace main model with SWA model
             self.model.load_state_dict(self.swa_model.module.state_dict())
             logger.info("SWA model loaded for final evaluation.")
        
        # Convert history to DataFrame
        history_df = pd.DataFrame(self.history)
        
        return history_df
