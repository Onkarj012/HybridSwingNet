"""
Loss functions for StockXpert.

Includes:
- StockXpertLoss: Original combined loss
- DirectionAwareLoss: Asymmetric focal loss for direction prediction
- QuantileLoss: Pinball loss for quantile regression
- DirectionConsistencyLoss: Penalize inconsistent direction across horizons
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


class DirectionAwareLoss(nn.Module):
    """
    Asymmetric loss for direction prediction.
    
    Penalizes:
    - Confident wrong predictions more heavily
    - Missing downside moves more than upside (risk management)
    
    Args:
        gamma_pos: Focal parameter for positive class (up moves)
        gamma_neg: Focal parameter for negative class (down moves) - typically higher
        downside_weight: Extra weight for missing downside predictions
    """
    
    def __init__(
        self,
        gamma_pos: float = 1.5,  # Balanced default
        gamma_neg: float = 1.5,  # Balanced default
        downside_weight: float = 1.0  # Balanced default (no extra penalty)
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.downside_weight = downside_weight
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            logits: (B, H) raw logits
            targets: (B, H) binary targets (1 = up, 0 = down)
            weights: (Optional) (B, H) or (1, H) weights per element/horizon
        
        Returns:
            Scalar loss
        """
        probs = torch.sigmoid(logits)
        
        # Asymmetric focal weighting
        # When target=1 (up): weight = (1 - prob)^gamma_pos
        # When target=0 (down): weight = prob^gamma_neg
        focal_pos = (1 - probs) ** self.gamma_pos
        focal_neg = probs ** self.gamma_neg
        
        # Select weight based on target
        focal_weight = torch.where(targets == 1, focal_pos, focal_neg)
        
        # Downside penalty: when target=0 (down move) and we predicted up
        # Structural fix: Only apply if weight > 1.0 to ensure symmetry when weight=1.0
        # Original logic multiplied by 'probs' which inherently skewed the loss.
        # modified to be a scalar multiplier for the class if downside_weight > 1.0
        
        downside_penalty = torch.ones_like(probs)
        if self.downside_weight > 1.0:
            # Apply extra penalty for False Positives (predicting UP when wrong)
            # Weighted by confidence but kept consistent
            # Use raw scalar weight for simplicity and symmetry
             downside_penalty = torch.where(
                targets == 0,
                torch.tensor(self.downside_weight, device=probs.device), 
                torch.ones_like(probs)
            )
        
        # Combined weight
        combined_weight = focal_weight * downside_penalty
        
        # BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        loss = bce * combined_weight
        
        if weights is not None:
             loss = loss * weights
             
        return loss.mean()


class QuantileLoss(nn.Module):
    """
    Pinball loss for quantile regression.
    
    Predicts multiple quantiles (e.g., 10%, 50%, 90%) for uncertainty quantification.
    
    Args:
        quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
    """
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(
        self,
        predictions: List[torch.Tensor],
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: List of (B, H) tensors, one per quantile
            target: (B, H) target values
        
        Returns:
            Scalar loss
        """
        losses = []
        
        for q, pred in zip(self.quantiles, predictions):
            error = target - pred
            # Pinball loss: max(q * error, (q - 1) * error)
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.mean())
        
        return sum(losses) / len(losses)


class DirectionConsistencyLoss(nn.Module):
    """
    Penalizes inconsistent direction predictions across horizons.
    
    If H1 predicts UP strongly, H3 should not predict DOWN strongly.
    Enforces temporal consistency.
    
    Args:
        flip_threshold: Probability difference threshold to trigger penalty
        max_flip_penalty: Maximum penalty for direction flips
    """
    
    def __init__(self, flip_threshold: float = 0.3, max_flip_penalty: float = 1.0):
        super().__init__()
        self.flip_threshold = flip_threshold
        self.max_flip_penalty = max_flip_penalty
        
    def forward(self, direction_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            direction_logits: (B, num_horizons) raw logits
        
        Returns:
            Scalar consistency loss
        """
        probs = torch.sigmoid(direction_logits)  # (B, H)
        
        # Compare adjacent horizons
        # diffs[i] = probs[i+1] - probs[i]
        diffs = torch.diff(probs, dim=1)  # (B, H-1)
        
        # Penalize large flips in opposite direction
        # If diff > threshold (e.g., +0.3) when previous was down, that's OK
        # If diff < -threshold (e.g., -0.3) when previous was up, that's a flip
        
        # We want to penalize |diff| > threshold that represents direction change
        # Direction change: probs[i] > 0.5 and probs[i+1] < 0.5 (or vice versa)
        
        prev_probs = probs[:, :-1]  # (B, H-1)
        next_probs = probs[:, 1:]   # (B, H-1)
        
        # Detect direction flips
        prev_up = prev_probs > 0.5
        next_down = next_probs < 0.5
        flip_up_to_down = prev_up & next_down
        
        prev_down = prev_probs < 0.5
        next_up = next_probs > 0.5
        flip_down_to_up = prev_down & next_up
        
        any_flip = flip_up_to_down | flip_down_to_up  # (B, H-1)
        
        # Penalty proportional to flip magnitude
        flip_magnitude = torch.abs(diffs) - self.flip_threshold
        flip_magnitude = F.relu(flip_magnitude)  # Only penalize beyond threshold
        
        # Apply penalty only where flip occurred
        penalty = flip_magnitude * any_flip.float()
        
        return penalty.mean() * self.max_flip_penalty


class CumulativeReturnConsistencyLoss(nn.Module):
    """
    Enforces approximate consistency between short and long horizon returns.
    
    For a random walk: H3 ≈ sqrt(3) * H1
    With momentum: ratio can be higher
    With mean reversion: ratio can be lower
    
    Args:
        expected_ratios: Expected |H_i| / |H_1| ratios for each horizon
        tolerance: Tolerance for ratio deviation
    """
    
    def __init__(
        self,
        horizons: List[int] = [1, 3, 5, 7, 10],
        tolerance: float = 0.5
    ):
        super().__init__()
        # Expected ratios based on random walk: sqrt(horizon)
        self.expected_ratios = torch.tensor([1.0] + [h ** 0.5 for h in horizons[1:]])
        self.tolerance = tolerance
        
    def forward(self, magnitude_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            magnitude_pred: (B, H) predicted magnitudes (log returns)
        
        Returns:
            Scalar consistency loss
        """
        device = magnitude_pred.device
        expected_ratios = self.expected_ratios.to(device)
        
        h1 = magnitude_pred[:, 0:1]  # (B, 1)
        other_h = magnitude_pred[:, 1:]  # (B, H-1)
        
        # Compute actual ratios
        h1_abs = torch.abs(h1) + 1e-6
        actual_ratios = torch.abs(other_h) / h1_abs  # (B, H-1)
        
        # Deviation from expected
        expected_ratios = expected_ratios[1:].view(1, -1)  # (1, H-1)
        deviation = torch.abs(actual_ratios - expected_ratios)
        
        # Only penalize beyond tolerance
        loss = F.relu(deviation - self.tolerance)
        
        return loss.mean() * 0.1  # Small weight


class StockXpertLoss(nn.Module):
    """
    Combined loss: direction (BCE) + magnitude (L1) + variance penalty.
    
    Prevents model collapse by penalizing under-predicted variance.
    """
    
    def __init__(
        self,
        direction_weight: float = 0.5,
        magnitude_weight: float = 0.4,
        variance_weight: float = 0.1,
        zone_weight: float = 0.3,
        confidence_weight: float = 0.2,
        aux_trend_weight: float = 0.5,
        level_weight: float = 0.2,
        consistency_weight: float = 0.1,
        magnitude_symmetry_weight: float = 0.1,
        use_direction_aware: bool = True,
        use_consistency: bool = True,
        horizon_weights: Optional[List[float]] = None,
        # Direction Aware Params
        gamma_pos: float = 1.5,
        gamma_neg: float = 1.5,
        downside_weight: float = 1.0
    ):
        super().__init__()
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.variance_weight = variance_weight
        self.zone_weight = zone_weight
        self.confidence_weight = confidence_weight
        self.aux_trend_weight = aux_trend_weight
        self.level_weight = level_weight
        self.consistency_weight = consistency_weight
        self.magnitude_symmetry_weight = magnitude_symmetry_weight
        self.use_direction_aware = use_direction_aware
        self.use_consistency = use_consistency
        
        # Horizon weights (default to 1.0 if not provided)
        # Higher weight for shorter horizons usually
        self.horizon_weights = torch.tensor(horizon_weights) if horizon_weights else None
        
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.l1_loss = nn.L1Loss(reduction='none')
        self.loss_zone = nn.CrossEntropyLoss(reduction='none')
        
        # New loss components
        if use_direction_aware:
            self.direction_aware_loss = DirectionAwareLoss(
                gamma_pos=gamma_pos,
                gamma_neg=gamma_neg,
                downside_weight=downside_weight
            )
        
        if use_consistency:
            self.consistency_loss = DirectionConsistencyLoss(
                flip_threshold=0.3,
                max_flip_penalty=1.0
            )
            self.cumulative_loss = CumulativeReturnConsistencyLoss()
        
    def set_horizon_weights(self, weights: list):
        """Set weights for each horizon [H1, H3, H5, H7, H10]."""
        self.horizon_weights = torch.tensor(weights, dtype=torch.float32)

    def forward(
        self,
        direction_logits: torch.Tensor,
        magnitude_pred: torch.Tensor,
        confidence: torch.Tensor,
        target_zone_logits: torch.Tensor,
        aux_trend_logit: Optional[torch.Tensor],
        level_pred: Optional[torch.Tensor],
        targets: Dict
    ) -> tuple:
        """
        Compute combined loss.
        
        Args:
            direction_logits: (B, H) direction logits
            magnitude_pred: (B, H) magnitude predictions
            confidence: (B, H) confidence scores
            target_zone_logits: (B, H, 7) zone classification logits
            aux_trend_logit: (B, 1) auxiliary trend logit or None
            level_pred: (B, H, 3) level predictions or None
            targets: Dict with 'magnitude', 'zone', 'trend', 'levels'
        
        Returns:
            (total_loss, loss_dict)
        """
        device = direction_logits.device
        
        # Unpack targets
        if isinstance(targets, dict):
            mag_targets = targets['magnitude']  # (B, H)
            zone_targets = targets.get('zone', None)
            trend_targets = targets.get('trend', None)
            level_targets = targets.get('levels', None)
        else:
            mag_targets = targets
            zone_targets = None
            trend_targets = None
            level_targets = None
            
        # Ensure horizon weights are on correct device
        if self.horizon_weights is not None:
            if self.horizon_weights.device != device:
                self.horizon_weights = self.horizon_weights.to(device)
            w = self.horizon_weights.view(1, -1)
        else:
            w = 1.0
            
        # 1. Direction loss
        direction_targets = (mag_targets > 0).float()
        
        # Phase 5: Confidence-weight direction based on target magnitude
        # Near-zero moves have unreliable direction labels → down-weight them
        # Use horizon-adaptive thresholds: H1 has smaller magnitudes
        # Threshold: [0.5, 0.7, 1.0, 1.2, 1.5] for H1→H10
        H = mag_targets.shape[1]
        thresholds = torch.linspace(0.5, 1.5, H, device=device).unsqueeze(0)  # (1, H)
        dir_confidence = torch.clamp(torch.abs(mag_targets) / thresholds, 0.1, 1.0)
        
        # Combine with horizon weights
        combined_w = dir_confidence
        if isinstance(w, torch.Tensor):
            combined_w = combined_w * w
            
        # FIX: Dynamic class balancing to prevent DOWN-only bias
        # Calculate frequency of UP vs DOWN in this batch
        pos_count = (direction_targets == 1).sum() + 1.0  # +1 for smoothing
        neg_count = (direction_targets == 0).sum() + 1.0
        total_count = pos_count + neg_count
        
        # Inverse frequency weighting
        pos_weight = total_count / (2.0 * pos_count)
        neg_weight = total_count / (2.0 * neg_count)
        
        # Apply balancer to sample weights
        class_balancer = torch.where(direction_targets == 1, pos_weight, neg_weight)
        combined_w = combined_w * class_balancer
        
        if self.use_direction_aware:
            # Use asymmetric/balanced direction-aware loss with weights
            loss_direction = self.direction_aware_loss(direction_logits, direction_targets, weights=combined_w)
        else:
            # Original focal loss
            epsilon = 0.1
            direction_targets_smooth = direction_targets * (1 - epsilon) + 0.5 * epsilon
            bce_raw = self.bce_loss(direction_logits, direction_targets_smooth)
            
            with torch.no_grad():
                probs = torch.sigmoid(direction_logits)
                p_t = direction_targets * probs + (1 - direction_targets) * (1 - probs)
                focal_weight = (1 - p_t) ** 2.0
            
            loss_direction_raw = bce_raw * focal_weight
            loss_direction = (loss_direction_raw * combined_w).mean()
        
        # 2. Magnitude loss (L1 is more robust than MSE)
        loss_magnitude_raw = self.l1_loss(magnitude_pred, mag_targets)
        loss_magnitude = (loss_magnitude_raw * w).mean()
        
        # 3. Variance Penalty (Anti-Collapse)
        # Force model to predict variations, not just safe 0.0 values
        pred_std = magnitude_pred.std(dim=0).mean()
        target_std = mag_targets.std(dim=0).mean()
        # Scale up variance penalty severely if model collapses (std < 0.1)
        collapse_penalty = F.relu(0.1 - pred_std) * 10.0 
        loss_variance = F.relu(target_std - pred_std) + collapse_penalty
        
        # 4. Target Zone Loss (CrossEntropy)
        if zone_targets is not None:
            B, H, C = target_zone_logits.shape
            loss_zone_raw = self.loss_zone(target_zone_logits.view(-1, C), zone_targets.view(-1))
            
            if isinstance(w, torch.Tensor):
                w_flat = w.expand(B, -1).reshape(-1)
                loss_zone = (loss_zone_raw * w_flat).mean()
            else:
                loss_zone = loss_zone_raw.mean()
        else:
            loss_zone = torch.tensor(0.0, device=device)
            
        # 5. Confidence Loss (Entropy Regularized)
        with torch.no_grad():
            dir_probs = torch.sigmoid(direction_logits)
            # Soft correctness: measure agreement between prediction and target
            # If target=1 and prob=0.9 -> soft_correct=0.9
            # If target=1 and prob=0.1 -> soft_correct=0.1
            soft_correct = dir_probs * direction_targets + (1 - dir_probs) * (1 - direction_targets)
        
        loss_confidence_raw = self.bce_loss(confidence, soft_correct)
        loss_confidence_cal = (loss_confidence_raw * w).mean()
        
        # Entropy penalty: Penalize low standard deviation of confidence
        # We want diversity in confidence scores (some high, some low)
        conf_std = confidence.std(dim=0).mean()
        loss_confidence_entropy = F.relu(0.15 - conf_std) 
        
        loss_confidence = loss_confidence_cal + loss_confidence_entropy
        
        # 6. Auxiliary Trend Loss
        if aux_trend_logit is not None and trend_targets is not None:
            trend_loss_raw = self.bce_loss(aux_trend_logit, trend_targets)
            loss_aux = trend_loss_raw.mean()
        else:
            loss_aux = torch.tensor(0.0, device=device)

        # 7. Level Loss
        if level_targets is not None and level_pred is not None:
            loss_level_raw = self.l1_loss(level_pred, level_targets)
            
            if isinstance(w, torch.Tensor):
                w_level = w.view(1, -1, 1)
                loss_level = (loss_level_raw * w_level).mean()
            else:
                loss_level = loss_level_raw.mean()
        else:
            loss_level = torch.tensor(0.0, device=device)
        
        # 8. Consistency Loss (NEW)
        if self.use_consistency:
            loss_consistency = self.consistency_loss(direction_logits)
            loss_cumulative = self.cumulative_loss(magnitude_pred)
        else:
            loss_consistency = torch.tensor(0.0, device=device)
            loss_cumulative = torch.tensor(0.0, device=device)

        # 9. Magnitude Symmetry Penalty (NEW)
        # Penalize if mean magnitude prediction is far from 0 (batch-wise)
        # This forces the model to predict both positive and negative moves
        magnitude_mean_bias = magnitude_pred.mean().abs()
        loss_magnitude_symmetry = magnitude_mean_bias
            
        # Combined
        total_loss = (
            self.direction_weight * loss_direction +
            self.magnitude_weight * loss_magnitude +
            self.variance_weight * loss_variance +
            self.zone_weight * loss_zone +
            self.confidence_weight * loss_confidence +
            self.aux_trend_weight * loss_aux +
            self.level_weight * loss_level +
            self.consistency_weight * (loss_consistency + loss_cumulative) +
            self.magnitude_symmetry_weight * loss_magnitude_symmetry
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'direction': loss_direction.item(),
            'magnitude': loss_magnitude.item(),
            'variance': loss_variance.item(),
            'zone': loss_zone.item(),
            'confidence': loss_confidence.item(),
            'aux_trend': loss_aux.item(),
            'level': loss_level.item(),
            'consistency': loss_consistency.item() if self.use_consistency else 0.0
        }
        
        return total_loss, loss_dict


class ProbabilisticLoss(nn.Module):
    """
    Loss for probabilistic magnitude prediction (mean + variance).
    
    Uses negative log-likelihood of Gaussian distribution.
    """
    
    def __init__(self, min_variance: float = 1e-6):
        super().__init__()
        self.min_variance = min_variance
        
    def forward(
        self,
        mean_pred: torch.Tensor,
        std_pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            mean_pred: (B, H) predicted mean
            std_pred: (B, H) predicted std (must be positive)
            target: (B, H) target values
        
        Returns:
            Scalar NLL loss
        """
        # Ensure positive variance
        variance = F.softplus(std_pred) ** 2 + self.min_variance
        
        # Negative log-likelihood: 0.5 * log(var) + (x - mu)^2 / (2 * var)
        nll = 0.5 * torch.log(variance) + (target - mean_pred) ** 2 / (2 * variance)
        
        return nll.mean()
