
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger("stockxpert.training.eval_loop")

def evaluate(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    device: torch.device,
    horizons: List[int] = [1, 3, 5, 7, 10]
) -> Dict[str, Any]:
    """
    Evaluate model on test set.
    
    Args:
        model: StockXpertModel
        dataloader: Test DataLoader
        device: Device
        horizons: List of horizons
        
    Returns:
        Dict containing metrics
    """
    model.eval()
    
    all_dir_preds = []
    all_dir_targets = []
    all_mag_preds = []
    all_mag_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move inputs to device
            X_short = inputs['X_short'].to(device)
            X_mid = inputs['X_mid'].to(device)
            X_long = inputs['X_long'].to(device)
            X_context = inputs['X_context'].to(device)
            X_sentiment = inputs['X_sentiment'].to(device)
            stock_idx = inputs['stock_idx'].to(device)
            
            # Move targets
            if isinstance(targets, dict):
                mag_targets = targets['magnitude'].to(device)
            else:
                mag_targets = targets.to(device)
            
            # Forward
            outputs = model(
                X_short, X_mid, X_long, X_context, X_sentiment, stock_idx
            )
            # Handle both 6 and 7 return values (with or without attention_dict)
            if len(outputs) == 7:
                direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred, _ = outputs
            else:
                direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred = outputs
            
            # Direction predictions
            dir_probs = torch.sigmoid(direction_logits)
            dir_preds = (dir_probs > 0.5).int()
            
            # Direction targets
            dir_targets = (mag_targets > 0).int()
            
            all_dir_preds.append(dir_preds.cpu().numpy())
            all_dir_targets.append(dir_targets.cpu().numpy())
            all_mag_preds.append(magnitude_pred.cpu().numpy())
            all_mag_targets.append(mag_targets.cpu().numpy())
            
    # Concatenate
    all_dir_preds = np.concatenate(all_dir_preds, axis=0) # (N, H)
    all_dir_targets = np.concatenate(all_dir_targets, axis=0)
    all_mag_preds = np.concatenate(all_mag_preds, axis=0)
    all_mag_targets = np.concatenate(all_mag_targets, axis=0)
    
    # Compute metrics per horizon
    metrics_per_horizon = []
    
    for i, h in enumerate(horizons):
        # Direction Accuracy
        correct = (all_dir_preds[:, i] == all_dir_targets[:, i])
        acc = np.mean(correct)
        
        # Phase 5: Check unbiased UP vs DOWN accuracy
        up_mask = (all_dir_targets[:, i] == 1)
        down_mask = (all_dir_targets[:, i] == 0)
        up_acc = np.mean(correct[up_mask]) if np.sum(up_mask) > 0 else 0.0
        down_acc = np.mean(correct[down_mask]) if np.sum(down_mask) > 0 else 0.0
        
        # Magnitude metrics
        mae = np.mean(np.abs(all_mag_preds[:, i] - all_mag_targets[:, i]))
        mse = np.mean((all_mag_preds[:, i] - all_mag_targets[:, i]) ** 2)
        rmse = np.sqrt(mse)
        
        # Price movement accuracy
        pm_acc = float(acc) * 100.0 # percentage?
        
        metrics_per_horizon.append({
            "horizon": h,
            "direction_accuracy": float(acc),
            "direction_up_accuracy": float(up_acc),
            "direction_down_accuracy": float(down_acc),
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "price_movement_accuracy": pm_acc
        })
        
        logger.info(f"Horizon {h}: Acc={acc:.4f} (Up: {up_acc:.4f}, Dn: {down_acc:.4f}), MAE={mae:.4f}")
        
    return {
        "metrics_per_horizon": metrics_per_horizon,
        "raw_predictions": {
            "direction": all_dir_preds,
            "magnitude": all_mag_preds
        },
        "raw_targets": {
            "direction": all_dir_targets,
            "magnitude": all_mag_targets
        }
    }
