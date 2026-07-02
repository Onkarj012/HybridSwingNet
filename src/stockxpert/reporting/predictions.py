"""
Prediction generation and export.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger("stockxpert.reporting.predictions")


def generate_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    horizons: List[int],
    output_path: Path
) -> pd.DataFrame:
    """
    Generate predictions and export to CSV.
    
    Args:
        model: Trained model
        dataloader: DataLoader (typically test set)
        device: Device
        horizons: List of prediction horizons
        output_path: Path to save predictions CSV
    
    Returns:
        DataFrame with predictions
    """
    model.eval()
    
    # Access dataset directly to get metadata which isn't in DataLoader tensors
    dataset = dataloader.dataset
    samples = dataset.samples
    
    # Ensure dataloader is not shuffled and matches sample order
    if isinstance(dataloader.sampler, torch.utils.data.RandomSampler):
        logger.warning("DataLoader is shuffled! Predictions will be misaligned with metadata.")
    
    records = []
    sample_idx = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            batch_size = inputs['X_short'].size(0)
            
            # Move to device
            # Move to device
            X_short = inputs['X_short'].to(device)
            X_mid = inputs['X_mid'].to(device)
            X_long = inputs['X_long'].to(device)
            X_context = inputs['X_context'].to(device)
            X_sentiment = inputs['X_sentiment'].to(device)
            stock_idx = inputs['stock_idx'].to(device)
            
            # Forward
            outputs = model(
                X_short, X_mid, X_long, X_context, X_sentiment, stock_idx
            )
            # Handle both 6 and 7 return values (with or without attention_dict)
            if len(outputs) == 7:
                direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred, _ = outputs
            else:
                direction_logits, magnitude_pred, confidence, target_zone_logits, aux_trend_logit, level_pred = outputs
            
            # Convert to numpy
            direction_logits_np = direction_logits.cpu().numpy()
            magnitude_pred_np = magnitude_pred.cpu().numpy()
            confidence_np = confidence.cpu().numpy()
            target_zone_logits_np = target_zone_logits.cpu().numpy() # (B, H, 7)
            level_pred_np = level_pred.cpu().numpy() # (B, H, 3) where [res, sup, tgt]
            
            if isinstance(targets, dict):
                 targets_np = targets['magnitude'].cpu().numpy()
            else:
                 targets_np = targets.cpu().numpy()
            
            # Create records for this batch
            for i in range(batch_size):
                if sample_idx >= len(samples):
                    break
                    
                sample = samples[sample_idx]
                sample_idx += 1
                
                record = {
                    'date': sample.date,
                    'symbol': sample.symbol,
                    'close_t': sample.close_t,
                    'vol_ref': sample.vol_ref
                }
                
                # Predictions per horizon
                for h_idx, h in enumerate(horizons):
                    # Direction
                    logit_h = direction_logits_np[i, h_idx]
                    p_up = 1.0 / (1.0 + np.exp(-logit_h))
                    
                    # Magnitude (Z-score predicted)
                    z_pred = magnitude_pred_np[i, h_idx]
                    z_actual = targets_np[i, h_idx]
                    
                    # Inverse transform: dlog = z * vol_ref
                    # Note: We use the sample's vol_ref for both pred and actual to compare apples-to-apples,
                    # though actual target was created with this vol_ref anyway.
                    dlog_pred = z_pred * sample.vol_ref
                    dlog_actual = z_actual * sample.vol_ref
                    
                    # Prices
                    pred_close = sample.close_t * np.exp(dlog_pred)
                    actual_close_proxy = sample.close_t * np.exp(dlog_actual) # Approximate if dlog is smoothed
                    
                    # Return %
                    pred_ret_pct = (np.exp(dlog_pred) - 1.0) * 100.0
                    actual_ret_pct = (np.exp(dlog_actual) - 1.0) * 100.0
                    
                    record[f'p_up_h{h}'] = p_up
                    record[f'pred_z_h{h}'] = z_pred
                    record[f'actual_z_h{h}'] = z_actual
                    record[f'actual_close_h{h}'] = actual_close_proxy
                    record[f'pred_dlog_h{h}'] = dlog_pred
                    record[f'pred_close_h{h}'] = pred_close
                    record[f'pred_curr_return_pct_h{h}'] = pred_ret_pct
                    
                    # New outputs
                    record[f'confidence_h{h}'] = confidence_np[i, h_idx]
                    
                    # Target Zone (argmax)
                    zone_logits = target_zone_logits_np[i, h_idx, :] # (7,)
                    zone_idx = np.argmax(zone_logits)
                    record[f'pred_zone_h{h}'] = zone_idx
                    # Maybe store probability of predicted zone?
                    zone_probs = np.exp(zone_logits) / np.sum(np.exp(zone_logits))
                    record[f'pred_zone_prob_h{h}'] = zone_probs[zone_idx]

                    # Price Levels [Support, Resistance, Target]
                    # z_lvl values for [res, sup, tgt]
                    z_res, z_sup, z_tgt = level_pred_np[i, h_idx]
                    
                    # Convert Z-scores back to nominal prices
                    # Price = Close_t * exp(z * vol_ref)
                    record[f'pred_res_h{h}'] = sample.close_t * np.exp(z_res * sample.vol_ref)
                    record[f'pred_sup_h{h}'] = sample.close_t * np.exp(z_sup * sample.vol_ref)
                    record[f'pred_tgt_h{h}'] = sample.close_t * np.exp(z_tgt * sample.vol_ref)
                
                records.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Save if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path} ({len(df)} rows)")
    
    return df
