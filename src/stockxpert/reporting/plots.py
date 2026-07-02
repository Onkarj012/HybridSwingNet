"""
Visualization plots for training results and predictions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger("stockxpert.reporting.plots")

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


def plot_loss_curves(history_df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot training and validation loss curves.
    
    Args:
        history_df: DataFrame with columns [epoch, train_total, val_total, ...]
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total loss
    axes[0].plot(history_df['epoch'], history_df['train_total'], label='Train', marker='o')
    axes[0].plot(history_df['epoch'], history_df['val_total'], label='Val', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Direction loss
    axes[1].plot(history_df['epoch'], history_df['train_direction'], label='Train', marker='o')
    axes[1].plot(history_df['epoch'], history_df['val_direction'], label='Val', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Direction Loss (BCE)')
    axes[1].set_title('Direction Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Magnitude loss
    axes[2].plot(history_df['epoch'], history_df['train_magnitude'], label='Train', marker='o')
    axes[2].plot(history_df['epoch'], history_df['val_magnitude'], label='Val', marker='s')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Magnitude Loss (MSE)')
    axes[2].set_title('Magnitude Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Loss curves saved to {output_path}")


def plot_scatter_pred_vs_actual(
    magnitude_pred: np.ndarray,
    targets: np.ndarray,
    horizons: List[int],
    output_dir: Path
) -> None:
    """
    Plot scatter of predicted vs actual for each horizon.
    
    Args:
        magnitude_pred: (N, num_horizons)
        targets: (N, num_horizons)
        horizons: List of horizons
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for h_idx, h in enumerate(horizons):
        pred_h = magnitude_pred[:, h_idx]
        actual_h = targets[:, h_idx]
        
        fig, ax = plt.subplots(figsize=(7, 7))
        
        ax.scatter(actual_h, pred_h, alpha=0.3, s=10)
        
        # Add diagonal line
        lim_min = min(actual_h.min(), pred_h.min())
        lim_max = max(actual_h.max(), pred_h.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Δlog(Close)')
        ax.set_ylabel('Predicted Δlog(Close)')
        ax.set_title(f'Horizon {h}-day: Predicted vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = output_dir / f'scatter_pred_vs_actual_dlog_h{h}.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Scatter plot for h={h} saved to {output_path}")


def plot_residual_histograms(
    magnitude_pred: np.ndarray,
    targets: np.ndarray,
    horizons: List[int],
    output_dir: Path
) -> None:
    """
    Plot residual histograms for each horizon.
    
    Args:
        magnitude_pred: (N, num_horizons)
        targets: (N, num_horizons)
        horizons: List of horizons
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    
    for h_idx, h in enumerate(horizons):
        residuals = targets[:, h_idx] - magnitude_pred[:, h_idx]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', lw=2, label='Zero Residual')
        ax.set_xlabel('Residual (Actual - Predicted)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Horizon {h}-day: Residual Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        output_path = output_dir / f'residual_hist_h{h}.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Residual histogram for h={h} saved to {output_path}")


def plot_confusion_matrices(
    direction_logits: np.ndarray,
    targets: np.ndarray,
    horizons: List[int],
    output_dir: Path
) -> None:
    """
    Plot confusion matrices for direction prediction.
    
    Args:
        direction_logits: (N, num_horizons)
        targets: (N, num_horizons)
        horizons: List of horizons
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    
    for h_idx, h in enumerate(horizons):
        # Convert logits to predictions
        logits_h = direction_logits[:, h_idx]
        pred_prob_up = 1.0 / (1.0 + np.exp(-logits_h))
        pred_dir = (pred_prob_up > 0.5).astype(int)
        true_dir = (targets[:, h_idx] > 0).astype(int)
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(true_dir, pred_dir)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Down/Flat', 'Up'],
                    yticklabels=['Down/Flat', 'Up'])
        
        ax.set_xlabel('Predicted Direction')
        ax.set_ylabel('True Direction')
        ax.set_title(f'Horizon {h}-day: Confusion Matrix')
        
        output_path = output_dir / f'confusion_matrix_h{h}.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix for h={h} saved to {output_path}")


def plot_price_predictions(
    predictions_df: pd.DataFrame,
    horizons: List[int],
    output_dir: Path,
    num_symbols: int = 5
) -> None:
    """
    Plot actual vs predicted close prices for a subset of symbols.
    
    Args:
        predictions_df: DataFrame containing metadata and predictions
        horizons: List of horizons
        output_dir: Directory to save plots
        num_symbols: Number of symbols to plot
    """
    output_dir = Path(output_dir)
    
    # Check if we have price prediction columns
    if 'pred_close_h1' not in predictions_df.columns:
        logger.warning("Price predictions not found in DataFrame. Skipping price plots.")
        return
        
    # Select subset of symbols
    unique_symbols = predictions_df['symbol'].unique()
    selected_symbols = unique_symbols[:min(len(unique_symbols), num_symbols)]
    
    for sym in selected_symbols:
        df_sym = predictions_df[predictions_df['symbol'] == sym].copy()
        df_sym['date'] = pd.to_datetime(df_sym['date'])
        df_sym = df_sym.sort_values('date').set_index('date')
        
        n_plots = len(horizons)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3.5 * n_plots), sharex=True)
        if n_plots == 1:
            axes = [axes]
            
        palette = sns.color_palette("husl", n_plots)
        
        for i, h in enumerate(horizons):
            ax = axes[i]
            col_pred = f'pred_close_h{h}'
            col_actual = f'actual_close_h{h}'
            
            # Predict t+h at time t. 
            # To visualize alignment properly, we plot both the prediction made at t
            # and the actual realization at t+h, but indexed at time t. 
            # This shows: "At date t, what did we predict for the future vs what actually happened?"
            
            # Actual future price realized at t+h (stored in CSV at index t)
            if col_actual in df_sym.columns:
                ax.plot(df_sym.index, df_sym[col_actual], color='black', lw=1.5, label=f'Actual Close (at t+{h})', alpha=0.5)
            else:
                # Fallback to shifting if column doesn't exist yet
                actual_future = df_sym['close_t'].shift(-h)
                ax.plot(df_sym.index, actual_future, color='black', lw=1.5, label=f'Actual Close (at t+{h})', alpha=0.5)
            
            # Plot predicted close (no shift - compare directly to the outcome for the same target date)
            ax.plot(df_sym.index, df_sym[col_pred], color=palette[i], lw=1.5, 
                   label=f'Pred Close ({h}d ahead)', alpha=0.9)
            
            ax.set_title(f'{sym} - {h}-Day Horizon Forecast (Indexed at Decision Date)')
            ax.set_ylabel('Price')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        save_path = output_dir / f'price_pred_{sym}.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Price plot for {sym} saved to {save_path}")


def create_all_plots(
    history_df: Optional[pd.DataFrame],
    eval_results: Dict,
    horizons: List[int],
    output_dir: Path,
    predictions_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Create all required plots.
    
    Args:
        history_df: Training history DataFrame (optional)
        eval_results: Evaluation results from eval_loop
        horizons: List of horizons
        output_dir: Output directory for plots
        predictions_df: Optional DataFrame with detailed predictions
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating plots in {output_dir}")
    
    # 1. Loss curves (if history available)
    if history_df is not None:
        plot_loss_curves(history_df, output_dir / 'loss_curves.png')
    else:
        logger.info("Skipping loss curves (no history provided)")
    
    # 2. Scatter plots (only if raw predictions available)
    if 'raw_predictions' in eval_results and 'raw_targets' in eval_results:
        plot_scatter_pred_vs_actual(
            eval_results['raw_predictions']['magnitude'],
            eval_results['raw_targets']['magnitude'],
            horizons,
            output_dir
        )
        
        # 3. Residual histograms
        plot_residual_histograms(
            eval_results['raw_predictions']['magnitude'],
            eval_results['raw_targets']['magnitude'],
            horizons,
            output_dir
        )
        
        # 4. Confusion matrices
        plot_confusion_matrices(
            eval_results['raw_predictions']['direction'],
            eval_results['raw_targets']['magnitude'],
            horizons,
            output_dir
        )
    else:
        logger.warning("Skipping scatter/residual/confusion plots: raw predictions not in eval_results")
    
    # 5. Price prediction plots (if predictions available)
    if predictions_df is not None:
        plot_price_predictions(
            predictions_df,
            horizons,
            output_dir
        )
    
    logger.info(f"All plots created in {output_dir}")
