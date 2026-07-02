"""
Metrics computation and export.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List
import logging

logger = logging.getLogger("stockxpert.reporting.metrics")

import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        return super().default(obj)


def compute_and_save_metrics(
    metrics_per_horizon: List[Dict],
    output_dir: Path
) -> None:
    """
    Compute and save metrics to JSON and CSV.
    
    Args:
        metrics_per_horizon: List of dicts with metrics for each horizon
        output_dir: Directory to save metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    metrics_json_path = output_dir / 'metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_per_horizon, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Metrics JSON saved to {metrics_json_path}")
    
    # Save as CSV
    metrics_csv_path = output_dir / 'per_horizon_metrics.csv'
    df = pd.DataFrame(metrics_per_horizon)
    df.to_csv(metrics_csv_path, index=False)
    
    logger.info(f"Metrics CSV saved to {metrics_csv_path}")
    
    # Print summary
    logger.info("Metrics Summary:")
    for metric in metrics_per_horizon:
        logger.info(
            f"  Horizon {metric['horizon']}: "
            f"Dir Acc={metric['direction_accuracy']:.3f}, "
            f"MAE={metric['mae']:.4f}, "
            f"RMSE={metric['rmse']:.4f}"
        )
