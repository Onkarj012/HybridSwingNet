"""
Statistical significance testing utilities for StockXpert.

Provides:
  - Bootstrap confidence intervals for Sharpe ratio
  - Paired return t-test (model vs benchmark)
  - Binomial test for directional accuracy vs 50%
  - Diebold-Mariano test for forecast comparison
"""

import numpy as np
from scipy import stats
from typing import Tuple, Dict, Optional


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    risk_free_rate: float = 0.0,
    annualization_factor: float = 252,
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for the annualized Sharpe ratio.

    Args:
        returns: Daily return series.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        risk_free_rate: Daily risk-free rate.
        annualization_factor: Trading days per year.

    Returns:
        Dictionary with 'sharpe', 'ci_lower', 'ci_upper', 'std'.
    """
    returns = np.asarray(returns, dtype=np.float64)
    excess = returns - risk_free_rate
    n = len(returns)

    # Point estimate
    sharpe = (excess.mean() / excess.std()) * np.sqrt(annualization_factor) if excess.std() > 0 else 0.0

    # Bootstrap
    rng = np.random.default_rng(42)
    bootstrap_sharpes = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(excess, size=n, replace=True)
        std = sample.std()
        bootstrap_sharpes[i] = (sample.mean() / std) * np.sqrt(annualization_factor) if std > 0 else 0.0

    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_sharpes, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_sharpes, 100 * (1 - alpha / 2)))

    return {
        "sharpe": float(sharpe),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": float(np.std(bootstrap_sharpes)),
    }


def paired_return_test(
    model_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> Dict[str, float]:
    """
    Paired t-test: Is the model's return significantly different from benchmark?

    Args:
        model_returns: Daily returns of model strategy.
        benchmark_returns: Daily returns of benchmark (buy-and-hold).

    Returns:
        Dictionary with 't_stat', 'p_value', 'mean_diff', 'ci_lower', 'ci_upper'.
    """
    model_returns = np.asarray(model_returns, dtype=np.float64)
    benchmark_returns = np.asarray(benchmark_returns, dtype=np.float64)

    diff = model_returns - benchmark_returns
    t_stat, p_value = stats.ttest_1samp(diff, 0)

    n = len(diff)
    mean_diff = diff.mean()
    se = diff.std(ddof=1) / np.sqrt(n)
    ci_lower = mean_diff - 1.96 * se
    ci_upper = mean_diff + 1.96 * se

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": float(mean_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def direction_accuracy_test(
    n_correct: int,
    n_total: int,
    null_probability: float = 0.5,
) -> Dict[str, float]:
    """
    Binomial test: Is directional accuracy significantly better than random (50%)?

    Args:
        n_correct: Number of correct direction predictions.
        n_total: Total number of predictions.
        null_probability: Null hypothesis probability (default: 0.5 = random).

    Returns:
        Dictionary with 'accuracy', 'p_value', 'ci_lower', 'ci_upper'.
    """
    result = stats.binomtest(n_correct, n_total, null_probability, alternative="greater")

    accuracy = n_correct / n_total
    ci = result.proportion_ci(confidence_level=0.95, method="wilson")

    return {
        "accuracy": float(accuracy),
        "p_value": float(result.pvalue),
        "ci_lower": float(ci.low),
        "ci_upper": float(ci.high),
        "n_correct": int(n_correct),
        "n_total": int(n_total),
    }


def diebold_mariano_test(
    errors_model: np.ndarray,
    errors_benchmark: np.ndarray,
    power: int = 2,
) -> Dict[str, float]:
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Tests H0: equal predictive accuracy.

    Args:
        errors_model: Forecast errors from model.
        errors_benchmark: Forecast errors from benchmark.
        power: Loss function power (1 = MAE, 2 = MSE).

    Returns:
        Dictionary with 'dm_stat', 'p_value'.
    """
    d = np.abs(errors_benchmark) ** power - np.abs(errors_model) ** power
    n = len(d)
    mean_d = d.mean()
    var_d = np.var(d, ddof=1) / n

    dm_stat = mean_d / np.sqrt(var_d) if var_d > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "mean_loss_diff": float(mean_d),
    }


def information_coefficient(
    predicted_returns: np.ndarray,
    actual_returns: np.ndarray,
) -> Dict[str, float]:
    """
    Compute Information Coefficient (Spearman rank correlation).

    Args:
        predicted_returns: Predicted returns / z-scores.
        actual_returns: Actual realized returns.

    Returns:
        Dictionary with 'ic', 'p_value'.
    """
    corr, p_value = stats.spearmanr(predicted_returns, actual_returns)
    return {
        "ic": float(corr),
        "p_value": float(p_value),
    }


def compute_precision_recall_direction(
    predicted_direction: np.ndarray,
    actual_direction: np.ndarray,
) -> Dict[str, float]:
    """
    Precision and Recall for directional prediction (UP class).

    Args:
        predicted_direction: 1 for UP, 0 for DOWN.
        actual_direction: 1 for UP, 0 for DOWN.

    Returns:
        Dictionary with 'precision', 'recall', 'f1', 'accuracy'.
    """
    tp = int(np.sum((predicted_direction == 1) & (actual_direction == 1)))
    fp = int(np.sum((predicted_direction == 1) & (actual_direction == 0)))
    fn = int(np.sum((predicted_direction == 0) & (actual_direction == 1)))
    tn = int(np.sum((predicted_direction == 0) & (actual_direction == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
