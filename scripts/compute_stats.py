#!/usr/bin/env python3
"""
Compute all statistical test results from existing predictions.csv.
Fills in TBD values for the paper revision document.

Includes:
  - Table 8: Binomial test for directional accuracy
  - Table 9: Bootstrap Sharpe ratio CI
  - Table 10: Precision/Recall for direction
  - Table 11: Information Coefficient (Spearman rank corr) per horizon
  - Table 12: Calibration analysis (predicted probability vs observed frequency)
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

PREDICTIONS_CSV = "/tmp/predictions.csv"
HORIZONS = [1, 3, 5, 7, 10]
RUNS_DIR = Path("runs")


def count_configurations(runs_dir: Path) -> int:
    """
    Count distinct parameter configurations evaluated during optimization.
    Scans runs/ for config.yaml files and counts unique (positions, horizons,
    confidence_threshold, stop_mult) combinations.
    """
    import yaml
    configs = set()
    # Enumerate all run config.yaml files
    for cfg_path in runs_dir.glob("*/config.yaml"):
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            positions = cfg.get("backtest", {}).get("max_positions", "?")
            horizons = tuple(sorted(cfg.get("features", {}).get("horizons", [])))
            lr = cfg.get("train", {}).get("lr", "?")
            hidden = cfg.get("model", {}).get("hidden_dim", "?")
            configs.add((positions, horizons, lr, hidden))
        except Exception:
            pass

    # Also count the ablation sweep dimensions from the paper:
    # positions: [3,5,8,10], horizons: [H1,H3,H5,H7,H10] subsets,
    # confidence thresholds: [0.58,0.60,0.65,0.70], stop ATR mults: [1.0,1.5,2.0,3.0]
    # This reflects the actual sweep documented in the paper.
    positions_swept = [3, 5, 8, 10]
    horizon_subsets = [[1], [3], [5], [7], [10], [5, 7], [7, 10], [1, 3, 5, 7, 10]]
    confidence_thresholds = [0.55, 0.58, 0.60, 0.65, 0.70]
    stop_atr_mults = [1.0, 1.5, 2.0, 3.0]
    n_sweep = len(positions_swept) * len(horizon_subsets) * len(confidence_thresholds) * len(stop_atr_mults)
    n_run_configs = max(len(configs), 1)
    return max(n_sweep, n_run_configs)


def deflated_sharpe(point_sharpe: float, n_configs: int, n_observations: int, p: float = 0.05) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    DSR = SR* × (1 − (k/N) × log(1/p))
    where k = number of configurations evaluated,
          N = number of daily return observations,
          p = significance level (0.05).

    Corrects for multiple-testing bias when the reported Sharpe is the best
    across many configurations.
    """
    import math
    if n_observations <= 0 or n_configs <= 0:
        return point_sharpe
    deflation = 1 - (n_configs / n_observations) * math.log(1 / p)
    return point_sharpe * deflation


def bootstrap_ci_metrics(returns: np.ndarray, n_bootstrap: int = 10000, ci: float = 0.95):
    """
    Bootstrap 95% CI on return, Sharpe, max-drawdown, and win-rate simultaneously.
    """
    np.random.seed(42)
    n = len(returns)
    alpha = (1 - ci) / 2

    boot_returns, boot_sharpes, boot_maxdds = [], [], []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=n, replace=True)
        mean_ret = np.mean(sample)
        std_ret = np.std(sample, ddof=1)
        ann_sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
        # Max drawdown on bootstrap path
        cum = np.cumprod(1 + sample)
        peak = np.maximum.accumulate(cum)
        dd = np.max((peak - cum) / peak) * 100 if len(cum) > 0 else 0
        boot_returns.append(mean_ret * 252 * 100)
        boot_sharpes.append(ann_sharpe)
        boot_maxdds.append(dd)

    return {
        "return_ci_low": float(np.percentile(boot_returns, alpha * 100)),
        "return_ci_high": float(np.percentile(boot_returns, (1 - alpha) * 100)),
        "sharpe_ci_low": float(np.percentile(boot_sharpes, alpha * 100)),
        "sharpe_ci_high": float(np.percentile(boot_sharpes, (1 - alpha) * 100)),
        "maxdd_ci_low": float(np.percentile(boot_maxdds, alpha * 100)),
        "maxdd_ci_high": float(np.percentile(boot_maxdds, (1 - alpha) * 100)),
        "n_bootstrap": n_bootstrap,
    }


def binomial_test_directional(df, horizon):
    """Binomial test: is direction accuracy > 50%?"""
    p_up_col = f"p_up_h{horizon}"
    actual_z_col = f"actual_z_h{horizon}"
    
    preds = df[p_up_col].values
    actuals = df[actual_z_col].values
    
    # Direction: pred up if p_up > 0.5, actual up if actual_z > 0
    pred_up = preds > 0.5
    actual_up = actuals > 0
    
    correct = (pred_up == actual_up)
    n_correct = correct.sum()
    n_total = len(correct)
    accuracy = n_correct / n_total
    
    # Binomial test: H0: p = 0.5
    pvalue = stats.binomtest(n_correct, n_total, p=0.5, alternative='greater').pvalue
    
    # Wilson confidence interval
    z = 1.96  # 95%
    p_hat = accuracy
    denom = 1 + z**2 / n_total
    center = (p_hat + z**2 / (2 * n_total)) / denom
    spread = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total) / denom
    ci_low = center - spread
    ci_high = center + spread
    
    return {
        "horizon": horizon,
        "accuracy": accuracy,
        "n_correct": int(n_correct),
        "n_total": int(n_total),
        "p_value": pvalue,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def precision_recall_direction(df, horizon):
    """Compute precision/recall/F1 for UP direction prediction."""
    p_up_col = f"p_up_h{horizon}"
    actual_z_col = f"actual_z_h{horizon}"
    
    pred_up = df[p_up_col].values > 0.5
    actual_up = df[actual_z_col].values > 0
    
    tp = int(np.sum(pred_up & actual_up))
    fp = int(np.sum(pred_up & ~actual_up))
    tn = int(np.sum(~pred_up & ~actual_up))
    fn = int(np.sum(~pred_up & actual_up))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def bootstrap_sharpe(returns, n_bootstrap=10000, ci=0.95):
    """Bootstrap confidence interval for Sharpe ratio."""
    np.random.seed(42)
    n = len(returns)
    sharpe_samples = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(returns, size=n, replace=True)
        mean = np.mean(sample)
        std = np.std(sample, ddof=1)
        if std > 0:
            sharpe_samples.append(mean / std * np.sqrt(252))
    
    sharpe_samples = np.array(sharpe_samples)
    alpha = (1 - ci) / 2
    ci_low = np.percentile(sharpe_samples, alpha * 100)
    ci_high = np.percentile(sharpe_samples, (1 - alpha) * 100)
    
    return {
        "sharpe_point": float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_std": float(np.std(sharpe_samples)),
        "n_resamples": n_bootstrap,
    }


def information_coefficient_per_horizon(df, horizon):
    """Compute Information Coefficient (Spearman rank corr) between predicted and actual z-scores."""
    pred_z_col = f"pred_z_h{horizon}"
    actual_z_col = f"actual_z_h{horizon}"

    pred = df[pred_z_col].values
    actual = df[actual_z_col].values

    # Full-sample IC
    ic, p_value = stats.spearmanr(pred, actual)

    # Rolling IC (monthly windows) for stability analysis
    df_sorted = df.sort_values("date").copy()
    df_sorted["month"] = pd.to_datetime(df_sorted["date"]).dt.to_period("M")
    monthly_ics = []
    for _, group in df_sorted.groupby("month"):
        if len(group) >= 20:
            m_ic, _ = stats.spearmanr(group[pred_z_col].values, group[actual_z_col].values)
            if not np.isnan(m_ic):
                monthly_ics.append(m_ic)

    return {
        "horizon": horizon,
        "ic": float(ic),
        "p_value": float(p_value),
        "monthly_ic_mean": float(np.mean(monthly_ics)) if monthly_ics else None,
        "monthly_ic_std": float(np.std(monthly_ics)) if monthly_ics else None,
        "monthly_ic_hit_rate": float(np.mean(np.array(monthly_ics) > 0)) if monthly_ics else None,
        "n_months": len(monthly_ics),
    }


def calibration_analysis(df, horizon, n_bins=10):
    """Compute calibration curve: binned P(Up) vs observed frequency."""
    p_up_col = f"p_up_h{horizon}"
    actual_z_col = f"actual_z_h{horizon}"

    p_up = df[p_up_col].values
    actual_up = (df[actual_z_col].values > 0).astype(float)

    # Brier score
    brier_score = float(np.mean((p_up - actual_up) ** 2))

    # Bin predictions into deciles
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (p_up >= lo) & (p_up < hi) if i < n_bins - 1 else (p_up >= lo) & (p_up <= hi)
        n_in_bin = int(mask.sum())
        if n_in_bin > 0:
            mean_predicted = float(np.mean(p_up[mask]))
            observed_freq = float(np.mean(actual_up[mask]))
        else:
            mean_predicted = (lo + hi) / 2
            observed_freq = None
        bins.append({
            "bin_low": float(lo),
            "bin_high": float(hi),
            "n_samples": n_in_bin,
            "mean_predicted": mean_predicted,
            "observed_frequency": observed_freq,
        })

    # Expected Calibration Error (ECE)
    total = len(p_up)
    ece = sum(
        (b["n_samples"] / total) * abs(b["mean_predicted"] - b["observed_frequency"])
        for b in bins if b["observed_frequency"] is not None and b["n_samples"] > 0
    )

    return {
        "horizon": horizon,
        "brier_score": brier_score,
        "expected_calibration_error": float(ece),
        "bins": bins,
    }


def main():
    print("=" * 70)
    print("STATISTICAL ANALYSIS FOR PAPER REVISION")
    print("=" * 70)
    
    df = pd.read_csv(PREDICTIONS_CSV)
    print(f"\nLoaded {len(df)} predictions from {len(df['symbol'].unique())} symbols\n")
    
    # ===== Table 8: Statistical Significance of Directional Accuracy =====
    print("=" * 70)
    print("TABLE 8: Statistical Significance of Directional Accuracy")
    print("=" * 70)
    print(f"{'Horizon':>8} | {'Accuracy':>10} | {'n_correct/n_total':>17} | {'p-value':>12} | {'95% CI (Wilson)':>20}")
    print("-" * 75)
    
    sig_results = []
    for h in HORIZONS:
        result = binomial_test_directional(df, h)
        sig_results.append(result)
        print(
            f"  H{h:>2}    | {result['accuracy']:>9.2%} | "
            f"{result['n_correct']:>6}/{result['n_total']:<6}    | "
            f"{result['p_value']:>12.2e} | "
            f"[{result['ci_low']:.4f}, {result['ci_high']:.4f}]"
        )
    
    # ===== Table 10: Precision/Recall for H1 =====
    print(f"\n{'=' * 70}")
    print("TABLE 10: Precision/Recall for Directional Prediction (H1)")
    print("=" * 70)
    
    pr = precision_recall_direction(df, 1)
    print(f"  Precision (UP class): {pr['precision']:.4f}")
    print(f"  Recall (UP class):    {pr['recall']:.4f}")
    print(f"  F1-Score:             {pr['f1']:.4f}")
    print(f"  True Positives:       {pr['tp']}")
    print(f"  False Positives:      {pr['fp']}")
    print(f"  True Negatives:       {pr['tn']}")
    print(f"  False Negatives:      {pr['fn']}")
    
    # ===== Table 9: Bootstrap Sharpe Ratio =====
    # Use H1 predicted returns as a proxy for strategy returns
    print(f"\n{'=' * 70}")
    print("TABLE 9: Bootstrap Sharpe Ratio CI")
    print("=" * 70)
    
    # Compute daily strategy returns based on H1 signals
    df_sorted = df.sort_values(["symbol", "date"])
    
    # Simple long/short strategy: go long if p_up > 0.55, short if p_up < 0.45
    strategy_returns = []
    for _, row in df_sorted.iterrows():
        actual_return = row["actual_z_h1"] * row["vol_ref"]  # Convert z-score to return
        p_up = row["p_up_h1"]
        confidence = row["confidence_h1"]
        
        # Only trade with confidence > 0.6 and p_up threshold
        if confidence > 0.6 and p_up > 0.55:
            strategy_returns.append(actual_return - 0.001)  # Long - transaction cost
        elif confidence > 0.6 and p_up < 0.45:
            strategy_returns.append(-actual_return - 0.001)  # Short - transaction cost
        else:
            strategy_returns.append(0.0)  # No trade
    
    strategy_returns = np.array(strategy_returns)
    non_zero_returns = strategy_returns[strategy_returns != 0]
    
    if len(non_zero_returns) > 10:
        sharpe = bootstrap_sharpe(non_zero_returns)
        print(f"  Sharpe Ratio (point): {sharpe['sharpe_point']:.4f}")
        print(f"  Bootstrap 95% CI:     [{sharpe['ci_low']:.4f}, {sharpe['ci_high']:.4f}]")
        print(f"  Bootstrap Std:        {sharpe['bootstrap_std']:.4f}")
        print(f"  Number of resamples:  {sharpe['n_resamples']}")
        print(f"  Number of trades:     {len(non_zero_returns)}")
    else:
        sharpe = {"sharpe_point": "N/A", "ci_low": "N/A", "ci_high": "N/A", "bootstrap_std": "N/A", "n_resamples": 10000}
        print("  Insufficient trades for Sharpe ratio computation")
    
    # ===== A4: Deflated Sharpe + Config Count =====
    print(f"\n{'=' * 70}")
    print("A4: DEFLATED SHARPE RATIO + CONFIGURATION COUNT")
    print("=" * 70)

    n_configs = count_configurations(RUNS_DIR)
    print(f"  Parameter configurations swept: {n_configs}")
    print(f"  (positions × horizon-subsets × confidence-thresholds × stop-mults)")

    n_obs = len(non_zero_returns) if len(non_zero_returns) > 10 else len(strategy_returns)
    if isinstance(sharpe.get("sharpe_point"), float):
        dsr = deflated_sharpe(
            point_sharpe=sharpe["sharpe_point"],
            n_configs=n_configs,
            n_observations=n_obs,
        )
        print(f"  Point Sharpe:     {sharpe['sharpe_point']:.2f}")
        print(f"  Deflated Sharpe:  {dsr:.2f}  (k={n_configs}, N={n_obs}, p=0.05)")
        print(f"  [Deflation factor: (1 - {n_configs}/{n_obs} × log(20)) = {1 - (n_configs/n_obs)*2.996:.4f}]")
    else:
        dsr = None
        print("  Insufficient data for deflated Sharpe.")

    # ===== A4: Bootstrap CIs on all headline metrics =====
    if len(non_zero_returns) > 30:
        ci_metrics = bootstrap_ci_metrics(non_zero_returns)
        print(f"\n  Bootstrap 95% CIs (N={len(non_zero_returns)} strategy returns, 10k resamples):")
        print(f"  Annualised Return: [{ci_metrics['return_ci_low']:.2f}%, {ci_metrics['return_ci_high']:.2f}%]")
        print(f"  Sharpe Ratio:      [{ci_metrics['sharpe_ci_low']:.2f}, {ci_metrics['sharpe_ci_high']:.2f}]")
        print(f"  Max Drawdown:      [{ci_metrics['maxdd_ci_low']:.2f}%, {ci_metrics['maxdd_ci_high']:.2f}%]")
    else:
        ci_metrics = {}
        print("  Insufficient strategy returns for bootstrap CI.")

    # Save all results as JSON
    output = {
        "table_8_significance": sig_results,
        "table_9_sharpe_ci": sharpe,
        "table_10_precision_recall": pr,
        "a4_config_count": n_configs,
        "a4_deflated_sharpe": dsr,
        "a4_bootstrap_cis": ci_metrics,
    }

    # ===== Table 11: Information Coefficient =====
    print(f"\n{'=' * 70}")
    print("TABLE 11: Information Coefficient (Spearman Rank Correlation)")
    print("=" * 70)
    print(f"{'Horizon':>8} | {'IC':>8} | {'p-value':>12} | {'Monthly IC Mean':>16} | {'IC Hit Rate':>12} | {'Months':>6}")
    print("-" * 75)

    ic_results = []
    for h in HORIZONS:
        ic_res = information_coefficient_per_horizon(df, h)
        ic_results.append(ic_res)
        monthly_mean = f"{ic_res['monthly_ic_mean']:.4f}" if ic_res['monthly_ic_mean'] is not None else "N/A"
        hit_rate = f"{ic_res['monthly_ic_hit_rate']:.1%}" if ic_res['monthly_ic_hit_rate'] is not None else "N/A"
        print(
            f"  H{h:>2}    | {ic_res['ic']:>7.4f} | {ic_res['p_value']:>12.2e} | "
            f"{monthly_mean:>16} | {hit_rate:>12} | {ic_res['n_months']:>6}"
        )
    output["table_11_information_coefficient"] = ic_results

    # ===== Table 12: Calibration Analysis =====
    print(f"\n{'=' * 70}")
    print("TABLE 12: Calibration Analysis (H1)")
    print("=" * 70)

    cal_h1 = calibration_analysis(df, 1)
    print(f"  Brier Score:                    {cal_h1['brier_score']:.4f}")
    print(f"  Expected Calibration Error:     {cal_h1['expected_calibration_error']:.4f}")
    print(f"\n  {'Bin':>12} | {'N':>6} | {'Mean P(Up)':>10} | {'Obs Freq':>10} | {'Gap':>8}")
    print("  " + "-" * 55)
    for b in cal_h1['bins']:
        bin_label = f"[{b['bin_low']:.1f}-{b['bin_high']:.1f})"
        obs = f"{b['observed_frequency']:.4f}" if b['observed_frequency'] is not None else "N/A"
        gap = f"{abs(b['mean_predicted'] - b['observed_frequency']):.4f}" if b['observed_frequency'] is not None else "N/A"
        print(f"  {bin_label:>12} | {b['n_samples']:>6} | {b['mean_predicted']:>10.4f} | {obs:>10} | {gap:>8}")

    # Calibration for all horizons (summary)
    cal_results = []
    print(f"\n  Calibration Summary (all horizons):")
    print(f"  {'Horizon':>8} | {'Brier':>8} | {'ECE':>8}")
    print("  " + "-" * 30)
    for h in HORIZONS:
        cal = calibration_analysis(df, h)
        cal_results.append(cal)
        print(f"  H{h:>2}      | {cal['brier_score']:>7.4f} | {cal['expected_calibration_error']:>7.4f}")
    output["table_12_calibration"] = cal_results

    # Save
    output_path = Path("reports/statistical_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nAll results saved to {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
