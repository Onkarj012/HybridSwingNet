#!/usr/bin/env python3
"""
Generate publication-quality figures for the regime-filtered production backtesting strategy.
All data sourced from the comprehensive backtest record (runs/comparison_report.md, PAPER_REVISION_GUIDE.md).

Output: docs/Paper/figures/fig_strategy_*.png (7 figures, 300 DPI)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "docs/Paper/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# B6: Canonical run directories — set via --canonical-run <path> or via these defaults.
# Figures load metrics from these CSVs; hardcoded fallbacks retained only as last resort.
BT_2024_DIR = PROJECT / "runs/historical_backtest_20260506_135936"
BT_6PROF_DIR = PROJECT / "runs/historical_backtest_20260505_145309"
BT_2025EQUITY_DIR = PROJECT / "runs/historical_backtest_20260505_132710"

# ── B6: Run-data loader ─────────────────────────────────────────────────────
def load_profile_metrics(run_dir: Path, profile_name: str) -> dict:
    """Load backtest metrics from a run's trade_log CSV. Returns {} on failure."""
    trade_log = run_dir / f"trade_log_{profile_name.lower()}.csv"
    equity_csv = run_dir / f"equity_curve_{profile_name.lower()}.csv"
    if not trade_log.exists():
        return {}
    try:
        trades = pd.read_csv(trade_log)
        if trades.empty:
            return {}
        pnls = trades["PnL"].astype(float).values
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        win_rate = len(wins) / len(pnls) * 100
        pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")

        eq_vals = None
        if equity_csv.exists():
            eq_df = pd.read_csv(equity_csv)
            eq_vals = eq_df["Portfolio_Value"].values

        total_return = (eq_vals[-1] / eq_vals[0] - 1) * 100 if eq_vals is not None and len(eq_vals) > 1 else 0.0
        daily_rets = np.diff(eq_vals) / eq_vals[:-1] if eq_vals is not None and len(eq_vals) > 1 else np.array([0.0])
        sharpe = float(np.mean(daily_rets) / (np.std(daily_rets) + 1e-10) * np.sqrt(252))
        peak = np.maximum.accumulate(eq_vals) if eq_vals is not None else np.array([1.0])
        max_dd = float(np.max((peak - eq_vals) / peak) * 100) if eq_vals is not None else 0.0

        return {
            "total_return_pct": round(total_return, 2),
            "sharpe": round(sharpe, 2),
            "max_dd": round(max_dd, 1),
            "win_rate": round(win_rate, 1),
            "trades": int(len(pnls)),
            "profit_factor": round(pf, 2),
            "avg_win": round(float(wins.mean()) if len(wins) else 0.0, 0),
            "avg_loss": round(float(losses.mean()) if len(losses) else 0.0, 0),
        }
    except Exception as e:
        print(f"  [warn] load_profile_metrics({run_dir.name}, {profile_name}): {e}")
        return {}

CB2 = ["#0072B2", "#009E73", "#F0E442", "#D55E00", "#CC79A7"]
CB3 = ["#0072B2", "#D55E00", "#009E73"]

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC",
})
sns.set_style("whitegrid")

def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {name}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Regime Filter Impact — Before vs After
# ═══════════════════════════════════════════════════════════════════════════
def fig_regime_filter_impact():
    periods = ["2025\n(Validation)", "2026\n(Test + OOS)"]
    without_regime = [15.90, -5.59]
    with_regime = [23.85, 3.04]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(periods))
    w = 0.35

    bars1 = ax.bar(x - w/2, without_regime, w, label="Without Regime Filter",
                   color="#D55E00", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + w/2, with_regime, w, label="With Regime Filter (ON)",
                   color="#0072B2", edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(periods, fontsize=10)
    ax.set_ylabel("Total Return (%)", fontsize=11, fontweight="bold")
    ax.set_title("Impact of Market Regime Filter on Conservative Strategy\n(H7/H10, 3 pos, Rs.10L, ALL 84 Stocks)",
                 fontsize=12, fontweight="bold")
    ax.axhline(y=0, color="black", linewidth=1)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    for bar, val in zip(bars1, without_regime):
        clr = "#C62828" if val < 0 else "#2E7D32"
        ypos = val + (1.5 if val >= 0 else -3)
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f"{val:+.2f}%",
                ha="center", va=va, fontsize=10, fontweight="bold", color=clr)
    for bar, val in zip(bars2, with_regime):
        clr = "#C62828" if val < 0 else "#2E7D32"
        ypos = val + (1.5 if val >= 0 else -3)
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f"{val:+.2f}%",
                ha="center", va=va, fontsize=10, fontweight="bold", color=clr)

    ax.annotate("+8.0pp", xy=(0, 19.9), xytext=(0, 26),
                fontsize=9, fontweight="bold", color="#0072B2",
                ha="center", arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.5))
    ax.annotate("+8.6pp", xy=(1, -1.3), xytext=(1, 10),
                fontsize=9, fontweight="bold", color="#0072B2",
                ha="center", arrowprops=dict(arrowstyle="->", color="#0072B2", lw=1.5))

    ax.set_ylim(-10, 30)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%+.0f%%'))
    plt.tight_layout()
    save(fig, "fig_regime_filter_impact.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Position Count — Signal Quality Decay
# ═══════════════════════════════════════════════════════════════════════════
def fig_position_quality_decay():
    # B6: load from canonical run; fall back to paper values if run dir absent
    positions = [3, 5, 8, 10]
    _prof_names = ["conservative", "moderate", "aggressive", "max_diversified"]
    _loaded = [load_profile_metrics(BT_6PROF_DIR, p) for p in _prof_names]
    returns  = [d.get("total_return_pct", fb) for d, fb in zip(_loaded, [28.77, 15.08, 9.03, 9.72])]
    sharpes  = [d.get("sharpe", fb) for d, fb in zip(_loaded, [2.94, 2.07, 1.46, 1.67])]
    winrates = [d.get("win_rate", fb) for d, fb in zip(_loaded, [60.2, 51.4, 46.7, 49.5])]
    trades   = [d.get("trades", fb) for d, fb in zip(_loaded, [83, 138, 225, 279])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: Return + Sharpe
    color_ret = "#0072B2"
    ax1_ret = ax1
    ax1_sharpe = ax1.twinx()

    bars = ax1_ret.bar(np.array(positions) - 0.15, returns, 0.3,
                       color=color_ret, edgecolor="white", linewidth=0.8, label="Return (%)")
    ax1_ret.plot(positions, returns, 'o-', color=color_ret, linewidth=2, markersize=8)
    ax1_ret.set_xlabel("Number of Positions", fontsize=11, fontweight="bold")
    ax1_ret.set_ylabel("Return (%)", fontsize=11, fontweight="bold", color=color_ret)
    ax1_ret.tick_params(axis="y", labelcolor=color_ret)
    ax1_ret.set_xticks(positions)

    ax1_sharpe.plot(positions, sharpes, 's--', color="#D55E00", linewidth=2, markersize=8, label="Sharpe Ratio")
    ax1_sharpe.set_ylabel("Sharpe Ratio", fontsize=11, fontweight="bold", color="#D55E00")
    ax1_sharpe.tick_params(axis="y", labelcolor="#D55E00")
    ax1_sharpe.set_ylim(0, 4.0)
    ax1_ret.set_title("Return & Sharpe by Position Count", fontsize=12, fontweight="bold")

    for i, (r, s) in enumerate(zip(returns, sharpes)):
        ax1_ret.text(positions[i] - 0.15, r + 1, f"{r:.1f}%", ha="center", fontsize=8, fontweight="bold", color=color_ret)
        ax1_sharpe.text(positions[i], s + 0.12, f"{s:.2f}", ha="center", fontsize=8, fontweight="bold", color="#D55E00")

    lines1, labels1 = ax1_ret.get_legend_handles_labels()
    lines2, labels2 = ax1_sharpe.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    # Right: Win Rate + Trade Count
    color_wr = "#009E73"
    ax2_wr = ax2
    ax2_trades = ax2.twinx()

    ax2_wr.plot(positions, winrates, 'D-', color=color_wr, linewidth=2, markersize=8, label="Win Rate (%)")
    ax2_wr.set_xlabel("Number of Positions", fontsize=11, fontweight="bold")
    ax2_wr.set_ylabel("Win Rate (%)", fontsize=11, fontweight="bold", color=color_wr)
    ax2_wr.tick_params(axis="y", labelcolor=color_wr)
    ax2_wr.set_xticks(positions)
    ax2_wr.axhline(y=50, color="grey", linestyle="--", alpha=0.5, label="Chance (50%)")

    ax2_trades.bar(np.array(positions) + 0.15, trades, 0.3,
                   color="#CC79A7", edgecolor="white", linewidth=0.8, alpha=0.6, label="Trades")
    ax2_trades.set_ylabel("Number of Trades", fontsize=11, fontweight="bold", color="#CC79A7")
    ax2_trades.tick_params(axis="y", labelcolor="#CC79A7")
    ax2_trades.set_ylim(0, 350)
    ax2_wr.set_title("Win Rate & Trade Count by Position Count", fontsize=12, fontweight="bold")

    for i, (w, t) in enumerate(zip(winrates, trades)):
        ax2_wr.text(positions[i], w + 1, f"{w:.1f}%", ha="center", fontsize=8, fontweight="bold", color=color_wr)
        ax2_trades.text(positions[i] + 0.15, t + 12, str(t), ha="center", fontsize=7, color="#CC79A7")

    lines1, labels1 = ax2_wr.get_legend_handles_labels()
    lines2, labels2 = ax2_trades.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    fig.suptitle("Signal Quality Collapses Beyond 3 Positions\n(Conservative, H7/H10, Regime ON, Rs.10L, 2025)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    save(fig, "fig_position_quality_decay.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Lever Impact Ranking — What Actually Works
# ═══════════════════════════════════════════════════════════════════════════
def fig_lever_impact_ranking():
    levers = [
        ("Regime Filter\n(skip bear entries)", 18.0, "#0072B2"),
        ("3 Positions Only\n(vs 5 positions)", 9.0, "#0072B2"),
        ("Wide Stops\n(3×/2× ATR)", 6.0, "#0072B2"),
        ("H7/H10 Horizons\n(vs H5/H7)", 4.0, "#0072B2"),
        ("Rs.10L Capital\n(vs Rs.1L)", 4.0, "#0072B2"),
        ("ALL 84 Stocks\n(vs sub-universe)", 3.0, "#0072B2"),
        ("0.70/0.30 Confidence\n(vs 0.65/0.35)", 2.0, "#0072B2"),
    ]
    antipatterns = [
        ("Kelly Sizing\n(equal-weight far better)", -20.0, "#D55E00"),
        ("Trailing Stops\n(cuts winners early)", -15.0, "#D55E00"),
        ("H1/H3 Horizons\n(noise factories)", -10.0, "#D55E00"),
        ("Dynamic Stock Filter\n(past accuracy ≠ future)", -5.0, "#D55E00"),
        ("More Positions (8+)\n(dilutes signal quality)", -5.0, "#D55E00"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))

    all_labels = [l[0] for l in levers + antipatterns]
    all_values = [l[1] for l in levers + antipatterns]
    all_colors = [l[2] for l in levers + antipatterns]

    y_pos = range(len(all_labels))
    bars = ax.barh(y_pos, all_values, color=all_colors, edgecolor="white", linewidth=0.8, height=0.65)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_labels, fontsize=9)
    ax.axvline(x=0, color="black", linewidth=1.5)
    ax.set_xlabel("Combined Return Impact (percentage points)", fontsize=11, fontweight="bold")
    ax.set_title("Lever Impact Ranking: What Improves vs Destroys Strategy Returns\n(2025–2026 Combined Backtest)",
                 fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    for bar, val in zip(bars, all_values):
        if val > 0:
            ax.text(val + 0.4, bar.get_y() + bar.get_height()/2,
                    f"+{val:.0f}pp", va="center", fontsize=9, fontweight="bold", color="#0072B2")
        else:
            ax.text(val - 0.4, bar.get_y() + bar.get_height()/2,
                    f"{val:.0f}pp", va="center", ha="right", fontsize=9, fontweight="bold", color="#D55E00")

    ax.text(0.02, 0.97, "[+] HELPS", transform=ax.transAxes, fontsize=10, fontweight="bold",
            color="#0072B2", va="top", ha="left")
    ax.text(0.02, 0.45, "[-] HARMS", transform=ax.transAxes, fontsize=10, fontweight="bold",
            color="#D55E00", va="top", ha="left")

    ax.set_xlim(-25, 24)
    plt.tight_layout()
    save(fig, "fig_lever_impact_ranking.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Cumulative Equity Curve — Production Strategy Across Periods
# ═══════════════════════════════════════════════════════════════════════════
def fig_cumulative_equity_strategy():
    # Load actual equity curve from 2024 production run
    eq_file = BT_2024_DIR / "equity_curve_custom.csv"
    if eq_file.exists():
        eq_2024 = pd.read_csv(eq_file, parse_dates=["Date"])
        eq_2024 = eq_2024.set_index("Date")
    else:
        eq_2024 = None

    # 2025 conservative equity
    eq_file_2025 = BT_2025EQUITY_DIR / "equity_curve_conservative.csv"
    if eq_file_2025.exists():
        eq_2025 = pd.read_csv(eq_file_2025, parse_dates=["Date"])
        eq_2025 = eq_2025.set_index("Date")
    else:
        eq_2025 = None

    fig, ax = plt.subplots(figsize=(10, 4.8))

    if eq_2024 is not None:
        cumulative = eq_2024["Portfolio_Value"] / eq_2024["Portfolio_Value"].iloc[0] * 100 - 100
        ax.plot(eq_2024.index, cumulative, color=CB2[0], linewidth=1.8, label="2024 (in-sample)")

    if eq_2025 is not None:
        cumulative_2025 = eq_2025["Portfolio_Value"] / eq_2025["Portfolio_Value"].iloc[0] * 100 - 100
        ax.plot(eq_2025.index, cumulative_2025, color=CB2[1], linewidth=1.8, label="2025 (validation)")

    # Add 2026 as a simpler segment if available, or annotate
    # Note: 2026 data may not be in the same run

    ax.axhline(y=0, color="grey", linestyle="--", alpha=0.4)
    ax.set_ylabel("Cumulative Return (%)", fontsize=11, fontweight="bold")
    ax.set_title("Cumulative Equity Curve — Regime-Filtered Conservative Strategy\n(H7/H10, 3 Positions, Rs.10L, 84 Stocks)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%+.0f%%'))
    ax.grid(True, alpha=0.3)

    # Annotate final returns
    if eq_2024 is not None:
        final_2024 = eq_2024["Portfolio_Value"].iloc[-1] / eq_2024["Portfolio_Value"].iloc[0] * 100 - 100
        ax.annotate(f"+{final_2024:.1f}%", xy=(eq_2024.index[-1], final_2024),
                    fontsize=10, fontweight="bold", color=CB2[0], ha="left")
    if eq_2025 is not None:
        final_2025 = eq_2025["Portfolio_Value"].iloc[-1] / eq_2025["Portfolio_Value"].iloc[0] * 100 - 100
        ax.annotate(f"+{final_2025:.1f}%", xy=(eq_2025.index[-1], final_2025),
                    fontsize=10, fontweight="bold", color=CB2[1], ha="left")

    plt.tight_layout()
    save(fig, "fig_cumulative_equity_strategy.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Capital Scaling Effect
# ═══════════════════════════════════════════════════════════════════════════
def fig_capital_scaling():
    # B6: load from canonical runs; fall back to paper values if run dirs absent
    _c1 = load_profile_metrics(BT_6PROF_DIR, "conservative")
    _c5 = load_profile_metrics(BT_6PROF_DIR, "moderate")
    _c10 = load_profile_metrics(BT_6PROF_DIR, "max_diversified")
    _c1_26 = load_profile_metrics(BT_2024_DIR, "conservative")
    capitals = ["Rs.1L", "Rs.5L", "Rs.10L"]
    ret_2025 = [_c1.get("total_return_pct", 23.85), _c5.get("total_return_pct", 28.47), _c10.get("total_return_pct", 28.77)]
    ret_2026 = [_c1_26.get("total_return_pct", 3.04), 2.71, 2.72]
    combined = [r25 + r26 for r25, r26 in zip(ret_2025, ret_2026)]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(capitals))
    w = 0.25

    ax.bar(x - w, ret_2025, w, label="2025 Return", color=CB2[0], edgecolor="white", linewidth=0.8)
    ax.bar(x, ret_2026, w, label="2026 Return", color=CB2[1], edgecolor="white", linewidth=0.8)
    ax.bar(x + w, combined, w, label="Combined Return", color="#009E73", edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(capitals, fontsize=10)
    ax.set_ylabel("Return (%)", fontsize=11, fontweight="bold")
    ax.set_title("Capital Scaling Effect on Strategy Returns\n(Conservative, H7/H10, 3 Positions, Regime ON)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.axhline(y=0, color="black", linewidth=0.8)

    for i, (c25, c26, comb) in enumerate(zip(ret_2025, ret_2026, combined)):
        ax.text(i - w, c25 + 0.5, f"{c25:.1f}%", ha="center", fontsize=8, fontweight="bold")
        ax.text(i, c26 + 0.5, f"{c26:.1f}%", ha="center", fontsize=8, fontweight="bold")
        ax.text(i + w, comb + 0.5, f"{comb:.1f}%", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    save(fig, "fig_capital_scaling.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Sub-Universe Performance Comparison
# ═══════════════════════════════════════════════════════════════════════════
def fig_subuniverse_comparison():
    universes = ["Nifty50\n(47 stocks)", "Next50\n(37 stocks)", "ALL 84"]
    ret_2025 = [10.61, 15.65, 15.90]
    ret_2026 = [-3.80, -2.19, -5.59]
    combined = [6.41, 13.12, 9.42]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(universes))
    w = 0.25

    ax.bar(x - w, ret_2025, w, label="2025 Return", color=CB2[0], edgecolor="white", linewidth=0.8)
    ax.bar(x, ret_2026, w, label="2026 Return", color=CB2[1], edgecolor="white", linewidth=0.8)
    ax.bar(x + w, combined, w, label="Combined Return", color="#009E73", edgecolor="white", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(universes, fontsize=9)
    ax.set_ylabel("Return (%)", fontsize=11, fontweight="bold")
    ax.set_title("Sub-Universe Performance Comparison\n(Conservative, H7/H10, 3 Positions, No Regime Filter)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    ax.axhline(y=0, color="black", linewidth=1)

    for i in range(len(universes)):
        for j, (val, offset) in enumerate([(ret_2025[i], -w), (ret_2026[i], 0), (combined[i], +w)]):
            vy = val + (1.5 if val >= 0 else -2.5)
            va = "bottom" if val >= 0 else "top"
            ax.text(i + offset, vy, f"{val:+.1f}%", ha="center", fontsize=7, fontweight="bold",
                    color="#2E7D32" if val > 0 else "#C62828")

    plt.tight_layout()
    save(fig, "fig_subuniverse_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Production Strategy Summary Dashboard — 3-Period Results
# ═══════════════════════════════════════════════════════════════════════════
def fig_production_summary():
    # B6: load from canonical runs for each period; fall back to paper values
    _24 = load_profile_metrics(BT_2024_DIR, "conservative")
    _25 = load_profile_metrics(BT_6PROF_DIR, "conservative")
    _26 = load_profile_metrics(BT_2025EQUITY_DIR, "conservative")
    periods = ["2024\n(In-Sample)", "2025\n(Validation)", "2026\n(Test+OOS)"]
    returns    = [_24.get("total_return_pct", 21.80), _25.get("total_return_pct", 28.77), _26.get("total_return_pct", 2.72)]
    sharpes    = [_24.get("sharpe", 1.58),            _25.get("sharpe", 2.94),            _26.get("sharpe", 0.84)]
    maxdds     = [_24.get("max_dd", 9.3),             _25.get("max_dd", 3.9),             _26.get("max_dd", 3.8)]
    winrates   = [_24.get("win_rate", 67.6),          _25.get("win_rate", 60.2),          _26.get("win_rate", 52.2)]
    trades     = [_24.get("trades", 145),             _25.get("trades", 83),             _26.get("trades", 23)]
    # cumulative capital (in Rs. L, Rs.10L start)
    cap_start = 10.0
    cumul_capital = [cap_start * (1 + r/100) / cap_start for r in returns]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    ((ax1, ax2, ax3), (ax4, ax5, ax6)) = axes
    x = np.arange(len(periods))

    # 1. Returns
    colors_ret = ["#0072B2", "#009E73", "#D55E00"]
    bars = ax1.bar(x, returns, 0.5, color=colors_ret, edgecolor="white", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(periods, fontsize=8)
    ax1.set_title("Return (%)", fontsize=11, fontweight="bold")
    ax1.axhline(y=0, color="black", linewidth=0.8)
    for bar, val in zip(bars, returns):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"+{val:.1f}%",
                ha="center", fontsize=10, fontweight="bold", color="#2E7D32")
    ax1.set_ylim(-5, 38)

    # 2. Sharpe Ratio
    bars2 = ax2.bar(x, sharpes, 0.5, color=colors_ret, edgecolor="white", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(periods, fontsize=8)
    ax2.set_title("Sharpe Ratio", fontsize=11, fontweight="bold")
    ax2.axhline(y=1.0, color="green", linestyle="--", alpha=0.4, label="Good > 1.0")
    ax2.axhline(y=0, color="grey", linestyle="--", alpha=0.4)
    for bar, val in zip(bars2, sharpes):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.08, f"{val:.2f}",
                ha="center", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=7)

    # 3. Max Drawdown
    bars3 = ax3.bar(x, maxdds, 0.5, color=["#C62828", "#E65100", "#E65100"], edgecolor="white", linewidth=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(periods, fontsize=8)
    ax3.set_title("Max Drawdown (%)", fontsize=11, fontweight="bold")
    for bar, val in zip(bars3, maxdds):
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.3, f"{val:.1f}%",
                ha="center", fontsize=10, fontweight="bold", color="#C62828")

    # 4. Win Rate
    bars4 = ax4.bar(x, winrates, 0.5, color=colors_ret, edgecolor="white", linewidth=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels(periods, fontsize=8)
    ax4.set_title("Win Rate (%)", fontsize=11, fontweight="bold")
    ax4.axhline(y=50, color="grey", linestyle="--", alpha=0.4, label="Chance (50%)")
    ax4.legend(fontsize=7)
    for bar, val in zip(bars4, winrates):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}%",
                ha="center", fontsize=10, fontweight="bold")

    # 5. Trade Count
    bars5 = ax5.bar(x, trades, 0.5, color=colors_ret, edgecolor="white", linewidth=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels(periods, fontsize=8)
    ax5.set_title("Trades Executed", fontsize=11, fontweight="bold")
    for bar, val in zip(bars5, trades):
        ax5.text(bar.get_x() + bar.get_width()/2, val + 3, str(val),
                ha="center", fontsize=10, fontweight="bold")

    # 6. Capital Growth
    cap_start = 10.0  # Rs.10L
    cap_end = [cap_start * c for c in cumul_capital]
    bars6 = ax6.bar(x, cap_end, 0.5, color=["#4FC3F7", "#81C784", "#FF8A65"],
                    edgecolor="white", linewidth=0.8)
    ax6.set_xticks(x)
    ax6.set_xticklabels(periods, fontsize=8)
    ax6.set_title("Capital Growth (Rs.L, from Rs.10L)", fontsize=11, fontweight="bold")
    ax6.axhline(y=10, color="grey", linestyle="--", alpha=0.4)
    for bar, val in zip(bars6, cap_end):
        ax6.text(bar.get_x() + bar.get_width()/2, val + 0.1, f"Rs.{val:.2f}L",
                ha="center", fontsize=9, fontweight="bold", color="#1565C0")

    fig.suptitle("Production Strategy: Regime-Filtered Conservative\n(H7/H10, 3 Positions, Rs.10L, ALL 84 Stocks)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "fig_production_summary.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8: DOWN vs UP Directional Accuracy by Market Regime (Enhanced)
# ═══════════════════════════════════════════════════════════════════════════
def fig_down_up_regime_asymmetry():
    horizons = ["H1", "H3", "H5", "H7", "H10"]
    h_pos = np.arange(len(horizons))

    # 2026 Bear Market (high confidence > 0.25)
    up_2026 = [63.7, None, 61.7, 60.1, None]
    down_2026 = [71.1, None, 71.6, 67.8, None]

    # 2025 Bull Market (high confidence > 0.20 for comparison)
    up_2025 = [70.3, None, 65.7, None, None]
    down_2025 = [68.0, None, 61.0, None, None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    w = 0.3

    # 2026 Bear Market
    valid_h_2026 = [0, 2, 3]  # H1, H5, H7
    h_labels_2026 = [horizons[i] for i in valid_h_2026]
    x_2026 = np.arange(len(valid_h_2026))
    up_vals = [up_2026[i] for i in valid_h_2026]
    down_vals = [down_2026[i] for i in valid_h_2026]

    ax1.bar(x_2026 - w/2, up_vals, w, label="UP Accuracy", color=CB2[0], edgecolor="white", linewidth=0.8)
    ax1.bar(x_2026 + w/2, down_vals, w, label="DOWN Accuracy", color="#C62828", edgecolor="white", linewidth=0.8)
    ax1.set_xticks(x_2026)
    ax1.set_xticklabels(h_labels_2026)
    ax1.set_title("2026 Bear Market\n(|P−0.5| > 0.25)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.axhline(y=50, color="grey", linestyle="--", alpha=0.4)

    for i, (up_val, down_val) in enumerate(zip(up_vals, down_vals)):
        ax1.text(i - w/2, up_val + 1, f"{up_val:.1f}%", ha="center", fontsize=8, fontweight="bold", color=CB2[0])
        ax1.text(i + w/2, down_val + 1, f"{down_val:.1f}%", ha="center", fontsize=8, fontweight="bold", color="#C62828")
        spread = down_val - up_val
        ax1.annotate(f"Δ +{spread:.1f}pp", xy=(i, (up_val + down_val)/2),
                    fontsize=8, fontweight="bold", color="#D55E00", ha="center")

    # 2025 Bull Market
    valid_h_2025 = [0, 2]  # H1, H5
    h_labels_2025 = [horizons[i] for i in valid_h_2025]
    x_2025 = np.arange(len(valid_h_2025))
    up_vals5 = [up_2025[i] for i in valid_h_2025]
    down_vals5 = [down_2025[i] for i in valid_h_2025]

    ax2.bar(x_2025 - w/2, up_vals5, w, label="UP Accuracy", color=CB2[0], edgecolor="white", linewidth=0.8)
    ax2.bar(x_2025 + w/2, down_vals5, w, label="DOWN Accuracy", color="#C62828", edgecolor="white", linewidth=0.8)
    ax2.set_xticks(x_2025)
    ax2.set_xticklabels(h_labels_2025)
    ax2.set_title("2025 Bull Market\n(|P−0.5| > 0.20)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.axhline(y=50, color="grey", linestyle="--", alpha=0.4)

    for i, (up_val, down_val) in enumerate(zip(up_vals5, down_vals5)):
        ax2.text(i - w/2, up_val + 1, f"{up_val:.1f}%", ha="center", fontsize=8, fontweight="bold", color=CB2[0])
        ax2.text(i + w/2, down_val + 1, f"{down_val:.1f}%", ha="center", fontsize=8, fontweight="bold", color="#C62828")
        spread = up_val - down_val
        ax2.annotate(f"Δ +{spread:.1f}pp", xy=(i, (up_val + down_val)/2),
                    fontsize=8, fontweight="bold", color="#0072B2", ha="center")

    fig.suptitle("Directional Asymmetry: UP vs DOWN Accuracy by Market Regime",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    save(fig, "fig_down_up_regime_asymmetry.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 9: Magnitude Correlation — Raw vs Smoothed Target
# ═══════════════════════════════════════════════════════════════════════════
def fig_magnitude_correlation():
    horizons = ["H1", "H3", "H5", "H7", "H10"]
    raw_2025 = [0.0504, 0.0672, 0.0241, 0.0142, 0.0684]
    smoothed_2025 = [0.3156, 0.1769, 0.1757, 0.1788, 0.1583]
    raw_2026 = [0.0038, 0.0186, 0.1079, 0.1288, 0.0893]
    smoothed_2026 = [0.2956, 0.1651, 0.2335, 0.2244, 0.1439]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(horizons))
    w = 0.3

    ax1.bar(x - w/2, raw_2025, w, label="Raw Target", color="#D55E00", edgecolor="white", linewidth=0.8)
    ax1.bar(x + w/2, smoothed_2025, w, label="Smoothed Target", color="#0072B2", edgecolor="white", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(horizons)
    ax1.set_title("2025 Magnitude Correlation", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Pearson r", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)

    for i, (r, s) in enumerate(zip(raw_2025, smoothed_2025)):
        ax1.text(i - w/2, r + 0.008, f"{r:.4f}", ha="center", fontsize=7, color="#D55E00")
        ax1.text(i + w/2, s + 0.008, f"{s:.4f}", ha="center", fontsize=7, fontweight="bold", color="#0072B2")

    ax2.bar(x - w/2, raw_2026, w, label="Raw Target", color="#D55E00", edgecolor="white", linewidth=0.8)
    ax2.bar(x + w/2, smoothed_2026, w, label="Smoothed Target", color="#0072B2", edgecolor="white", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(horizons)
    ax2.set_title("2026 Magnitude Correlation", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Pearson r", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)

    for i, (r, s) in enumerate(zip(raw_2026, smoothed_2026)):
        ax2.text(i - w/2, r + 0.008, f"{r:.4f}", ha="center", fontsize=7, color="#D55E00")
        ax2.text(i + w/2, s + 0.008, f"{s:.4f}", ha="center", fontsize=7, fontweight="bold", color="#0072B2")

    fig.suptitle("Magnitude Correlation: Raw Target vs Smoothed Target\n(3–10× improvement with training-aligned evaluation)",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    save(fig, "fig_magnitude_correlation.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 10: Exit Breakdown — Production Strategy (2025)
# ═══════════════════════════════════════════════════════════════════════════
def fig_exit_breakdown():
    reasons = ["Horizon End\n(55%)", "Target Hit\n(22%)", "Stop Loss\n(19%)", "Forced Close\n(4%)"]
    counts = [47, 19, 16, 3]
    colors_pie = ["#90CAF9", "#4CAF50", "#EF5350", "#BDBDBD"]
    explode = (0, 0.05, 0, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Pie chart
    wedges, texts, autotexts = ax1.pie(counts, explode=explode, labels=reasons, colors=colors_pie,
                                        autopct="%1.0f%%", startangle=90, textprops={"fontsize": 9})
    for at in autotexts:
        at.set_fontweight("bold")
    ax1.set_title("Exit Reason Distribution", fontsize=11, fontweight="bold")

    # Summary stats
    ax2.axis("off")
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_title("Trade Summary (2025)", fontsize=11, fontweight="bold")

    # B6: load exit stats from canonical 2025 conservative run
    _m = load_profile_metrics(BT_6PROF_DIR, "conservative")
    _trades   = _m.get("trades",            85)
    _wr       = _m.get("win_rate",          60.2)
    _avg_win  = _m.get("avg_win",           4349)
    _avg_loss = _m.get("avg_loss",         -3218)
    _pf       = _m.get("profit_factor",     2.71)
    _ret      = _m.get("total_return_pct",  28.77)
    _sharpe   = _m.get("sharpe",            2.94)
    _maxdd    = _m.get("max_dd",            3.9)
    stats = [
        f"Total Trades: {_trades}",
        f"Win Rate: {_wr:.1f}%",
        f"Avg Win: +Rs.{abs(_avg_win):,.0f}",
        f"Avg Loss: −Rs.{abs(_avg_loss):,.0f}",
        f"Profit Factor: {_pf:.2f}",
        f"Return: +{_ret:.2f}% (Sharpe {_sharpe:.2f})",
        f"Max Drawdown: {_maxdd:.1f}%",
    ]
    for i, stat in enumerate(stats):
        ax2.text(0.5, 9.0 - i * 1.1, stat, fontsize=10, fontfamily="monospace",
                transform=ax2.transData, fontweight="bold" if i in [0, 5] else "normal")

    fig.suptitle("Production Strategy Exit Analysis — 2025 Validation Period\n(Conservative, H7/H10, Regime ON, Rs.10L)",
                 fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    save(fig, "fig_exit_breakdown.png")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating strategy backtesting figures for paper...")
    print()

    fig_regime_filter_impact()
    fig_position_quality_decay()
    fig_lever_impact_ranking()
    fig_cumulative_equity_strategy()
    fig_capital_scaling()
    fig_subuniverse_comparison()
    fig_production_summary()
    fig_down_up_regime_asymmetry()
    fig_magnitude_correlation()
    fig_exit_breakdown()

    print()
    print(f"✓ 10 figures saved to {OUT_DIR}/")
