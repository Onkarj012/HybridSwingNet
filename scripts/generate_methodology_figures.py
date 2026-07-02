#!/usr/bin/env python3
"""Generate 5 clean methodology figures for the StockXpert paper."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import numpy as np

DPI = 300
OUT = "docs/figures"
COLORS = {
    'short': '#2563EB',   # blue
    'mid': '#16A34A',     # green
    'long': '#EA580C',    # orange
    'context': '#9333EA', # purple
    'sentiment': '#DB2777',# pink
    'fusion': '#0891B2',   # cyan
    'bg': '#F8FAFC',
    'highlight': '#FEF3C7',
    'text': '#1E293B',
    'arrow': '#64748B',
    'box_border': '#CBD5E1',
}

def draw_box(ax, x, y, w, h, text, color, fontsize=8, bold=False):
    """Draw a rounded box with text."""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor=color, alpha=0.85, linewidth=0)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color='white')

def draw_arrow(ax, x1, y1, x2, y2, color=COLORS['arrow'], lw=1.2):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, connectionstyle='arc3,rad=0'))

def draw_label(ax, x, y, text, fontsize=7, color=COLORS['text'], ha='center', bold=False):
    ax.text(x, y, text, ha=ha, va='center', fontsize=fontsize, color=color, fontweight='bold' if bold else 'normal')

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Overall Architecture Framework
# ═══════════════════════════════════════════════════════════════
def fig_overall_framework():
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16); ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])
    fig.patch.set_facecolor(COLORS['bg'])

    # Title
    ax.text(8, 9.7, 'StockXpert: Multi-Encoder Swing Trading Framework', ha='center', fontsize=14, fontweight='bold', color=COLORS['text'])

    # ── Stage 1: Inputs ──
    draw_label(ax, 0.8, 9.1, 'INPUT', fontsize=9, bold=True)
    inputs = ['OHLCV\nData', 'Fundamental\nData', 'Sentiment\nData', 'Market\nContext']
    for i, label in enumerate(inputs):
        draw_box(ax, 1 + i*1.8, 8.2, 1.5, 1.0, label, COLORS['short'] if i==0 else COLORS['mid'] if i==1 else COLORS['sentiment'] if i==2 else COLORS['context'], fontsize=7)

    # ── Stage 2: Preprocessing ──
    draw_label(ax, 0.8, 7.5, 'PREPROCESS', fontsize=9, bold=True)
    preproc = ['Normalization', 'Technical\nIndicators', 'Feature\nEngineering', 'Window\nConstruction']
    for i, label in enumerate(preproc):
        draw_box(ax, 1 + i*1.8, 6.6, 1.5, 0.8, label, '#94A3B8', fontsize=6.5)

    # ── Stage 3: Encoders (CORE) ──
    draw_label(ax, 0.8, 5.9, 'ENCODERS', fontsize=9, bold=True)
    encoders = [('ResNLS\nLocal Patterns', COLORS['short']),
                ('BiGRU\nMomentum', COLORS['mid']),
                ('SCSO-LSTM\nLong Trends', COLORS['long']),
                ('Context MLP\nRegime', COLORS['context']),
                ('Sentiment\nPsychology', COLORS['sentiment'])]
    for i, (label, col) in enumerate(encoders):
        draw_box(ax, 0.8 + i*1.7, 4.8, 1.5, 1.0, label, col, fontsize=7)

    # ── Stage 4: NOVELTY HIGHLIGHT ──
    highlight_box = FancyBboxPatch((0.5, 3.2), 8.8, 1.4, boxstyle="round,pad=0.2",
                                    facecolor='#FEF3C7', edgecolor='#F59E0B', linewidth=2, alpha=0.6)
    ax.add_patch(highlight_box)
    draw_label(ax, 4.9, 4.3, 'CORE NOVELTY', fontsize=9, bold=True)
    draw_box(ax, 0.8, 3.4, 4.0, 0.7, 'Graduated Feature Routing\nHorizon-Specific Encoder Blending', '#F59E0B', fontsize=7.5)
    draw_arrow(ax, 5.0, 3.75, 5.2, 3.75, '#F59E0B', lw=1.5)
    draw_box(ax, 5.3, 3.4, 3.8, 0.7, 'Cross-Attention\nDynamic Fusion', '#D97706', fontsize=7.5)

    # ── Stage 5: Prediction ──
    draw_label(ax, 0.8, 2.7, 'PREDICTION', fontsize=9, bold=True)
    draw_box(ax, 0.8, 1.8, 8.8, 0.7, 'Multi-Horizon Heads: H1  |  H3  |  H5  |  H7  |  H10', COLORS['fusion'], fontsize=8)

    # ── Stage 6: Decision ──
    draw_label(ax, 0.8, 1.1, 'DECISION', fontsize=9, bold=True)
    for i, (label, col) in enumerate([('Confidence\nFiltering', '#6366F1'), ('Buy / Sell\n/ Hold', '#8B5CF6'), ('Regime\nFilter', '#A855F7'), ('Backtesting\nEngine', '#C084FC')]):
        draw_box(ax, 1 + i*2.2, 0.2, 1.8, 0.7, label, col, fontsize=7)

    # ── Side panel: Data Flow ──
    draw_label(ax, 11.5, 9.1, 'DATA FLOW', fontsize=9, bold=True)
    flow_items = [
        '2013-2026 Data', '84 Nifty100 Stocks', '47 Nifty50 + 37 Next50',
        '', '5 Horizons (1/3/5/7/10d)', '5 Specialized Encoders',
        '25 Technical Features', '10 Context Features',
        '', 'Training: 2013-2025.06', 'Validation: 2025.07-12',
        'Test: 2026.01-02', 'OOS: 2026.03-04',
        '', 'Multi-Objective Loss', '7 Components',
        'Curriculum + Mixup',
        '', 'Calibrated Confidence', 'ECE = 0.023',
    ]
    for i, item in enumerate(flow_items):
        if item:
            ax.text(11.5, 8.5 - i*0.28, item, fontsize=6.5, color=COLORS['text'], fontweight='bold' if ':' in item else 'normal')
    # side panel border
    side = FancyBboxPatch((10.8, 0.1), 4.8, 9.3, boxstyle="round,pad=0.1",
                           facecolor='none', edgecolor=COLORS['box_border'], linewidth=1)
    ax.add_patch(side)

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig_overall_framework.png', dpi=DPI, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("  ✓ fig_overall_framework.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Graduated Feature Routing (Heatmap/Stacked Bar)
# ═══════════════════════════════════════════════════════════════
def fig_graduated_routing():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor(COLORS['bg'])
    fig.patch.set_facecolor(COLORS['bg'])

    weights = {
        'H1 (1-Day)':  [0.70, 0.30, 0.00],
        'H3 (3-Day)':  [0.30, 0.70, 0.00],
        'H5 (5-Day)':  [0.00, 0.50, 0.50],
        'H7 (7-Day)':  [0.00, 0.30, 0.70],
        'H10 (10-Day)': [0.00, 0.00, 1.00],
    }
    horizons = list(weights.keys())
    cols = [COLORS['short'], COLORS['mid'], COLORS['long']]
    enc_names = ['Short Encoder\n(Local Patterns)', 'Mid Encoder\n(Momentum)', 'Long Encoder\n(Trend)']

    x = np.arange(len(horizons))
    width = 0.55
    bottom = np.zeros(len(horizons))

    for i, (col, name) in enumerate(zip(cols, enc_names)):
        vals = [weights[h][i] for h in horizons]
        bars = ax.barh(x, vals, width, left=bottom, color=col, alpha=0.9, edgecolor='white', linewidth=0.5, label=name)
        for j, (bar, val) in enumerate(zip(bars, vals)):
            if val > 0.15:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                        f'{val:.0%}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        bottom += vals

    ax.set_yticks(x)
    ax.set_yticklabels(horizons, fontsize=11)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Encoder Weight', fontsize=10)
    ax.legend(loc='lower right', fontsize=8, ncol=3, framealpha=0.9)
    ax.set_title('Graduated Feature Routing — Horizon-Specific Encoder Blending', fontsize=13, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=9)
    plt.tight_layout()
    fig.savefig(f'{OUT}/fig_graduated_routing.png', dpi=DPI, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("  ✓ fig_graduated_routing.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Dynamic Fusion / Cross-Attention
# ═══════════════════════════════════════════════════════════════
def fig_dynamic_fusion():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_title('Cross-Attention Dynamic Fusion Mechanism', fontsize=13, fontweight='bold', pad=10)

    # Encoder outputs
    encoders = [
        ('Short\nEncoder', 1.5, COLORS['short']),
        ('Mid\nEncoder', 3.5, COLORS['mid']),
        ('Long\nEncoder', 5.5, COLORS['long']),
        ('Context\nEncoder', 7.5, COLORS['context']),
        ('Sentiment\nEncoder', 9.0, COLORS['sentiment']),
    ]
    for label, x, col in encoders:
        draw_box(ax, x-0.55, 4.8, 1.1, 0.7, label, col, fontsize=7.5)

    # Arrow to attention generator
    for _, x, _ in encoders:
        draw_arrow(ax, x, 4.8, x, 3.8, COLORS['arrow'], lw=1.0)

    # Attention weight generator
    draw_box(ax, 2.5, 3.2, 5.0, 0.6, 'Learned Attention Weights: w = Softmax(W·[e_short; e_mid; e_long; e_context; e_sentiment])  |  Σw = 1',
             COLORS['fusion'], fontsize=7)

    # Weight arrows
    for _, x, _ in encoders:
        draw_arrow(ax, x, 3.8, 4.0, 3.8, COLORS['arrow'], lw=0.5)

    draw_arrow(ax, 5.0, 3.2, 5.0, 2.2, COLORS['arrow'], lw=1.5)

    # Weighted fusion
    draw_box(ax, 3.0, 1.4, 4.0, 0.7, 'Weighted Fusion\nf = w₁·e_short + w₂·e_mid + w₃·e_long', COLORS['fusion'], fontsize=7.5)

    # Per-horizon adaptations
    draw_label(ax, 9.5, 5.5, 'Per-Horizon\nAdaptation →', fontsize=7, ha='right')
    for i, (horizon, enc) in enumerate([('H1: 70% Short', COLORS['short']), ('H3: 70% Mid', COLORS['mid']),
                                         ('H5: 50/50 Mid/Long', '#65A30D'), ('H7: 70% Long', COLORS['long']),
                                         ('H10: 100% Long', COLORS['long'])]):
        ax.text(9.5, 5.1 - i*0.3, horizon, fontsize=6.5, color=enc, ha='right', fontweight='bold')

    # Side note
    note = FancyBboxPatch((0.3, 0.2), 9.4, 0.8, boxstyle="round,pad=0.1",
                           facecolor=COLORS['highlight'], edgecolor='#F59E0B', linewidth=1, alpha=0.5)
    ax.add_patch(note)
    ax.text(5, 0.6, 'The same mechanism produces different weights per horizon — validated by attention heatmap analysis.',
            ha='center', fontsize=7.5, color=COLORS['text'], fontstyle='italic')

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig_dynamic_fusion.png', dpi=DPI, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("  ✓ fig_dynamic_fusion.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Multi-Horizon Prediction Heads
# ═══════════════════════════════════════════════════════════════
def fig_multihorizon():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5.5)
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_title('Multi-Horizon Forecasting Architecture', fontsize=13, fontweight='bold', pad=10)

    # Unified representation
    draw_box(ax, 3.5, 4.3, 3.0, 0.6, 'Unified Representation z ∈ R³⁵⁷', COLORS['fusion'], fontsize=8)

    # Branch to 5 heads
    horizon_colors = [COLORS['short'], '#3B82F6', COLORS['mid'], COLORS['long'], '#DC2626']
    for i, (h, col) in enumerate(zip([1, 3, 5, 7, 10], horizon_colors)):
        x = 0.8 + i * 2.05
        draw_arrow(ax, 5.0, 4.3, x + 0.55, 3.1, COLORS['arrow'])
        draw_box(ax, x, 2.5, 1.1, 0.5, f'H{h}\nHead', col, fontsize=8)
        draw_arrow(ax, x + 0.55, 2.5, x + 0.55, 1.7, COLORS['arrow'])
        interval = ['1 day', '3 days', '5 days', '7 days', '10 days'][i]
        ax.text(x + 0.55, 1.4, f'r{h}', ha='center', fontsize=10, fontweight='bold', color=col)
        ax.text(x + 0.55, 1.1, interval, ha='center', fontsize=7, color=COLORS['arrow'])

    # Consistency label
    note = FancyBboxPatch((1.5, 0.2), 7.0, 0.5, boxstyle="round,pad=0.1",
                           facecolor=COLORS['highlight'], edgecolor='#F59E0B', linewidth=1, alpha=0.5)
    ax.add_patch(note)
    ax.text(5, 0.45, 'Inter-Horizon Consistency Constraints: Direction + Cumulative Return Coherence',
            ha='center', fontsize=7.5, color=COLORS['text'], fontstyle='italic')

    # Feature annotation
    ax.text(0.3, 4.0, 'Features:', fontsize=7, color=COLORS['arrow'], fontweight='bold')
    features = ['Short (7 features × 7d)', 'Mid (12 features × 21d)', 'Long (8 features × 60d)',
                'Context (19 features)', 'Sentiment (9 features)', 'Stock Embedding (48d)']
    for i, f in enumerate(features):
        ax.text(0.3, 3.7 - i*0.3, f'  • {f}', fontsize=6.5, color=COLORS['text'])

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig_multihorizon.png', dpi=DPI, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("  ✓ fig_multihorizon.png")

# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Trading Signal Generation Workflow
# ═══════════════════════════════════════════════════════════════
def fig_trading_workflow():
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set_xlim(0, 8); ax.set_ylim(0, 9)
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])
    fig.patch.set_facecolor(COLORS['bg'])
    ax.set_title('Confidence-Based Trading Signal Execution Pipeline', fontsize=13, fontweight='bold', pad=10)

    steps = [
        (3.0, 8.0, 2.0, 0.5, 'Predicted Returns\nr₁...r₁₀', COLORS['fusion']),
        (3.0, 7.0, 2.0, 0.5, 'Trend\nClassification', '#6366F1'),
        (3.0, 6.0, 2.0, 0.5, 'Confidence\nEstimation', '#8B5CF6'),
    ]
    for i, (x, y, w, h, text, col) in enumerate(steps):
        draw_box(ax, x, y, w, h, text, col, fontsize=8)
        if i > 0:
            draw_arrow(ax, 4.0, steps[i-1][1], 4.0, y + h + 0.15, COLORS['arrow'])

    # Threshold branch
    draw_arrow(ax, 4.0, 6.0, 4.0, 5.2, COLORS['arrow'])
    draw_box(ax, 2.0, 4.5, 4.0, 0.6, 'P(Up) > 0.70 | P(Up) < 0.30\nConfidence > 0.60', '#A855F7', fontsize=8)

    # Three branches
    for i, (label, col, x) in enumerate([('BUY', '#16A34A', 0.5), ('HOLD', '#94A3B8', 3.0), ('SELL', '#DC2626', 5.5)]):
        draw_arrow(ax, 4.0, 4.5, x + 0.8, 3.5, col)
        draw_box(ax, x, 2.8, 1.6, 0.6, label, col, fontsize=9, bold=True)

    # Risk management
    draw_arrow(ax, 1.3, 2.8, 1.3, 2.1, COLORS['arrow'])
    draw_arrow(ax, 3.8, 2.8, 3.8, 2.1, COLORS['arrow'])
    draw_arrow(ax, 6.3, 2.8, 6.3, 2.1, COLORS['arrow'])

    draw_box(ax, 1.5, 1.4, 5.0, 0.6, 'Risk Management: Position Sizing, Stop-Loss, Take-Profit', '#0891B2', fontsize=7.5)
    draw_arrow(ax, 4.0, 1.4, 4.0, 0.8, COLORS['arrow'])
    draw_box(ax, 2.0, 0.15, 4.0, 0.5, 'Trade Execution → Backtesting Engine', '#0F766E', fontsize=8, bold=True)

    # Annotations
    ax.text(4.0, 5.8, 'Regime Filter: Skip entries when\n21d median return < 0', ha='center', fontsize=6.5,
            color=COLORS['arrow'], fontstyle='italic')
    ax.text(4.0, 3.2, '3 positions max | Equal-weight | H7/H10 only', ha='center', fontsize=6.5,
            color=COLORS['arrow'], fontstyle='italic')

    plt.tight_layout()
    fig.savefig(f'{OUT}/fig_trading_workflow.png', dpi=DPI, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("  ✓ fig_trading_workflow.png")

if __name__ == '__main__':
    import os
    os.makedirs(OUT, exist_ok=True)
    print("Generating paper methodology figures...\n")
    fig_overall_framework()
    fig_graduated_routing()
    fig_dynamic_fusion()
    fig_multihorizon()
    fig_trading_workflow()
    print(f"\nDone → {OUT}/")
