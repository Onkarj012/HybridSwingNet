#!/usr/bin/env python3
"""
Generate System Architecture Diagram for IJATEE Paper
Figure 1: Proposed System Architecture
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set up the figure with clean academic style
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")

# Color scheme - professional academic colors
colors = {
    "input": "#E8F4FD",  # Light blue
    "encoder": "#FFF4E6",  # Light orange
    "fusion": "#E8F5E9",  # Light green
    "output": "#F3E5F5",  # Light purple
    "arrow": "#424242",  # Dark gray
    "text": "#212121",  # Near black
    "border": "#757575",  # Gray
}


def draw_box(
    ax, x, y, width, height, text, color, fontsize=9, bold=False, border_color=None
):
    """Draw a rounded rectangle with text"""
    if border_color is None:
        border_color = colors["border"]

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color,
        edgecolor=border_color,
        linewidth=1.5,
    )
    ax.add_patch(box)

    weight = "bold" if bold else "normal"
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=colors["text"],
        weight=weight,
        wrap=True,
    )

    return box


def draw_arrow(ax, start, end, color=None, style="->"):
    """Draw an arrow between two points"""
    if color is None:
        color = colors["arrow"]

    arrow = FancyArrowPatch(
        start, end, arrowstyle=style, color=color, linewidth=1.5, mutation_scale=15
    )
    ax.add_patch(arrow)


# Title
ax.text(
    7,
    9.7,
    "Proposed System Architecture",
    ha="center",
    va="center",
    fontsize=16,
    weight="bold",
    color=colors["text"],
)

# ===== INPUT LAYER (Top) =====
input_y = 8.5
input_width = 2.2
input_height = 0.8
input_spacing = 0.3

# Input boxes
inputs = [
    ("OHLCV\nPrice Data", "Technical\nIndicators"),
    ("Fundamental\nData", "Financial\nRatios"),
    ("Market\nContext", "VIX, Breadth"),
    ("Sentiment\nAnalysis", "News, Social"),
]

input_x_positions = [1.2, 4.2, 7.2, 10.2]
for i, (main, sub) in enumerate(inputs):
    x = input_x_positions[i]
    # Main input box
    draw_box(
        ax, x, input_y, input_width, input_height, main, colors["input"], bold=True
    )
    # Sub-label
    ax.text(
        x + input_width / 2,
        input_y - 0.3,
        sub,
        ha="center",
        va="top",
        fontsize=8,
        style="italic",
        color=colors["text"],
    )

# Input layer label
ax.text(
    0.3,
    input_y + 0.4,
    "INPUTS",
    ha="left",
    va="center",
    fontsize=10,
    weight="bold",
    color=colors["text"],
)

# ===== ENCODER LAYER (Middle-Top) =====
encoder_y = 6.0
encoder_width = 2.2
encoder_height = 1.2

encoders = [
    ("ResNLS\nEncoder", "Local Pattern\nExtraction"),
    ("BiGRU\nEncoder", "Short-term\nMomentum"),
    ("BiLSTM\nEncoder", "Long-term\nTrends"),
    ("Context\nEncoder", "Market\nConditions"),
    ("Sentiment\nEncoder", "News\nAnalysis"),
]

encoder_x_positions = [0.5, 2.8, 5.1, 7.4, 9.7]
for i, (name, desc) in enumerate(encoders):
    x = encoder_x_positions[i]
    # Encoder box
    draw_box(
        ax,
        x,
        encoder_y,
        encoder_width,
        encoder_height,
        name,
        colors["encoder"],
        bold=True,
    )
    # Description
    ax.text(
        x + encoder_width / 2,
        encoder_y - 0.35,
        desc,
        ha="center",
        va="top",
        fontsize=7,
        style="italic",
        color=colors["text"],
    )

# Encoder layer label
ax.text(
    0.3,
    encoder_y + 0.7,
    "ENCODERS",
    ha="left",
    va="center",
    fontsize=10,
    weight="bold",
    color=colors["text"],
)

# ===== FUSION LAYER (Middle) =====
fusion_y = 4.0
fusion_width = 5.0
fusion_height = 1.0

# Dynamic Fusion Module
fusion_x = 4.5
draw_box(
    ax,
    fusion_x,
    fusion_y,
    fusion_width,
    fusion_height,
    "Dynamic Fusion Module\n(Multi-Head Attention)",
    colors["fusion"],
    fontsize=10,
    bold=True,
)

# Fusion description
ax.text(
    7,
    fusion_y - 0.4,
    "Adaptive Feature Weighting based on Market Regime",
    ha="center",
    va="top",
    fontsize=8,
    style="italic",
    color=colors["text"],
)

# Fusion label
ax.text(
    0.3,
    fusion_y + 0.6,
    "FUSION",
    ha="left",
    va="center",
    fontsize=10,
    weight="bold",
    color=colors["text"],
)

# ===== PREDICTION HEADS (Middle-Bottom) =====
heads_y = 2.0
head_width = 2.0
head_height = 0.8

heads = [
    "Direction\n(Up/Down)",
    "Magnitude\n(Z-score)",
    "Confidence\n(0-1)",
    "Target Zone\n(7 classes)",
    "Price Levels\n(S/R/T)",
]

head_x_positions = [0.7, 2.9, 5.1, 7.3, 9.5]
for i, head in enumerate(heads):
    x = head_x_positions[i]
    draw_box(
        ax,
        x,
        heads_y,
        head_width,
        head_height,
        head,
        colors["output"],
        fontsize=8,
        bold=True,
    )

# Prediction heads label
ax.text(
    0.3,
    heads_y + 0.5,
    "PREDICTION\nHEADS",
    ha="left",
    va="center",
    fontsize=9,
    weight="bold",
    color=colors["text"],
)

# ===== OUTPUT LAYER (Bottom) =====
output_y = 0.5
output_width = 3.0
output_height = 0.9

# Main outputs
draw_box(
    ax,
    1.5,
    output_y,
    output_width,
    output_height,
    "Buy / Sell / Hold\nSignal",
    "#FFEBEE",
    fontsize=10,
    bold=True,
    border_color="#C62828",
)

draw_box(
    ax,
    5.5,
    output_y,
    output_width,
    output_height,
    "Risk Score\n(Conservative / Balanced / Aggressive)",
    "#E3F2FD",
    fontsize=10,
    bold=True,
    border_color="#1565C0",
)

draw_box(
    ax,
    9.5,
    output_y,
    output_width,
    output_height,
    "Confidence &\nExplainability",
    "#E8F5E9",
    fontsize=10,
    bold=True,
    border_color="#2E7D32",
)

# Output label
ax.text(
    0.3,
    output_y + 0.5,
    "OUTPUTS",
    ha="left",
    va="center",
    fontsize=10,
    weight="bold",
    color=colors["text"],
)

# ===== ARROWS =====
# Input to Encoders (with specific mappings)
# OHLCV -> ResNLS, BiGRU, BiLSTM
for i in range(3):
    draw_arrow(
        ax,
        (input_x_positions[0] + input_width / 2, input_y),
        (encoder_x_positions[i] + encoder_width / 2, encoder_y + encoder_height),
    )

# Fundamental -> Context
for i in [2, 3]:
    draw_arrow(
        ax,
        (input_x_positions[1] + input_width / 2, input_y),
        (encoder_x_positions[i] + encoder_width / 2, encoder_y + encoder_height),
    )

# Market Context -> Context
for i in [2, 3]:
    draw_arrow(
        ax,
        (input_x_positions[2] + input_width / 2, input_y),
        (encoder_x_positions[i] + encoder_width / 2, encoder_y + encoder_height),
    )

# Sentiment -> Sentiment
for i in [4]:
    draw_arrow(
        ax,
        (input_x_positions[3] + input_width / 2, input_y),
        (encoder_x_positions[i] + encoder_width / 2, encoder_y + encoder_height),
    )

# Encoders to Fusion
for x in encoder_x_positions:
    draw_arrow(ax, (x + encoder_width / 2, encoder_y), (7, fusion_y + fusion_height))

# Fusion to Prediction Heads
draw_arrow(ax, (7, fusion_y), (7, heads_y + head_height))

# Prediction Heads to Outputs
# Direction -> Signal
# draw_arrow(ax, (head_x_positions[0] + head_width/2, heads_y),
#            (3, output_y + output_height))
#
# # Confidence -> Risk
# draw_arrow(ax, (head_x_positions[2] + head_width/2, heads_y),
#            (7, output_y + output_height))

# ===== LEGEND/SIDE INFO =====
legend_x = 12.5
ax.text(
    legend_x,
    7.5,
    "Key Features:",
    ha="left",
    va="top",
    fontsize=9,
    weight="bold",
    color=colors["text"],
)

features = [
    "• 5 Encoders",
    "• Multi-Head\n  Attention",
    "• Multi-Horizon\n  (H1-H10)",
    "• Confidence\n  Calibration",
    "• 97 Stocks\n  (Nifty 100)",
    "• 80.07% H1\n  Accuracy",
]

y_pos = 7.0
for feat in features:
    ax.text(
        legend_x, y_pos, feat, ha="left", va="top", fontsize=8, color=colors["text"]
    )
    y_pos -= 0.8

# Add dotted box around entire system
system_box = FancyBboxPatch(
    (0.1, 0.2),
    13.8,
    9.3,
    boxstyle="round,pad=0.02,rounding_size=0.2",
    facecolor="none",
    edgecolor=colors["border"],
    linewidth=2,
    linestyle="--",
    alpha=0.5,
)
ax.add_patch(system_box)

# Add "Swing Trading System" label at bottom
ax.text(
    7,
    0.1,
    "Hybrid Multi-Encoder Deep Learning Framework for Explainable Swing Trading",
    ha="center",
    va="bottom",
    fontsize=9,
    style="italic",
    color=colors["text"],
)

plt.tight_layout()
plt.savefig(
    "docs/ACCENTS/figures/fig0_system_architecture.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.savefig(
    "docs/ACCENTS/figures/fig0_system_architecture.pdf",
    bbox_inches="tight",
    facecolor="white",
)
print("✓ System Architecture Diagram created: fig0_system_architecture.png")
print("✓ PDF version also saved: fig0_system_architecture.pdf")
