"""Shared chart style: one palette and one set of chrome rules for every figure."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e6e5e0"
# Categorical slots, always assigned in this order.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
# Single-hue sequential ramp (light = low, dark = high).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef4fc", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DPI = 200


def apply_style() -> None:
    """Set the global matplotlib defaults used by every figure."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
            "figure.dpi": 100,
        }
    )


def finish(fig: plt.Figure, path: str, title: str, subtitle: str = "") -> None:
    """Add a left-aligned title block above the plot area (sized in inches, so it never collides) and save."""
    height = fig.get_figheight()
    fig.text(0.01, 1 + 0.50 / height, title, ha="left", va="bottom", fontsize=13, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 1 + 0.20 / height, subtitle, ha="left", va="bottom", fontsize=9.5, color=INK_SECONDARY)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
