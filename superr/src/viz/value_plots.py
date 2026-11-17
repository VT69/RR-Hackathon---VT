"""Phase 3 figures: expectation calibration, the wicket exchange rate, and run-value sanity checks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz.style import BLUE, INK, INK_SECONDARY, MUTED, ORANGE, SEQUENTIAL, SURFACE, apply_style, finish

EXPECTATION_LABELS = {
    "xr_bat": "Runs off the bat",
    "xw_bat": "Striker dismissed",
    "xr_bowl": "Runs charged to bowler",
    "xw_bowl": "Bowler-credited wicket",
}


def plot_expectation_calibration(calibration: pd.DataFrame, path: Path) -> None:
    """Predicted v actual by tenth of the prediction range, per expectation (columns) and innings (rows)."""
    apply_style()
    names = list(EXPECTATION_LABELS)
    fig, axes = plt.subplots(2, len(names), figsize=(3.1 * len(names), 6.0))
    for col, name in enumerate(names):
        for row, innings in enumerate((1, 2)):
            ax = axes[row, col]
            d = calibration[(calibration["expectation"] == name) & (calibration["innings"] == innings)]
            lo, hi = min(d["predicted"].min(), d["actual"].min()), max(d["predicted"].max(), d["actual"].max())
            pad = (hi - lo) * 0.08
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            ax.plot(d["predicted"], d["actual"], color=BLUE if name.startswith("xr") else ORANGE, marker="o", markersize=5)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.locator_params(nbins=4)
            if row == 0:
                ax.set_title(EXPECTATION_LABELS[name], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{'First innings' if innings == 1 else 'Chase'}\nactual")
            if row == 1:
                ax.set_xlabel("Expected")
    fig.subplots_adjust(wspace=0.32, hspace=0.2)
    finish(
        fig, str(path), "Expected v actual on held-out seasons",
        "Each point is a tenth of the balls, sorted by expectation. On the dashed line = the expectation matches what happened.",
    )


def plot_cost_grid(grid: pd.DataFrame, path: Path, min_balls: int) -> None:
    """Mean run cost of a wicket by overs left and wickets in hand, one heatmap per innings."""
    apply_style()
    shown = grid[grid["balls"] >= min_balls]
    vmax = float(shown["mean_cost"].quantile(0.98))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, innings, label in ((axes[0], 1, "First innings"), (axes[1], 2, "Chase")):
        d = shown[shown["innings"] == innings].pivot(index="wickets_in_hand", columns="overs_left", values="mean_cost")
        d = d.reindex(index=range(10, 0, -1), columns=range(20, 0, -1))
        image = ax.imshow(d.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=vmax, aspect="auto")
        for (i, j), value in np.ndenumerate(d.to_numpy()):
            if np.isfinite(value):
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7.5,
                        color=SURFACE if value > 0.55 * vmax else INK)
        ax.set_xticks(range(20), labels=[str(v) for v in d.columns])
        ax.set_yticks(range(10), labels=[str(v) for v in d.index])
        ax.set_title(label, fontsize=10.5)
        ax.set_xlabel("Overs left (including the current one)")
        ax.grid(visible=False)
    axes[0].set_ylabel("Wickets in hand")
    fig.subplots_adjust(wspace=0.05)
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.015)
    bar.set_label("Runs", color=INK_SECONDARY)
    bar.outline.set_visible(False)
    finish(
        fig, str(path), "What a wicket costs, in runs",
        f"Average over the balls actually bowled in each state. Cells with fewer than {min_balls} balls are blank.",
    )


def plot_zero_mean(table: pd.DataFrame, path: Path) -> None:
    """Mean run value per 100 balls with a 95% interval, by slice, for batters and bowlers."""
    apply_style()
    keep = table[table["slice"].isin(["overall", "innings x phase", "pressure flag", "competition"])]
    labels = (keep["slice"] + ": " + keep["level"]).where(keep["slice"] != "overall", "All balls")
    keep = keep.assign(label=labels.str.replace("innings x phase: 1", "1st innings,").str.replace("innings x phase: 2", "Chase,")
                       .str.replace("pressure flag: ", "").str.replace("competition: ", ""))
    keep["label"] = keep["label"].str[:1].str.upper() + keep["label"].str[1:]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, role, colour, title in ((axes[0], "batter", BLUE, "Batting"), (axes[1], "bowler", ORANGE, "Bowling")):
        d = keep[keep["role"] == role].iloc[::-1]
        y = np.arange(len(d))
        ax.axvline(0, color=INK, linewidth=0.9)
        ax.errorbar(d["value_per_100"], y, xerr=1.96 * d["se"], fmt="o", color=colour, markersize=6, elinewidth=1.6,
                    markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.set_yticks(y, labels=d["label"])
        ax.set_title(title, fontsize=10.5)
        ax.set_xlabel("Mean run value per 100 balls (95% interval)")
        ax.grid(axis="y", visible=False)
    fig.subplots_adjust(wspace=0.06)
    finish(
        fig, str(path), "Run value averages zero wherever you cut the data",
        "Held-out seasons. An interval that covers zero means that kind of ball is neither favoured nor penalised.",
    )


def plot_face_validity(batters: pd.DataFrame, bowlers: pd.DataFrame, path: Path, min_balls: int) -> None:
    """Overall run value per 100 balls against the traditional headline number, one point per player."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    panels = ((axes[0], batters, "strike_rate", BLUE, "Batters: strike rate"), (axes[1], bowlers, "economy", ORANGE, "Bowlers: economy rate"))
    for ax, players, column, colour, label in panels:
        d = players[players["balls"] >= min_balls]
        ax.scatter(d[column], d["value_per_100"], s=34, color=colour, edgecolor=SURFACE, linewidth=0.8, alpha=0.85)
        ax.axhline(0, color=INK, linewidth=0.9)
        r = d[[column, "value_per_100"]].corr().iloc[0, 1]
        corner = (0.03, 0.95, "top") if r > 0 else (0.03, 0.05, "bottom")   # keep the note clear of the points
        ax.text(corner[0], corner[1], f"{len(d)} players, correlation {r:+.2f}", transform=ax.transAxes, ha="left",
                va=corner[2], fontsize=9, color=INK_SECONDARY)
        ax.set_xlabel(label.split(": ")[1].capitalize())
        ax.set_title(label.split(": ")[0], fontsize=10.5)
    axes[0].set_ylabel("Run value per 100 balls")
    fig.subplots_adjust(wspace=0.16)
    finish(
        fig, str(path), "Run value agrees with the traditional numbers, without copying them",
        f"One point per player with at least {min_balls} balls. Above the line = better than a league-average player in the same situations.",
    )
