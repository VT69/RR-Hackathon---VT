"""Phase 2b figures: model evidence and proof that leverage separates pressure from ordinary play."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz.style import (
    AQUA, BLUE, CATEGORICAL, GRID, INK, INK_SECONDARY, MUTED, ORANGE, SEQUENTIAL, SURFACE, apply_style, finish,
)

MODEL_LABELS = {
    "logistic": "Logistic regression",
    "xgboost": "XGBoost (monotone)",
    "lightgbm": "LightGBM (monotone)",
    "dp": "Dynamic programming",
    "dp_platt": "Dynamic programming + recalibration",
}
PHASE_ORDER = ["powerplay", "middle", "death"]


def plot_calibration(reliability: pd.DataFrame, path: Path) -> None:
    """Predicted v observed win rate on held-out seasons, one panel per model and innings."""
    apply_style()
    models = [m for m in MODEL_LABELS if m in set(reliability["model"])]
    fig, axes = plt.subplots(2, len(models), figsize=(2.6 * len(models), 5.2), sharex=True, sharey=True)
    for col, model in enumerate(models):
        for row, innings in enumerate((1, 2)):
            ax = axes[row, col]
            d = reliability[(reliability["model"] == model) & (reliability["innings"] == innings)]
            ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            ax.plot(d["predicted"], d["observed"], color=CATEGORICAL[col], marker="o", markersize=5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1], labels=["0", "0.5", "1"])
            ax.set_yticks([0, 0.5, 1], labels=["0", "0.5", "1"])
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(MODEL_LABELS[model], fontsize=9.5)
            if col == 0:
                ax.set_ylabel(f"{'First innings' if innings == 1 else 'Chase'}\nobserved win rate")
            if row == 1:
                ax.set_xlabel("Predicted win probability")
    fig.subplots_adjust(wspace=0.18, hspace=0.12)
    finish(
        fig, str(path), "Calibration on held-out seasons",
        "Each point is a tenth of the probability range. On the dashed line = predicted chances match what happened.",
    )


def plot_leverage_slice(slices: pd.DataFrame, path: Path, caption: str) -> None:
    """Leverage along one line through the state space, one panel per model (smoothness check)."""
    apply_style()
    models = [m for m in MODEL_LABELS if m in set(slices["model"])]
    fig, axes = plt.subplots(1, len(models), figsize=(2.75 * len(models), 3.3), sharey=True)
    for col, model in enumerate(models):
        d = slices[slices["model"] == model]
        axes[col].plot(d["runs_required"], d["li"], color=CATEGORICAL[col], linewidth=1.6)
        axes[col].set_title(MODEL_LABELS[model], fontsize=9.5)
        axes[col].set_xlabel("Runs required")
    axes[0].set_ylabel("Leverage Index")
    fig.subplots_adjust(wspace=0.08)
    finish(fig, str(path), "How smooth is leverage between neighbouring states?", caption)


def plot_li_distribution(balls: pd.DataFrame, cut: float, path: Path) -> None:
    """Distribution of the Leverage Index, overall and by innings."""
    apply_style()
    scored = balls[balls["state_ok"]]
    bins = np.linspace(0, 6, 61)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    axes[0].hist(scored["li"].clip(upper=6), bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    axes[0].axvline(cut, color=INK, linewidth=1.2)
    axes[0].annotate(
        f"top 25% starts at LI {cut:.2f}", xy=(cut, 1), xycoords=("data", "axes fraction"),
        xytext=(6, -12), textcoords="offset points", fontsize=9, color=INK,
    )
    axes[0].set_title("All scored balls")
    axes[0].set_xlabel("Leverage Index (values above 6 shown at 6)")
    axes[0].set_ylabel("Balls")
    for innings, colour, label in ((1, BLUE, "First innings"), (2, ORANGE, "Chase")):
        d = scored.loc[scored["innings"] == innings, "li"]
        axes[1].hist(d.clip(upper=6), bins=bins, histtype="step", color=colour, linewidth=1.8,
                     label=f"{label} (mean {d.mean():.2f})")
    axes[1].axvline(cut, color=INK, linewidth=1.2)
    axes[1].set_title("By innings")
    axes[1].set_xlabel("Leverage Index")
    axes[1].legend(loc="upper right")
    fig.subplots_adjust(wspace=0.06)
    finish(
        fig, str(path), "Most balls matter a little; a minority matter a lot",
        "Leverage Index = expected win-probability swing on the ball, relative to the average ball (1.0).",
    )


def plot_li_heatmap(grid: pd.DataFrame, path: Path, min_balls: int) -> None:
    """Mean LI by over and wickets fallen, one panel per innings, shared colour scale."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    shown = grid[grid["balls"] >= min_balls]
    vmax = float(np.ceil(shown["mean_li"].max() * 2) / 2)
    image = None
    for ax, innings, label in zip(axes, (1, 2), ("First innings", "Chase")):
        d = shown[shown["innings"] == innings]
        mat = d.pivot(index="wickets_before", columns="overid", values="mean_li").reindex(
            index=range(0, 10), columns=range(1, 21)
        )
        image = ax.imshow(mat.to_numpy(), origin="lower", aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=vmax)
        for (w, over), value in np.ndenumerate(mat.to_numpy()):
            if not np.isnan(value):
                ax.text(over, w, f"{value:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white" if value > 0.55 * vmax else INK)
        ax.set_xticks(range(0, 20))
        ax.set_xticklabels(range(1, 21), fontsize=8)
        ax.set_yticks(range(0, 10))
        ax.set_title(label)
        ax.set_xlabel("Over")
        ax.grid(False)
        ax.set_facecolor("#f1f0ec")
    axes[0].set_ylabel("Wickets already down")
    fig.subplots_adjust(wspace=0.04)
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.015)
    bar.set_label("Mean Leverage Index")
    bar.outline.set_visible(False)
    finish(
        fig, str(path), "Where pressure lives: average leverage by over and wickets down",
        f"Cells with fewer than {min_balls} balls are left blank. 1.0 = an average ball.",
    )


def plot_separation(summary: pd.DataFrame, path: Path, definition: str) -> None:
    """High-pressure v ordinary balls on five outcome measures, overall and within each phase."""
    apply_style()
    metrics = [
        ("run_rate", "Runs per over"),
        ("dismissal_pct", "Dismissal % of balls"),
        ("dot_pct", "Dot-ball %"),
        ("boundary_pct", "Boundary %"),
        ("runs_variance", "Variance of runs per ball"),
    ]
    groups = ["all", *PHASE_ORDER]
    labels = ["All", "Power-\nplay", "Middle", "Death"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15.5, 3.9))
    x = np.arange(len(groups))
    for ax, (col, title) in zip(axes, metrics):
        for offset, flag, colour, name in ((-0.2, False, BLUE, "Ordinary"), (0.2, True, ORANGE, "High pressure")):
            d = summary[summary["high_pressure"] == flag].set_index("phase").reindex(groups)
            bars = ax.bar(x + offset, d[col], width=0.36, color=colour, label=name, edgecolor=SURFACE, linewidth=1)
            for rect, value in zip(bars, d[col]):
                ax.text(rect.get_x() + rect.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom",
                        fontsize=7.5, color=INK_SECONDARY)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.14)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower right", ncol=2, bbox_to_anchor=(0.9, 1.0), fontsize=9.5)
    fig.subplots_adjust(wspace=0.26)
    finish(
        fig, str(path), "High-pressure balls are played differently, in every phase",
        f"High pressure = {definition}. Each phase is compared with itself, so the gap is not a death-overs effect.",
    )


def plot_close_finishes(curves: list[pd.DataFrame], titles: list[str], cut: float, path: Path) -> None:
    """Ball-by-ball win probability (top) and Leverage Index (bottom) for selected close finishes."""
    apply_style()
    n = len(curves)
    fig, axes = plt.subplots(2, n, figsize=(4.1 * n, 6.2), sharex="col", gridspec_kw={"height_ratios": [1, 1.15]})
    for col, (d, title) in enumerate(zip(curves, titles)):
        top, bottom = axes[0, col], axes[1, col]
        split = int((d["innings"] == 1).sum())
        x = np.arange(len(d))
        top.plot(x, d["wp_chasing"] * 100, color=BLUE, linewidth=1.8)
        top.axhline(50, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
        top.set_ylim(0, 100)
        top.set_title("\n".join(textwrap.fill(line, 40) for line in title.split("\n")), fontsize=8.5)
        colours = np.where(d["li"] >= cut, ORANGE, "#b9d3f3")
        bottom.bar(x, d["li"], width=1.0, color=colours, linewidth=0)
        wk = d["is_wicket"].to_numpy()
        bottom.scatter(x[wk], d["li"].to_numpy()[wk], s=16, color=INK, zorder=3, marker="x", linewidths=1)
        bottom.axhline(cut, color=INK, linewidth=0.9)
        for ax in (top, bottom):
            ax.axvline(split - 0.5, color=MUTED, linewidth=0.9)
            ax.grid(axis="x", visible=False)
        bottom.set_xlabel("Delivery number in the match")
        top.text(split * 0.5, 4, "1st innings", ha="center", fontsize=8, color=INK_SECONDARY)
        top.text(split + (len(d) - split) * 0.5, 4, "chase", ha="center", fontsize=8, color=INK_SECONDARY)
    axes[0, 0].set_ylabel("Win probability of the\nside batting second (%)")
    axes[1, 0].set_ylabel("Leverage Index")
    fig.subplots_adjust(wspace=0.2, hspace=0.08)
    finish(
        fig, str(path), "Pressure ball by ball in four close finishes",
        f"Orange bars are high-pressure balls (LI above {cut:.2f}, the line). Crosses mark wickets.",
    )


def plot_role_shares(position: pd.DataFrame, phase: pd.DataFrame, path: Path) -> None:
    """Share of each batting position's and bowling phase's balls that are high pressure, under both flags."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [2.4, 1]})
    series = (("hp_common_rate", BLUE, "One cut for the whole match"), ("hp_within_innings_rate", AQUA, "Separate cut per innings"))
    for ax, table, key, xlabel in (
        (axes[0], position, "batting_order", "Batting position"),
        (axes[1], phase, "phase", "Bowling phase"),
    ):
        x = np.arange(len(table))
        for offset, (col, colour, name) in zip((-0.2, 0.2), series):
            bars = ax.bar(x + offset, table[col] * 100, width=0.38, color=colour, label=name, edgecolor=SURFACE, linewidth=1)
            for rect, value in zip(bars, table[col] * 100):
                ax.text(rect.get_x() + rect.get_width() / 2, value, f"{value:.0f}", ha="center", va="bottom",
                        fontsize=8, color=INK_SECONDARY)
        ax.axhline(25, color=INK, linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v).capitalize() for v in table[key]])
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.12)
    axes[0].set_ylabel("% of that group's balls that are high pressure")
    axes[0].legend(loc="lower left", ncol=2, fontsize=9, bbox_to_anchor=(0, 1.0))
    fig.subplots_adjust(wspace=0.12)
    finish(
        fig, str(path), "Who gets the pressure balls under each way of drawing the line",
        "The horizontal line is 25%, the share an evenly exposed group would have.",
    )


def plot_same_player_gaps(gaps: pd.DataFrame, path: Path) -> None:
    """Same-player gaps before and after the opposition adjustment, with 95% intervals."""
    apply_style()
    d = gaps[gaps["players"] >= 10].reset_index(drop=True)
    labels = [f"{r.role.capitalize()}s: v {r.against}\nminus v {r.minus_against} ({r.players} players)" for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 0.9 + 0.75 * len(d)))
    y = np.arange(len(d))[::-1]
    for offset, kind, colour, name in ((0.16, "raw", MUTED, "Run value"), (-0.16, "adj", BLUE, "Opposition-adjusted")):
        ax.errorbar(d[f"gap_{kind}"], y + offset, xerr=1.96 * d[f"se_{kind}"], fmt="o", color=colour, markersize=7,
                    elinewidth=1.6, markeredgecolor=SURFACE, markeredgewidth=0.8, label=name)
    ax.axvline(0, color=INK, linewidth=0.9)
    ax.set_yticks(y, labels=labels, fontsize=9)
    ax.set_xlabel("Difference in value per 100 balls for the same players")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    finish(fig, str(path), "The same players, against different opposition",
           "A gap of zero means a player's value does not depend on who they faced. 95% intervals.")


def plot_draft_top(rated: pd.DataFrame, path: Path, role: str, n: int) -> None:
    """Shrunk pressure value with its 90% interval for the top ``n`` by rating, ranks 11+ shaded."""
    apply_style()
    d = rated.sort_values("rank").head(n).iloc[::-1]
    y = np.arange(len(d))
    colour = np.where(d["rank"] <= 10, BLUE if role == "batter" else ORANGE, MUTED)
    fig, ax = plt.subplots(figsize=(9, 0.9 + 0.36 * len(d)))
    ax.hlines(y, d["pressure_value_p05"], d["pressure_value_p95"], color=colour, linewidth=2)
    ax.scatter(d["pressure_value_shrunk"], y, s=46, color=colour, edgecolor=SURFACE, linewidth=0.8, zorder=3)
    ax.axvline(0, color=INK, linewidth=0.9)
    labels = [f"{int(r)}. {nm}  (P top 10 {p:.0%})" for r, nm, p in zip(d["rank"], d["name"], d["p_top_n"])]
    ax.set_yticks(y, labels=labels, fontsize=9)
    ax.set_xlabel("Shrunk pressure value, runs per 100 balls above same-state expectation (90% interval)")
    ax.grid(axis="y", visible=False)
    finish(fig, str(path), f"Draft pressure ranking: {role}s",
           "Ordered by the composite rating. Grey = ranks 11 onwards. Intervals from a match bootstrap.")


SHORT_LABELS = {
    "headline": "A (headline)", "pressure: logistic leverage": "logistic leverage", "pressure: per-innings cut": "per-innings cut",
    "weight: sqrt LI": "sqrt LI", "weight: LI cap": "LI cap", "knockouts x1.25": "knockouts x1.25",
    "eligibility 150": "150 balls", "eligibility 200": "200 balls", "IPL only": "IPL only", "T20I only": "T20I only",
    "associates excluded": "no associates", "role: position / phase relative": "position / phase",
    "pressure: rule index": "rule index", "option B (hierarchical)": "option B",
}


def plot_stability(table: pd.DataFrame, groups: dict[str, list[str]], path: Path, role: str, top_n: int) -> None:
    """Rank of each final top-15 player in every variant, columns grouped, share in the top 10 on the right."""
    apply_style()
    layout = [(g, [c for c in cols if c in table.columns]) for g, cols in groups.items()]
    layout = [(g, cols) for g, cols in layout if cols]
    cols = [c for _, c_list in layout for c in c_list]
    data = table[cols].to_numpy(dtype=float)
    width, height = 2.2 + 0.5 * len(cols), 0.9 + 0.4 * len(table)
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(top=0.94, bottom=0.2, left=0.26, right=0.9)
    shown = np.where(np.isnan(data), np.nan, np.minimum(data, 40))
    ax.imshow(-shown, cmap=SEQUENTIAL, aspect="auto", vmin=-40, vmax=-1)
    for (i, j), value in np.ndenumerate(data):
        if np.isfinite(value):
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=7.5,
                    color=SURFACE if value <= top_n else INK, fontweight="bold" if value <= top_n else "normal")
    position = -0.5
    for group, c_list in layout:
        ax.text(position + len(c_list) / 2, -0.8, "" if group == "headline" else group, ha="center", va="bottom", fontsize=8, color=INK_SECONDARY,
                fontweight="bold")
        position += len(c_list)
        if position < len(cols) - 0.5:
            ax.axvline(position, color=SURFACE, linewidth=3)
    for i, share in enumerate(table["share_in_top_n"]):
        label = "–" if pd.isna(share) else f"{share:.0%}"
        ax.text(len(cols) - 0.2, i, label, ha="left", va="center", fontsize=8.5, color=INK)
    ax.text(len(cols) - 0.2, -0.8, "% in top 10", ha="left", va="bottom", fontsize=8.5, color=INK_SECONDARY, fontweight="bold")
    labels = [f"{int(r)}. {n}  [{s}]" for r, n, s in zip(table["final_rank"], table["name"], table["stability"])]
    ax.set_yticks(range(len(table)), labels=labels, fontsize=8.5)
    ax.set_xticks(range(len(cols)), labels=[SHORT_LABELS.get(c, c.replace("without ", "no ")) for c in cols],
                  rotation=60, ha="right", fontsize=8)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.grid(visible=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    finish(fig, str(path), f"How stable is each {role}'s rank?",
           f"Rank in each variant (dark = higher; bold = inside the top {top_n}; blank = not eligible). "
           "% in top 10 counts the 16 perturbations, not the headline or the alternatives.")


def plot_exposure(players: pd.DataFrame, dependence: pd.DataFrame, path: Path) -> None:
    """Each player's high-pressure share against the share of their balls that came in chases."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, role, colour, label in ((axes[0], "batter", BLUE, "Batters"), (axes[1], "bowler", ORANGE, "Bowlers")):
        d = players[players["role"] == role]
        stats = dependence[dependence["role"] == role].iloc[0]
        x, y = d["chase_share"] * 100, d["hp_share"] * 100
        ax.scatter(x, y, s=34, color=colour, edgecolor=SURFACE, linewidth=0.8, alpha=0.85)
        slope, intercept = np.polyfit(x, y, 1)
        span = np.array([x.min(), x.max()])
        ax.plot(span, intercept + slope * span, color=INK, linewidth=1.2)
        ax.set_title(f"{label} ({len(d)} players)", fontsize=10.5)
        ax.set_xlabel("% of the player's balls that came in chases")
        ax.text(0.03, 0.95, f"Innings mix explains {stats['r2_chase_share'] * 100:.0f}% of the spread",
                transform=ax.transAxes, ha="left", va="top", fontsize=9, color=INK_SECONDARY)
    axes[0].set_ylabel("% of the player's balls that are high pressure")
    fig.subplots_adjust(wspace=0.08)
    finish(
        fig, str(path), "A player's pressure exposure depends on how often their side bats second",
        "One point per player with enough balls. The line is the straight-line fit.",
    )
