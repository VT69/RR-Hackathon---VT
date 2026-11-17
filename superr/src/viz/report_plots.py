"""Phase 7: the figures of the methodology report, drawn from the generated output tables only.

Every figure is built from CSVs under ``outputs/`` (never from objects in memory), so the report's charts
show exactly the numbers in the tables. One style for all of them: A4 text width, 300 dpi, the project's
palette, no chart titles (the report's caption carries the takeaway).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.robustness.sensitivity import variant_groups
from src.viz.style import (
    AQUA, BLUE, GRID, INK, INK_SECONDARY, MUTED, ORANGE, SEQUENTIAL, SURFACE, apply_style,
)
from src.viz.pressure_plots import SHORT_LABELS

logger = logging.getLogger(__name__)
PHASES = ["powerplay", "middle", "death"]
PHASE_LABELS = {"all": "All", "powerplay": "Powerplay", "middle": "Middle", "death": "Death"}
EMPTY = "#f1f0ec"


class ReportFigures:
    """Draws every report figure into ``out_dir`` from the tables under ``outputs_dir``."""

    def __init__(self, outputs_dir: Path, out_dir: Path, config: dict[str, Any]) -> None:
        self.root, self.tables, self.out = outputs_dir, outputs_dir / "tables", out_dir
        self.cfg, self.config = config["report"], config
        self.width, self.dpi = float(self.cfg["figure_width_in"]), int(self.cfg["figure_dpi"])
        self.top_n = int(config["rating"]["top_n"])
        out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ helpers
    def _style(self) -> None:
        apply_style()
        plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                             "ytick.labelsize": 7, "legend.fontsize": 7, "lines.linewidth": 1.4,
                             "axes.titleweight": "bold", "grid.linewidth": 0.5, "axes.linewidth": 0.6})

    def _read(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.tables / name)

    def _save(self, fig: plt.Figure, name: str) -> Path:
        path = self.out / name
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        logger.info("Report figure written: %s", path.name)
        return path

    # ------------------------------------------------------------------ 1. pressure
    def li_heatmap(self) -> Path:
        """Mean Leverage Index by over and wickets down, one panel per innings, one colour scale."""
        self._style()
        grid = self._read("phase2b/li_by_over_wickets.csv")
        grid = grid[grid["balls"] >= self.config["pressure"]["heatmap_min_balls"]]
        vmax = float(np.floor(self._read("phase2b/li_summary.csv").set_index("group").loc["all", "p99"]))
        fig, axes = plt.subplots(1, 2, figsize=(self.width, 2.35), sharey=True)
        image = None
        for ax, innings, label in zip(axes, (1, 2), ("First innings", "Chase")):
            mat = grid[grid["innings"] == innings].pivot(index="wickets_before", columns="overid", values="mean_li")
            mat = mat.reindex(index=range(10), columns=range(1, 21))
            image = ax.imshow(mat.to_numpy(), origin="lower", aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=vmax,
                              extent=(0.5, 20.5, -0.5, 9.5))
            ax.set_facecolor(EMPTY)
            ax.set_title(label, loc="left")
            ax.set_xticks([1, 6, 11, 16, 20])
            ax.set_xlabel("Over")
            ax.grid(False)
            for x in (6.5, 15.5):
                ax.axvline(x, color=SURFACE, linewidth=1.2)
        axes[0].set_ylabel("Wickets already down")
        axes[0].set_yticks(range(0, 10, 2))
        fig.subplots_adjust(wspace=0.05)
        bar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, extend="max")
        bar.set_label(f"Mean Leverage Index (scale capped at {vmax:.0f})")
        bar.outline.set_visible(False)
        return self._save(fig, "li_heatmap.png")

    def close_finishes(self) -> Path:
        """Win probability and leverage ball by ball in the four close finishes, IPL 2023 final first."""
        self._style()
        curves = self._read("phase2b/close_finish_curves.csv")
        matches = self._read("phase2b/close_finishes.csv")
        cut = float(self._read("phase2b/li_cuts.csv").query("flag == 'hp_common'")["li_cut"].iloc[0])
        fig = plt.figure(figsize=(self.width, 3.7))
        outer = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.12)
        for k, m in enumerate(matches.itertuples()):
            inner = outer[k // 2, k % 2].subgridspec(2, 1, height_ratios=[1, 1.25], hspace=0.08)
            top, bottom = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            d = curves[curves["match_id"] == m.match_id].sort_values(["innings", "innings_ball_num"]).reset_index(drop=True)
            x = np.arange(len(d))
            split = int((d["innings"] == 1).sum())
            top.plot(x, d["wp_chasing"] * 100, color=BLUE, linewidth=1.1)
            top.axhline(50, color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
            top.set_ylim(0, 100)
            top.set_yticks([0, 50, 100])
            top.set_xticks([])
            bottom.bar(x, d["li"], width=1.0, color=np.where(d["li"] >= cut, ORANGE, "#b9d3f3"), linewidth=0)
            wk = d["is_wicket"].to_numpy(bool)
            bottom.scatter(x[wk], d["li"].to_numpy()[wk], s=7, color=INK, marker="x", linewidths=0.7, zorder=3)
            bottom.axhline(cut, color=INK, linewidth=0.6)
            for ax in (top, bottom):
                ax.axvline(split - 0.5, color=MUTED, linewidth=0.6)
                ax.grid(axis="x", visible=False)
                ax.set_xlim(-1, len(d))
            top.text(split / 2, 6, "1st innings", ha="center", fontsize=6, color=INK_SECONDARY)
            top.text(split + (len(d) - split) / 2, 6, "chase", ha="center", fontsize=6, color=INK_SECONDARY)
            heading = "IPL 2023 final (knockout)" if m.category == "Knockout" else m.category
            top.set_title(f"{heading}, {m.match_date}\n{m.home_team} v {m.away_team}", loc="left", fontsize=6.4,
                          fontweight="normal")
            if k % 2 == 0:
                top.set_ylabel("Chasing side\nWP (%)", fontsize=6.5)
                bottom.set_ylabel("Leverage Index", fontsize=6.5)
            if k // 2 == 1:
                bottom.set_xlabel("Delivery number in the match")
        handles = [Line2D([], [], color=BLUE, label="Win probability of the side batting second"),
                   Patch(color=ORANGE, label=f"High-pressure ball (LI above {cut:.2f}, the black line)"),
                   Patch(color="#b9d3f3", label="Other ball"),
                   Line2D([], [], color=INK, marker="x", linestyle="none", markersize=4, label="Wicket")]
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.1, 0.94), ncol=2, frameon=False)
        return self._save(fig, "close_finishes.png")

    def separation(self) -> Path:
        """High-pressure v ordinary balls on four outcomes, overall and within each phase."""
        self._style()
        d = self._read("phase2b/separation.csv").query("flag == 'hp_common'")
        metrics = [("run_rate", "Runs per over"), ("dismissal_pct", "Dismissals / 100 balls"),
                   ("dot_pct", "Dot balls %"), ("boundary_pct", "Boundaries %")]
        groups = ["all", *PHASES]
        fig, axes = plt.subplots(1, 4, figsize=(self.width, 1.75), sharey=True)
        y = np.arange(len(groups))[::-1]
        for ax, (col, label) in zip(axes, metrics):
            ordinary = d[~d["high_pressure"]].set_index("phase").reindex(groups)[col].to_numpy()
            pressure = d[d["high_pressure"]].set_index("phase").reindex(groups)[col].to_numpy()
            ax.hlines(y, ordinary, pressure, color=GRID, linewidth=2.2, zorder=1)
            ax.scatter(ordinary, y, s=22, color=MUTED, label="Ordinary", zorder=3)
            ax.scatter(pressure, y, s=22, color=ORANGE, label="High pressure", zorder=3)
            ax.set_title(label, loc="left", fontsize=7.2)
            ax.grid(axis="y", visible=False)
            ax.margins(x=0.15)
        axes[0].set_yticks(y, [PHASE_LABELS[g] for g in groups])
        axes[0].set_ylim(-0.6, len(groups) - 0.4)
        axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.16), ncol=2, frameon=False)
        fig.subplots_adjust(wspace=0.15)
        return self._save(fig, "separation.png")

    def rule_vs_leverage(self) -> Path:
        """Balls flagged by leverage only, the rule index only, or both: how close and how late they are."""
        self._style()
        d = self._read("phase2b/rule_vs_leverage_profile.csv").set_index("agreement").reindex(
            ["both", "leverage only", "rule only"])
        labels = ["Both", "Leverage only", "Rule only"]
        colours = [AQUA, BLUE, ORANGE]
        panels = [("mean_li", "Mean Leverage Index", 1.0, "{:.2f}"),
                  ("mean_wp_distance_from_even", "WP distance from 50% (pts)", 100.0, "{:.0f}"),
                  ("share_chase", "Share in chases (%)", 100.0, "{:.0f}")]
        fig, axes = plt.subplots(1, 3, figsize=(self.width, 1.55), sharey=True)
        y = np.arange(3)[::-1]
        for ax, (col, title, scale, fmt) in zip(axes, panels):
            vals = d[col].to_numpy() * scale
            ax.barh(y, vals, color=colours, height=0.62)
            for yy, v in zip(y, vals):
                ax.text(v, yy, " " + fmt.format(v), va="center", fontsize=6.5, color=INK_SECONDARY)
            ax.set_title(title, loc="left", fontsize=7)
            ax.grid(axis="y", visible=False)
            ax.margins(x=0.2)
        axes[0].set_yticks(y, labels)
        fig.subplots_adjust(wspace=0.22)
        return self._save(fig, "rule_vs_leverage.png")

    # ------------------------------------------------------------------ 2. performance
    def wicket_cost_grid(self) -> Path:
        """Mean wicket cost in runs by overs left and wickets in hand, both innings, one colour scale."""
        self._style()
        grid = self._read("phase3/wicket_cost_by_overs_wickets.csv")
        grid = grid[grid["balls"] >= self.config["pressure"]["heatmap_min_balls"]]
        cap = float(self._read("phase3/wicket_cost_cap.csv")["cap_after_calibration"].iloc[0])
        fig, axes = plt.subplots(1, 2, figsize=(self.width, 2.35), sharey=True)
        image = None
        for ax, innings, label in zip(axes, (1, 2), ("First innings", "Chase")):
            mat = grid[grid["innings"] == innings].pivot(index="wickets_in_hand", columns="overs_left", values="mean_cost")
            mat = mat.reindex(index=range(1, 11), columns=range(20, 0, -1))
            image = ax.imshow(mat.to_numpy(), origin="lower", aspect="auto", cmap=SEQUENTIAL, vmin=0, vmax=cap,
                              extent=(0.5, 20.5, 0.5, 10.5))
            ax.set_facecolor(EMPTY)
            ax.set_title(label, loc="left")
            ax.set_xticks([1, 5, 10, 15, 20], ["20", "16", "11", "6", "1"])
            ax.set_xlabel("Overs left")
            ax.grid(False)
        axes[0].set_ylabel("Wickets in hand")
        axes[0].set_yticks([1, 4, 7, 10])
        fig.subplots_adjust(wspace=0.05)
        bar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
        bar.set_label("Mean wicket cost (runs)")
        bar.outline.set_visible(False)
        return self._save(fig, "wicket_cost_grid.png")

    def wicket_cost_check(self) -> Path:
        """Runs actually lost after a first-innings wicket against the model's cost, before and after calibration."""
        self._style()
        d = self._read("phase3/wicket_cost_event_study.csv").set_index("overs_left").reindex(
            ["16-20", "11-15", "6-10", "3-5", "1-2"])
        x = np.arange(len(d))
        fig, ax = plt.subplots(figsize=(self.width, 1.9))
        ax.errorbar(x - 0.16, d["runs_actually_lost"], yerr=1.96 * d["se"], fmt="o", color=INK, markersize=4,
                    elinewidth=0.9, capsize=2, label="Runs actually lost (95% interval)")
        ax.scatter(x + 0.0, d["cost_before"], s=22, color=MUTED, marker="s", label="Model cost before calibration", zorder=3)
        ax.scatter(x + 0.16, d["cost_after"], s=22, color=BLUE, marker="D", label="Model cost after calibration", zorder=3)
        ax.set_xticks(x, [f"{b} overs left" for b in d.index])
        ax.set_ylabel("Runs per wicket")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right", frameon=False)
        return self._save(fig, "wicket_cost_check.png")

    # ------------------------------------------------------------------ 3. fair comparisons
    def exposure(self) -> Path:
        """Share of balls that are high pressure by innings and phase; team chasing share against batting exposure."""
        self._style()
        d = self._read("phase2b/exposure_by_innings_phase.csv")
        teams = self._read("phase2b/exposure_by_team.csv")
        fig, axes = plt.subplots(1, 2, figsize=(self.width, 2.0), gridspec_kw={"width_ratios": [1, 1.25]})
        ax = axes[0]
        x = np.arange(3)
        for off, inn, colour, name in ((-0.19, "1", MUTED, "First innings"), (0.19, "2", ORANGE, "Chase")):
            s = d[d["innings"].astype(str) == inn].set_index("phase").reindex(PHASES)["hp_rate"] * 100
            bars = ax.bar(x + off, s, width=0.36, color=colour, label=name, edgecolor=SURFACE, linewidth=0.5)
            for rect, v in zip(bars, s):
                ax.text(rect.get_x() + rect.get_width() / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=6.3,
                        color=INK_SECONDARY)
        ax.set_xticks(x, [PHASE_LABELS[p] for p in PHASES])
        ax.set_ylabel("% of balls high pressure")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, float(d["hp_rate"].max()) * 100 * 1.4)
        ax.legend(loc="upper center", ncol=2, frameon=False)
        ax = axes[1]
        for comp, colour in (("IPL", BLUE), ("T20I", AQUA)):
            t = teams[teams["comp_type"] == comp]
            ax.scatter(t["share_of_matches_chasing"] * 100, t["batting_hp_share"] * 100, s=18, color=colour,
                       edgecolor=SURFACE, linewidth=0.5, label=f"{comp} side", zorder=3)
        for name in self.cfg["exposure_labels"]:
            r = teams[teams["team"] == name]
            if len(r):
                r = r.iloc[0]
                ax.annotate(f"{name} ({r.comp_type})", (r.share_of_matches_chasing * 100, r.batting_hp_share * 100),
                            xytext=(4, -2), textcoords="offset points", fontsize=6, color=INK_SECONDARY)
        ax.set_xlabel("% of matches batting second")
        ax.set_ylabel("% of the side's batting balls\nthat are high pressure")
        ax.legend(loc="upper left", frameon=False)
        fig.subplots_adjust(wspace=0.32)
        return self._save(fig, "exposure.png")

    def same_player_gaps(self) -> Path:
        """Acceptance test: the same players' value against different opposition, before and after adjustment."""
        self._style()
        d = self._read("phase4/same_player_gaps.csv")
        d = d[d["players"] >= 10].reset_index(drop=True)
        names = {"T20I top-8": "top-8", "T20I other full member": "other full", "IPL": "IPL"}
        labels = [f"{r.role.capitalize()}s ({r.players}):\nv {names[r.against]} minus v {names[r.minus_against]}"
                  for r in d.itertuples()]
        fig, ax = plt.subplots(figsize=(self.width, 2.5))
        y = np.arange(len(d))[::-1]
        for off, kind, colour, name in ((0.15, "raw", MUTED, "Before adjustment"), (-0.15, "adj", BLUE, "After adjustment")):
            ax.errorbar(d[f"gap_{kind}"], y + off, xerr=1.96 * d[f"se_{kind}"], fmt="o", color=colour, markersize=4,
                        elinewidth=1.0, label=name)
        ax.axvline(0, color=INK, linewidth=0.7)
        ax.set_yticks(y, labels)
        ax.set_xlabel("Gap in value per 100 balls for the same players (95% interval)")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
        return self._save(fig, "same_player_gaps.png")

    # ------------------------------------------------------------------ 4. results
    def top15(self) -> Path:
        """Final ranks 1-15 per role: shrunk pressure value with its 90% interval, coloured by profile label."""
        self._style()
        d = pd.read_csv(self.root / "final_rankings_top15.csv")
        fig, axes = plt.subplots(2, 1, figsize=(self.width, 4.8), gridspec_kw={"hspace": 0.32})
        colours = {"All-round": BLUE, "Pressure-specific": ORANGE}
        for ax, role in zip(axes, ("batter", "bowler")):
            r = d[d["role"] == role].sort_values("rank").iloc[::-1].reset_index(drop=True)
            y = np.arange(len(r))
            colour = [colours[lab] if k <= self.top_n else MUTED for lab, k in zip(r["label"], r["rank"])]
            ax.hlines(y, r["pressure_value_p05"], r["pressure_value_p95"], color=colour, linewidth=1.6)
            ax.scatter(r["pressure_value_shrunk"], y, s=20, color=colour, edgecolor=SURFACE, linewidth=0.5, zorder=3)
            ax.axvline(0, color=INK, linewidth=0.6)
            ax.axhline(len(r) - self.top_n - 0.5, color=GRID, linewidth=0.8, linestyle=(0, (3, 2)))
            ax.set_yticks(y, [f"{int(k)}. {n}" for k, n in zip(r["rank"], r["player_name"])])
            xmax = float(r["pressure_value_p95"].max())
            for yy, p in zip(y, r["p_top_10"]):
                ax.text(xmax * 1.04 + 1, yy, f"{p:.2f}", va="center", fontsize=6.3, color=INK_SECONDARY)
            ax.text(xmax * 1.04 + 1, len(r) - 0.2, "P(top 10)", va="bottom", fontsize=6.3, color=INK_SECONDARY,
                    fontweight="bold")
            ax.set_xlim(right=xmax * 1.04 + 1 + 0.12 * (xmax - float(r["pressure_value_p05"].min())))
            ax.set_title(f"{role.capitalize()}s", loc="left")
            ax.grid(axis="y", visible=False)
        axes[1].set_xlabel("Shrunk pressure value, runs per 100 pressure balls above expectation (90% interval)")
        handles = [Line2D([], [], color=BLUE, marker="o", label="All-round (option B agrees)"),
                   Line2D([], [], color=ORANGE, marker="o", label="Pressure-specific (B rank > "
                          f"{self.cfg['framing_min_b_rank']})"),
                   Line2D([], [], color=MUTED, marker="o", label="Ranks 11-15")]
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.12, 0.905), ncol=3, frameon=False)
        return self._save(fig, "top15.png")

    def overall_v_pressure(self) -> Path:
        """Rank on all balls against the final pressure rank, every eligible player, named examples labelled."""
        self._style()
        d = pd.read_csv(self.root / "final_rankings_eligible.csv")
        fig, axes = plt.subplots(1, 2, figsize=(self.width, 3.0))
        for ax, role, colour in zip(axes, ("batter", "bowler"), (BLUE, ORANGE)):
            r = d[d["role"] == role]
            top = r["rank"] <= self.top_n
            n = len(r)
            ax.plot([1, n], [1, n], color=GRID, linewidth=0.8, zorder=1)
            ax.scatter(r.loc[~top, "overall_rank"], r.loc[~top, "rank"], s=7, color=MUTED, alpha=0.55, linewidth=0)
            ax.scatter(r.loc[top, "overall_rank"], r.loc[top, "rank"], s=16, color=colour, edgecolor=SURFACE,
                       linewidth=0.5, zorder=3, label=f"Final top {self.top_n}")
            labelled = r[r["player_id"].isin(self.cfg["overall_v_pressure_labels"][role])].sort_values("overall_rank")
            in_band = 0
            for row in labelled.itertuples():
                if row.rank <= self.top_n:
                    # Inside the top-10 band: label below the point, staggered so neighbours do not collide.
                    offset, ha = (0, -9 - 8 * (in_band % 2)), "center"
                    in_band += 1
                else:
                    near_right = row.overall_rank < 0.3 * n      # the x axis runs from rank n (left) to 1 (right)
                    offset, ha = (-5 if near_right else 5, 2), "right" if near_right else "left"
                ax.annotate(row.player_name, (row.overall_rank, row.rank), xytext=offset, textcoords="offset points",
                            ha=ha, va="top" if ha == "center" else "baseline", fontsize=6, color=INK,
                            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.5, "shrinkA": 0, "shrinkB": 2})
            ax.axhspan(0.5, self.top_n + 0.5, color=GRID, alpha=0.45, linewidth=0, zorder=0)
            ax.set_xlim(n + 3, -2)
            ax.set_ylim(n + 3, -2)
            ax.set_title(f"{role.capitalize()}s ({n} eligible)", loc="left")
            ax.set_xlabel("Rank on all balls (opposition-adjusted value)")
            ax.legend(loc="lower right", frameon=False)
        axes[0].set_ylabel("Final pressure rank")
        fig.subplots_adjust(wspace=0.22)
        return self._save(fig, "overall_v_pressure.png")

    # ------------------------------------------------------------------ 5. reliability
    def reliability(self) -> Path:
        """Split-half reliability (Spearman-Brown corrected) of each component, with the 90% range over splits."""
        self._style()
        d = self._read("phase5/split_half_reliability.csv").query("weighting == 'uniform'")
        comps = [("pressure", "Pressure value\n(the ranking)"), ("consistency", "Consistency"),
                 ("uplift", "Uplift (\"clutch\")"), ("usual", "Usual value\n(ordinary balls)")]
        fig, ax = plt.subplots(figsize=(self.width, 2.1))
        x = np.arange(len(comps))
        for off, role, colour in ((-0.17, "batter", BLUE), (0.17, "bowler", ORANGE)):
            s = d[d["role"] == role].set_index("component").reindex([c for c, _ in comps])
            ax.bar(x + off, s["random_mean"], width=0.32, color=colour, label=f"{role.capitalize()}s", edgecolor=SURFACE)
            ax.errorbar(x + off, s["random_mean"], yerr=[s["random_mean"] - s["random_p05"], s["random_p95"] - s["random_mean"]],
                        fmt="none", ecolor=INK, elinewidth=0.7, capsize=2)
        ax.axhline(0, color=INK, linewidth=0.6)
        ax.set_xticks(x, [label for _, label in comps])
        ax.set_ylabel("Split-half reliability\n(corrected; 90% range)")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", frameon=False, ncol=2)
        return self._save(fig, "reliability.png")

    def stability(self, role: str) -> Path:
        """Compact rank-stability heatmap for the final top 15 of one role, columns grouped."""
        self._style()
        table = self._read(f"phase6/stability_{role}s.csv").sort_values("final_rank").reset_index(drop=True)
        variants = self._read("phase6/variant_ranks.csv")["variant"].unique().tolist()
        groups = variant_groups(variants)
        layout = [(g, [c for c in cols if c in table.columns]) for g, cols in groups.items()]
        layout = [(g, cols) for g, cols in layout if cols]
        cols = [c for _, cl in layout for c in cl]
        data = table[cols].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(self.width, 2.9))
        shown = np.where(np.isnan(data), np.nan, np.minimum(data, 40))
        ax.imshow(-shown, cmap=SEQUENTIAL, aspect="auto", vmin=-40, vmax=-1)
        ax.set_facecolor(EMPTY)
        for (i, j), value in np.ndenumerate(data):
            if np.isfinite(value):
                inside = value <= self.top_n
                ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=5.6,
                        color=SURFACE if inside else INK, fontweight="bold" if inside else "normal")
        pos = -0.5
        for group, cl in layout:
            label = "" if len(cl) < 2 else group        # single-column groups are named in the tick label instead
            ax.text(pos + len(cl) / 2, -0.75, label, ha="center", va="bottom", fontsize=6,
                    color=INK_SECONDARY, fontweight="bold")
            pos += len(cl)
            if pos < len(cols) - 0.5:
                ax.axvline(pos, color=SURFACE, linewidth=2.2)
        for i, share in enumerate(table["share_in_top_n"]):
            ax.text(len(cols) - 0.25, i, "–" if pd.isna(share) else f"{share:.0%}", ha="left", va="center", fontsize=6.3)
        ax.text(len(cols) - 0.25, -0.75, "% in top 10", ha="left", va="bottom", fontsize=6, color=INK_SECONDARY,
                fontweight="bold")
        ax.set_yticks(range(len(table)), [f"{int(r)}. {n} [{s}]" for r, n, s in
                                          zip(table["final_rank"], table["name"], table["stability"])], fontsize=6.3)
        ticks = {**SHORT_LABELS, "role: position / phase relative": "role: position / phase"}
        ax.set_xticks(range(len(cols)), [ticks.get(c, c.replace("without ", "no ")) for c in cols],
                      rotation=55, ha="right", fontsize=6.2)
        ax.set_xlim(-0.5, len(cols) - 0.5)
        ax.grid(visible=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        name = f"stability_{role}s.png"
        return self._save(fig, name)

    def all(self) -> list[Path]:
        """Draw every report figure; returns the paths written."""
        return [self.li_heatmap(), self.close_finishes(), self.separation(), self.rule_vs_leverage(),
                self.wicket_cost_grid(), self.wicket_cost_check(), self.exposure(), self.same_player_gaps(),
                self.top15(), self.overall_v_pressure(), self.reliability(), self.stability("batter"),
                self.stability("bowler")]
