"""Phase 3 evidence tables: are the expectations right, and does run value behave?"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.metrics.wicket_cost import event_rows, runs_lost
from src.models.expected import EXPECTATIONS
from src.models.evaluate import log_loss

PER_100 = 100.0
PER_OVER = 6


def clustered_rate(rows: pd.DataFrame) -> tuple[float, float]:
    """Value per ball and its standard error, treating matches (not balls) as independent."""
    per_match = rows.groupby("match_id").agg(value=("value", "sum"), balls=("count", "sum"))
    total = per_match["balls"].sum()
    rate = per_match["value"].sum() / total
    m = len(per_match)
    if m < 2:
        return float(rate), float("nan")
    resid = per_match["value"] - rate * per_match["balls"]
    return float(rate), float(np.sqrt(m / (m - 1) * (resid**2).sum()) / total)


def zero_mean_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean run value per 100 balls, overall and by slice. By construction it should be near zero."""
    role = rows["role"].iloc[0]
    frame = rows.assign(innings_phase=rows["innings"].astype(str) + " " + rows["phase"],
                        pressure=np.where(rows["pressure"], "high pressure", "ordinary"))
    slices = [("overall", None), ("innings", "innings"), ("phase", "phase"), ("innings x phase", "innings_phase"),
              ("pressure flag", "pressure"), ("season", "year"), ("competition", "comp_type"),
              ("involves associate", "involves_associate")]
    out = []
    for label, column in slices:
        groups = [("all", frame)] if column is None else frame.groupby(column)
        for level, part in groups:
            rate, se = clustered_rate(part)
            out.append({"role": role, "slice": label, "level": str(level), "balls": part["count"].sum(),
                        "value_per_100": rate * PER_100, "se": se * PER_100, "z": rate / se if se else np.nan})
    return pd.DataFrame(out)


def raw_gap_table(rows: pd.DataFrame) -> pd.DataFrame:
    """The raw pressure-v-ordinary gap next to the same gap in run value (which should vanish)."""
    role = rows["role"].iloc[0]
    out = []
    for flag, part in rows.groupby("pressure"):
        balls = part["count"].sum()
        rate, se = clustered_rate(part)
        out.append({"role": role, "pressure": bool(flag), "balls": balls, "runs_per_100": part["runs"].sum() / balls * PER_100,
                    "wickets_per_100": part["wickets"].sum() / balls * PER_100, "value_per_100": rate * PER_100, "value_se": se * PER_100})
    return pd.DataFrame(out)


def _target_rows(balls: pd.DataFrame, name: str) -> pd.DataFrame:
    spec = EXPECTATIONS[name]
    mask = balls["state_ok"] & (balls["is_ball_faced"] if spec.faced_only else True)
    return balls[mask]


def calibration_table(balls: pd.DataFrame, expected: pd.DataFrame, bins: int) -> pd.DataFrame:
    """Predicted v actual by tenth of the prediction range, per expectation and innings (held-out)."""
    out = []
    for name, spec in EXPECTATIONS.items():
        rows = _target_rows(balls, name)
        for innings, part in rows.groupby("innings"):
            pred, actual = expected.loc[part.index, name], part[spec.target]
            bucket = pd.qcut(pred, bins, labels=False, duplicates="drop")
            table = pd.DataFrame({"predicted": pred, "actual": actual}).groupby(bucket).agg(
                predicted=("predicted", "mean"), actual=("actual", "mean"), balls=("actual", "size"))
            out.append(table.reset_index(drop=True).assign(expectation=name, innings=innings, bin=lambda d: d.index + 1))
    return pd.concat(out, ignore_index=True)


def calibration_by_slice(balls: pd.DataFrame, expected: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Predicted v actual by innings, phase and pressure flag: the slices the ratings will be cut on."""
    out = []
    for name, spec in EXPECTATIONS.items():
        rows = _target_rows(balls, name)
        frame = pd.DataFrame({"innings": rows["innings"], "phase": rows["phase"], "pressure": rows[flag],
                              "predicted": expected.loc[rows.index, name], "actual": rows[spec.target]})
        table = frame.groupby(["innings", "phase", "pressure"]).agg(
            balls=("actual", "size"), predicted=("predicted", "mean"), actual=("actual", "mean"), sd=("actual", "std"))
        table["gap"] = table["actual"] - table["predicted"]
        table["gap_in_se"] = table["gap"] / (table["sd"] / np.sqrt(table["balls"]))
        out.append(table.drop(columns="sd").reset_index().assign(expectation=name))
    return pd.concat(out, ignore_index=True)


def expectation_scores(balls: pd.DataFrame, variants: dict[str, pd.DataFrame], clip: float) -> pd.DataFrame:
    """Held-out error of each set of expectations: squared error for runs, log-loss for wickets."""
    out = []
    for name, spec in EXPECTATIONS.items():
        rows = _target_rows(balls, name)
        actual = rows[spec.target].to_numpy()
        row: dict[str, Any] = {"expectation": name, "metric": "log-loss" if spec.binary else "mean squared error", "balls": len(rows)}
        for label, frame in variants.items():
            pred = frame.loc[rows.index, name].to_numpy()
            row[label] = log_loss(np.clip(pred, clip, 1 - clip), actual, clip) if spec.binary else float(np.mean((actual - pred) ** 2))
        out.append(row)
    return pd.DataFrame(out)


def cost_grid(balls: pd.DataFrame, per_over: int) -> pd.DataFrame:
    """Mean wicket cost (runs) by innings, overs left and wickets in hand, from the balls actually bowled."""
    scored = balls[balls["state_ok"]]
    overs_left = np.ceil(scored["balls_remaining"] / per_over).astype(int)
    grid = scored.assign(overs_left=overs_left).groupby(["innings", "overs_left", "wickets_in_hand"])["c"].agg(
        mean_cost="mean", balls="size")
    return grid.reset_index()


def cost_summary(balls: pd.DataFrame, cost: pd.DataFrame, cap: float) -> pd.DataFrame:
    """Distribution of the wicket cost and how often the fallback and the cap apply, by innings."""
    scored = balls["state_ok"]
    out = []
    for label, mask in (("all", scored), ("first innings", scored & (balls["innings"] == 1)), ("chase", scored & (balls["innings"] == 2))):
        c, raw = cost.loc[mask, "c"], cost.loc[mask & ~cost["used_fallback"], "c_raw"]
        q = raw.quantile([0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        out.append({"group": label, "balls": int(mask.sum()), "mean_cost": c.mean(), "median_cost": c.median(),
                    **{f"ratio_p{int(k * 100)}": v for k, v in q.items()}, "ratio_max": raw.max(),
                    "fallback_share": cost.loc[mask, "used_fallback"].mean(), "capped_share": cost.loc[mask, "capped"].mean(),
                    "last_ball_share": cost.loc[mask, "last_ball"].mean(), "cap": cap})
    return pd.DataFrame(out)


def cost_event_study(balls: pd.DataFrame, costs: dict[str, pd.Series], config: dict[str, Any]) -> pd.DataFrame:
    """An independent check of the wicket cost: runs actually lost after a wicket fell.

    Full-length first innings only (no target, so "runs scored afterwards" is the whole
    story). For every legal ball on which a wicket fell: runs added after a non-wicket
    ball in the same state, minus the runs that side actually added afterwards. Compared
    with each version of the model's cost (``costs``) on those same balls.
    """
    rows = event_rows(balls, config)
    lost = runs_lost(rows, pd.Series(True, index=rows.index), config["value"]["calibration_min_cell"])
    fell = rows.loc[lost.index].assign(runs_lost=lost, **{name: c.loc[lost.index] for name, c in costs.items()})
    edges = [0, *config["value"]["event_study_bands"]]
    labels = [f"{lo + 1}-{hi}" for lo, hi in zip(edges[:-1], edges[1:])]
    fell["band"] = pd.cut(fell["overs_left"], edges, labels=labels)
    out = []
    for label, part in [("all", fell), *fell.groupby("band", observed=True)]:
        row: dict[str, Any] = {"overs_left": str(label), "wickets": len(part), "runs_actually_lost": part["runs_lost"].mean(),
                               "se": part["runs_lost"].sem()}
        for name in costs:
            row[f"cost_{name}"] = part[name].mean()
            row[f"gap_{name}"] = part[name].mean() - part["runs_lost"].mean()
        out.append(row)
    return pd.DataFrame(out)


def value_by_batting_position(valued: pd.DataFrame, versions: dict[str, str]) -> pd.DataFrame:
    """Mean batting value per 100 balls faced by batting position, for each version of the value column."""
    faced = valued[valued["state_ok"] & valued["is_ball_faced"] & (valued["batting_order"] <= 11)]
    g = faced.groupby("batting_order")
    out = pd.DataFrame({"balls": g.size()})
    for label, column in versions.items():
        out[f"value_per_100_{label}"] = g[column].mean() * PER_100
    out["wickets_per_100"] = g["striker_out"].mean() * PER_100
    out["mean_wicket_cost_after"] = faced[faced["striker_out"] == 1].groupby("batting_order")["c"].mean()
    return out.reset_index()


def player_before_after(before: pd.DataFrame, after: pd.DataFrame, min_balls: int, positions: pd.Series) -> pd.DataFrame:
    """Overall value per 100 balls for each player before and after a change, with rank in both."""
    traditional = ["strike_rate", "average"] if "strike_rate" in before.columns else ["economy", "balls_per_wicket"]
    keep = ["player_id", "name", "teams", "balls", *traditional, "value_per_100"]
    pool_b = before[before["balls"] >= min_balls][keep].rename(columns={"value_per_100": "value_before"})
    pool_a = after[after["balls"] >= min_balls][["player_id", "value_per_100"]].rename(columns={"value_per_100": "value_after"})
    out = pool_b.merge(pool_a, on="player_id")
    out["change"] = out["value_after"] - out["value_before"]
    out["rank_before"] = out["value_before"].rank(ascending=False).astype(int)
    out["rank_after"] = out["value_after"].rank(ascending=False).astype(int)
    out["usual_position"] = out["player_id"].map(positions)
    return out.sort_values("change", ascending=False)


def value_spread(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-ball spread of run value by innings and phase, and how much of it is the wicket term."""
    counted = rows[rows["count"] > 0]
    g = counted.groupby(["innings", "phase"])
    out = pd.DataFrame({"balls": g.size(), "sd_value": g["value"].std(), "sd_runs_part": g["value_runs"].std(),
                        "sd_wicket_part": g["value_wickets"].std()})
    return out.reset_index().assign(role=rows["role"].iloc[0])


def face_validity(players: pd.DataFrame, min_balls: int) -> pd.DataFrame:
    """Correlation of overall run value per 100 balls with the traditional numbers, across players."""
    role = players["role"].iloc[0]
    pool = players[players["balls"] >= min_balls]
    columns = ["strike_rate", "average"] if role == "batter" else ["economy", "balls_per_wicket"]
    out = []
    for column in columns:
        pair = pool[["value_per_100", column]].dropna()
        out.append({"role": role, "players": len(pair), "traditional": column,
                    "pearson": pair.corr().iloc[0, 1], "spearman": pair.corr(method="spearman").iloc[0, 1]})
    return pd.DataFrame(out)


def extremes(players: pd.DataFrame, min_balls: int, n: int) -> pd.DataFrame:
    """Top and bottom ``n`` players on overall value per 100 balls (all balls, no pressure layer)."""
    pool = players[players["balls"] >= min_balls].sort_values("value_per_100", ascending=False)
    top = pool.head(n).assign(group="top", rank=range(1, n + 1))
    bottom = pool.tail(n).iloc[::-1].assign(group="bottom", rank=range(1, n + 1))
    return pd.concat([top, bottom], ignore_index=True)


def wpa_checks(rows: pd.DataFrame, players: pd.DataFrame, min_balls: int) -> pd.DataFrame:
    """WPA should average zero, and a player's WPA should track their leverage-weighted run value."""
    role = rows["role"].iloc[0]
    have = rows[rows["wpa"].notna()]
    weighted = (rows["value"] * rows["li"]).groupby(rows["player_id"]).sum().rename("li_weighted_value")
    pool = players[players["balls"] >= min_balls].set_index("player_id").join(weighted)
    pair = pool[["wpa_total", "li_weighted_value"]]
    plain = pool[["wpa_total", "value_total"]]
    return pd.DataFrame([{
        "role": role, "balls_with_wpa": len(have), "mean_wpa_per_100_balls_pct": have["wpa"].mean() * PER_100 * PER_100,
        "players": len(pool), "corr_wpa_v_li_weighted_value": pair.corr().iloc[0, 1],
        "corr_wpa_v_unweighted_value": plain.corr().iloc[0, 1],
    }])


def _real_spread(values: pd.Series, se: pd.Series) -> float:
    """Spread across players left after removing what measurement noise alone would produce."""
    return float(np.sqrt(max(values.var() - (se**2).mean(), 0.0)))


def split_reliability(split: pd.DataFrame, thresholds: list[int]) -> pd.DataFrame:
    """How noisy pressure value, usual value and uplift are at different sample sizes (no rankings).

    For each: the typical standard error, the observed spread across players, and the
    implied real spread (observed variance minus mean squared standard error). A real
    spread near zero means the component mostly measures luck at that sample size.
    """
    out = []
    for n in thresholds:
        part = split[split["pressure_balls"] >= n]
        out.append({"role": split["role"].iloc[0], "min_pressure_balls": n, "players": len(part),
                    "real_sd_pressure": _real_spread(part["pressure_value"], part["pressure_se"]),
                    "real_sd_usual": _real_spread(part["usual_value"], part["usual_se"]),
                    "real_sd_uplift": _real_spread(part["uplift"], part["uplift_se"]),
                    "median_pressure_se": part["pressure_se"].median(), "median_usual_se": part["usual_se"].median(),
                    "median_uplift_se": part["uplift_se"].median(), "sd_pressure_value": part["pressure_value"].std(),
                    "sd_usual_value": part["usual_value"].std(), "sd_uplift": part["uplift"].std(),
                    "median_matched_coverage": part["matched_coverage"].median(),
                    "share_fully_covered": (part["matched_coverage"] >= 0.999).mean()})
    return pd.DataFrame(out)
