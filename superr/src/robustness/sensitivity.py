"""Phase 6: tie-break rule, notable absences, evidence cards and the sensitivity analysis.

Every sensitivity variant changes one thing (pressure definition, weighting, eligibility,
competition, season, role baseline, match importance) and re-scores the players with the
headline component weights held fixed, so that only the data or definition differs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.metrics.player_value import player_balls
from src.rating.ratings import point_scores

logger = logging.getLogger(__name__)
PER_100 = 100.0


# ---------------------------------------------------------------- tie-break
def _indistinguishable(x: pd.Series, y: pd.Series, max_p_gap: float) -> tuple[bool, bool, float]:
    """Overlapping 90% composite intervals and P(top 10) within ``max_p_gap``."""
    overlap = bool(x["composite_p05"] <= y["composite_p95"] and y["composite_p05"] <= x["composite_p95"])
    gap = float(abs(x["p_top_n"] - y["p_top_n"]))
    return overlap and gap <= max_p_gap, overlap, gap


def apply_tie_break(rated: pd.DataFrame, zone: int, max_p_gap: float, classes: pd.Series | None = None,
                    top_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Order statistically indistinguishable players in ranks 1..zone of A by the hierarchical rating (B).

    Pairwise, not chained: whenever two players are distinguishable from each other directly
    (no interval overlap, or P(top 10) further apart than ``max_p_gap``), they keep A's order.
    Among the orders that respect every such pair, the one that follows B as closely as
    possible is chosen (at each position, the highest-B player not outranked by a remaining
    distinguishable player).

    Stability guard (``classes``: player_id -> stable / moderately stable / fragile):
      * the tie-break never moves a fragile player above a stable or moderately stable one
        that A ranks higher;
      * only stable or moderately stable players may occupy the final top ``top_n``; a fragile
        player who would be there moves just below it, and the next non-fragile player in the
        order moves up.
    Returns (table with ``final_rank`` and ``final_reason``, adjacent-pair checks).
    """
    a = rated.sort_values("rank").reset_index(drop=True)
    cls = a["player_id"].map(classes).fillna("fragile") if classes is not None else pd.Series("stable", index=a.index)
    a["stability"] = cls
    head, rest = a.head(zone).copy(), a.iloc[zone:].copy()
    rows = [head.iloc[i] for i in range(len(head))]
    fragile = [classes is not None and rows[i]["stability"] == "fragile" for i in range(len(rows))]
    must_beat = {}
    for i in range(len(rows)):
        beat = {j for j in range(i) if not _indistinguishable(rows[j], rows[i], max_p_gap)[0]}
        if fragile[i]:
            beat |= {j for j in range(i) if not fragile[j]}
        must_beat[i] = beat
    placed: list[int] = []
    remaining = set(range(len(rows)))
    while remaining:
        free = [i for i in remaining if must_beat[i] <= set(placed)]
        nxt = max(free, key=lambda i: rows[i]["hierarchical"])
        placed.append(nxt)
        remaining.remove(nxt)
    order = pd.concat([head.iloc[placed], rest], ignore_index=True)
    order["tie_break_rank"] = np.arange(1, len(order) + 1)
    if classes is not None:
        ok = order["stability"] != "fragile"
        top = order[ok].head(top_n)
        order = pd.concat([top, order.drop(top.index)], ignore_index=True)
    order["final_rank"] = np.arange(1, len(order) + 1)
    order["moved_vs_a"] = order["rank"] - order["final_rank"]
    order["final_reason"] = [_reason_for_move(r, top_n) for r in order.itertuples()]
    checks = []
    for i in range(len(rows) - 1):
        tied, overlap, gap = _indistinguishable(rows[i], rows[i + 1], max_p_gap)
        checks.append({"rank_a": int(rows[i]["rank"]), "player_a": rows[i]["name"], "rank_b": int(rows[i + 1]["rank"]),
                       "player_b": rows[i + 1]["name"], "intervals_overlap": overlap, "p_top_n_gap": gap, "tied": tied})
    return order, pd.DataFrame(checks)


def _reason_for_move(r: Any, top_n: int) -> str:
    """Why a player's final rank differs from their rank in A."""
    if r.final_rank == r.rank:
        return ""
    if r.stability == "fragile" and r.tie_break_rank <= top_n < r.final_rank:
        return "stability guard: fragile, so not allowed in the top 10"
    if r.final_rank <= top_n < r.tie_break_rank:
        return "moved up as fragile players left the top 10"
    if r.tie_break_rank != r.rank:
        side = "above" if r.tie_break_rank < r.rank else "below"
        return f"tie-break: indistinguishable from neighbours in A, ordered by B (B rank {int(r.rank_hierarchical)}); moved {side}"
    return "shifted by moves of other players"


def overlap_summary(rated: pd.DataFrame, top_n: int) -> dict[str, Any]:
    """Top-n membership under A, B and the final (tie-broken) order."""
    a = set(rated.loc[rated["rank"] <= top_n, "name"])
    b = set(rated.loc[rated["rank_hierarchical"] <= top_n, "name"])
    f = set(rated.loc[rated["final_rank"] <= top_n, "name"])
    return {"a_and_b": len(a & b), "only_a": sorted(a - b), "only_b": sorted(b - a), "final_and_a": len(f & a),
            "spearman_a_b": rated[["composite", "hierarchical"]].corr(method="spearman").iloc[0, 1]}


# ---------------------------------------------------------------- pressure-ball records
def pressure_record(balls: pd.DataFrame, player_id: int, role: str, flag: str) -> dict[str, float]:
    """What actually happened on a player's pressure balls, against what the league would expect."""
    scored = balls[balls["state_ok"] & balls[flag]]
    if role == "batter":
        strike = scored[scored["striker_id"] == player_id]          # includes wides (no ball faced, no value)
        rows = strike[strike["is_ball_faced"]]
        runs, expected = rows["bat_runs"].sum(), rows["xr_bat"].sum()
        outs, expected_outs = rows["striker_out"].sum(), rows["xw_bat"].sum()
        adj = (rows["v_bat_adj"] - rows["v_bat"]).mean()
        parts = {"runs": strike["v_bat_runs"].sum(), "wickets": strike["v_bat_wicket"].sum(),
                 "non_striker": scored.loc[scored["non_striker_id"] == player_id, "v_ns"].sum(),
                 "opposition": (strike["v_bat_adj"] - strike["v_bat"]).sum()}
    else:
        rows = scored[scored["bowler_id"] == player_id]
        runs, expected = rows["bowler_runs"].sum(), rows["xr_bowl"].sum()
        outs, expected_outs = rows["bowler_wicket"].sum(), rows["xw_bowl"].sum()
        adj = (rows["v_bowl_adj"] - rows["v_bowl"]).mean()
        parts = {"runs": rows["v_bowl_runs"].sum(), "wickets": rows["v_bowl_wicket"].sum(), "non_striker": 0.0,
                 "opposition": (rows["v_bowl_adj"] - rows["v_bowl"]).sum()}
    n = len(rows)
    boundaries = (rows["is_four"] | rows["is_six"]).sum()
    # The four parts of the raw pressure value (they sum to it): runs above expectation, the wicket term
    # (each dismissal priced at that ball's cost), the non-striker run-out term, and the opposition adjustment.
    breakdown = {f"{k}_part_per_100": v / n * PER_100 if n else np.nan for k, v in parts.items()}
    return {"pressure_balls_record": n, "runs": runs, "expected_runs": expected,
            "runs_v_expected_per_100": (runs - expected) / n * PER_100 if n else np.nan,
            "wickets": outs, "expected_wickets": expected_outs, "boundary_pct": boundaries / n * PER_100 if n else np.nan,
            "mean_wicket_cost": rows["c"].mean(), "opposition_adjustment_per_100": adj * PER_100, **breakdown}


def _reason(role: str, r: dict[str, float], league_boundary_pct: float) -> str:
    """One plain-language sentence from the pressure-ball record."""
    runs = r["runs_v_expected_per_100"]
    wk, xwk = r["wickets"], r["expected_wickets"]
    b = r["boundary_pct"]
    if role == "batter":
        scoring = f"scored {abs(runs):.0f} runs per 100 pressure balls {'more' if runs >= 0 else 'fewer'} than expected"
        outs = f"was dismissed {int(wk)} times against {xwk:.1f} expected"
        hits = f"hit boundaries on {b:.0f}% of balls (pressure-ball average {league_boundary_pct:.0f}%)"
    else:
        scoring = f"conceded {abs(runs):.0f} runs per 100 pressure balls {'more' if runs >= 0 else 'fewer'} than expected"
        outs = f"took {int(wk)} wickets against {xwk:.1f} expected"
        hits = f"conceded boundaries on {b:.0f}% of balls (pressure-ball average {league_boundary_pct:.0f}%)"
    return f"On {int(r['pressure_balls_record'])} pressure balls, {scoring}, {outs}, and {hits}."


def absences(rated: pd.DataFrame, balls: pd.DataFrame, role: str, ids: list[int], overall_top: int,
             top_n: int, flag: str, names: dict[int, str]) -> pd.DataFrame:
    """Named players and overall top players who miss the final top n, with the evidence for why."""
    scored = balls[balls["state_ok"] & balls[flag]]
    if role == "batter":
        faced = scored[scored["is_ball_faced"]]
        league_b = (faced["is_four"] | faced["is_six"]).mean() * PER_100
    else:
        league_b = (scored["is_four"] | scored["is_six"]).mean() * PER_100
    pick = rated[(rated["player_id"].isin(ids) | (rated["overall_rank"] <= overall_top)) & (rated["final_rank"] > top_n)]
    out = []
    for row in pick.itertuples():
        record = pressure_record(balls, int(row.player_id), role, flag)
        out.append({"role": role, "player_id": row.player_id, "name": row.name, "overall_rank": row.overall_rank,
                    "pressure_rank_a": row.rank, "final_rank": row.final_rank, "pressure_balls": row.pressure_balls,
                    "pressure_value_raw": row.pressure_value_raw, "pressure_value_shrunk": row.pressure_value_shrunk,
                    "pressure_value_p05": row.pressure_value_p05, "pressure_value_p95": row.pressure_value_p95,
                    "p_top_n": row.p_top_n, "usual_value_raw": row.usual_value_raw, **record,
                    "reason": _reason(role, record, league_b)})
    for pid in sorted(set(ids) - set(rated["player_id"])):
        out.append({"role": role, "player_id": pid, "name": names.get(pid, str(pid)),
                    "reason": "Not eligible in this role: fewer than 100 pressure balls, or pressure balls in fewer than 10 matches."})
    return pd.DataFrame(out)


def evidence_card(rated: pd.DataFrame, balls: pd.DataFrame, player_id: int, role: str, flag: str) -> pd.DataFrame:
    """Everything that supports one player's rating, on one card."""
    row = rated[rated["player_id"] == player_id].iloc[0]
    record = pressure_record(balls, player_id, role, flag)
    fields = {
        "name": row["name"], "teams": row["teams"], "final_rank": row["final_rank"], "rank_a": row["rank"],
        "rank_b": row["rank_hierarchical"], "matches": row["matches"], "pressure_matches": row["pressure_matches"],
        "pressure_balls": row["pressure_balls"], "pressure_value_raw": row["pressure_value_raw"],
        "pressure_value_shrunk": row["pressure_value_shrunk"], "pressure_value_p05": row["pressure_value_p05"],
        "pressure_value_p95": row["pressure_value_p95"], "p_top_n": row["p_top_n"],
        "p_top_n_hierarchical": row["p_top_n_hierarchical"], "p_top_n_equal_balls": row["p_top_n_equal_balls"],
        "rank_interval": f"{row['rank_p05']:.0f}-{row['rank_p95']:.0f}", "usual_value_raw": row["usual_value_raw"],
        "consistency_shrunk_pts": row["consistency_shrunk"] * PER_100, "share_balls_ipl": row["share_balls_ipl"],
        **record,
    }
    return pd.DataFrame({"field": list(fields), "value": list(fields.values())})


# ---------------------------------------------------------------- sensitivity variants
def _flag_from(values: pd.Series, ok: pd.Series, share: float) -> pd.Series:
    pct = values.where(ok).rank(pct=True)
    return (pct > 1 - share).fillna(False)


def build_variants(balls: pd.DataFrame, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each variant: the ball table to use, the pressure flag, the weighting and the eligibility thresholds."""
    share = config["pressure"]["high_pressure_share"]
    ok = balls["state_ok"]
    flag = config["pressure"]["high_pressure_flag"]
    base = balls.assign(v_bat=balls["v_bat_adj"], v_bowl=balls["v_bowl_adj"])
    li_log = base["lev_logistic"] / base.loc[ok, "lev_logistic"].mean()
    base["hp_logistic"] = _flag_from(li_log, ok, share)
    base["hp_rule"] = _flag_from(base["rule_index"], ok, share)
    elig = config["eligibility"]
    std = (elig["min_pressure_balls"], elig["min_pressure_matches"])
    variants: dict[str, dict[str, Any]] = {
        "headline": {"balls": base, "flag": flag},
        "pressure: logistic leverage": {"balls": base.assign(li=li_log), "flag": "hp_logistic"},
        "pressure: rule index": {"balls": base, "flag": "hp_rule"},
        "pressure: per-innings cut": {"balls": base, "flag": "hp_within_innings"},
        "weight: sqrt LI": {"balls": base, "flag": flag, "weight": "sqrt"},
        "weight: LI cap": {"balls": base, "flag": flag, "weight": "cap"},
        "knockouts x1.25": {"balls": base.assign(weight_mult=np.where(base["knock_out"] == 1,
                                                                    config["robustness"]["knockout_multiplier"], 1.0)), "flag": flag},
        "eligibility 150": {"balls": base, "flag": flag, "eligibility": (150, std[1])},
        "eligibility 200": {"balls": base, "flag": flag, "eligibility": (200, std[1])},
        "IPL only": {"balls": base[base["comp_type"] == "IPL"], "flag": flag},
        "T20I only": {"balls": base[base["comp_type"] == "T20I"], "flag": flag},
        "associates excluded": {"balls": base[base["involves_associate"] == 0], "flag": flag},
    }
    for year in sorted(base["year"].unique()):
        variants[f"without {year}"] = {"balls": base[base["year"] != year], "flag": flag}
    # Role baselines: value relative to the average for the batting position / bowling phase.
    faced = base[ok & base["is_ball_faced"]]
    position_mean = faced.groupby("batting_order")["v_bat"].mean()
    phase_mean = base[ok].groupby("phase")["v_bowl"].mean()
    variants["role: position / phase relative"] = {
        "balls": base.assign(v_bat=base["v_bat"] - base["batting_order"].map(position_mean).fillna(0),
                             v_bowl=base["v_bowl"] - base["phase"].map(phase_mean).fillna(0)), "flag": flag}
    for v in variants.values():
        v.setdefault("eligibility", std)
    return variants


def run_variants(variants: dict[str, dict[str, Any]], config: dict[str, Any], weight: str,
                 weights: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Ranks of every player in every variant (both ratings), long format."""
    out = []
    for label, v in variants.items():
        for role in ("batter", "bowler"):
            rows = player_balls(v["balls"], role, v["flag"])
            scores = point_scores(rows, config, v.get("weight", weight), weights[role], *v["eligibility"])
            out.append(scores.assign(variant=label, role=role))
        logger.info("Phase 6 variant done: %s", label)
    return pd.concat(out, ignore_index=True)


VARIANT_GROUPS: dict[str, list[str]] = {
    "headline": ["headline"],
    "definition": ["pressure: logistic leverage", "pressure: per-innings cut"],
    "weighting": ["weight: sqrt LI", "weight: LI cap", "knockouts x1.25"],
    "eligibility": ["eligibility 150", "eligibility 200"],
    "competition": ["IPL only", "T20I only", "associates excluded"],
    "seasons": [],                                     # filled with the leave-one-season-out variants
    "role": ["role: position / phase relative"],
    "alternatives": ["pressure: rule index", "option B (hierarchical)"],
}


def variant_groups(variants: list[str]) -> dict[str, list[str]]:
    """The column groups, with the season variants found in ``variants``."""
    groups = {k: list(v) for k, v in VARIANT_GROUPS.items()}
    groups["seasons"] = sorted(v for v in variants if v.startswith("without "))
    return groups


def counted_variants(groups: dict[str, list[str]]) -> list[str]:
    """The perturbation variants that count towards stability (not the headline, not the alternatives)."""
    return [v for k, cols in groups.items() if k not in ("headline", "alternatives") for v in cols]


def wide_ranks(ranks: pd.DataFrame) -> pd.DataFrame:
    """Player x variant table of ranks, with option B taken from the headline run."""
    wide = ranks.pivot_table(index="player_id", columns="variant", values="rank")
    wide["option B (hierarchical)"] = ranks[ranks["variant"] == "headline"].set_index("player_id")["rank_hierarchical"]
    return wide


def stability_classes(ranks: pd.DataFrame, counted: list[str], top_n: int, cuts: tuple[float, float]) -> pd.DataFrame:
    """For every player: share of the counted variants (where eligible) with rank <= top_n, and the class."""
    wide = wide_ranks(ranks)
    cols = [c for c in counted if c in wide.columns]
    present = wide[cols].notna().sum(axis=1)
    share = (wide[cols] <= top_n).sum(axis=1) / present.replace(0, np.nan)
    out = pd.DataFrame({"variants_eligible": present, "share_in_top_n": share,
                        "worst_rank": wide[cols].max(axis=1), "median_rank": wide[cols].median(axis=1)})
    out["stability"] = np.select([out["share_in_top_n"] >= cuts[0], out["share_in_top_n"] >= cuts[1]],
                                 ["stable", "moderately stable"], default="fragile")
    return out


def stability_table(final: pd.DataFrame, ranks: pd.DataFrame, classes: pd.DataFrame, zone: int,
                    groups: dict[str, list[str]]) -> pd.DataFrame:
    """The final top ``zone`` with their rank in every variant (grouped), share in the top 10 and class."""
    top = final[final["final_rank"] <= zone][["player_id", "name", "final_rank", "rank", "rank_hierarchical", "p_top_n",
                                              "composite_p05", "composite_p95", "pressure_value_shrunk",
                                              "pressure_value_p05", "pressure_value_p95", "final_reason"]]
    wide = wide_ranks(ranks)
    cols = [c for cols in groups.values() for c in cols if c in wide.columns]
    return top.join(wide[cols], on="player_id").join(classes, on="player_id").sort_values("final_rank")
