"""Phase 2b evidence tables: does leverage separate pressure from ordinary play?"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.pressure import outcome_summary, share_table
from src.models.evaluate import metric_row, reliability_table

FLAGS: tuple[str, str] = ("hp_common", "hp_within_innings")


def _slices(clean: pd.DataFrame, config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Row masks for the slices every candidate is judged on."""
    chase = clean["innings"].eq(2).to_numpy()
    left = clean["balls_remaining"].to_numpy()
    late = config["models"]["late_chase_balls"]
    masks = {"all": np.ones(len(clean), bool), "inn1": ~chase, "inn2": chase}
    for n in late:
        masks[f"chase_last{n}"] = chase & (left <= n)
    return masks


def _ball_log_loss(p: np.ndarray, y: np.ndarray, clip: float) -> np.ndarray:
    q = np.clip(p, clip, 1 - clip)
    return -(y * np.log(q) + (1 - y) * np.log(1 - q))


def bootstrap_vs_reference(
    clean: pd.DataFrame, candidates: tuple[str, ...], reference: str, config: dict[str, Any]
) -> pd.DataFrame:
    """Log-loss of each candidate minus the reference, with a 95% interval from resampling matches."""
    clip, reps = config["models"]["prob_clip"], config["models"]["bootstrap_reps"]
    y = clean["batting_team_won"].to_numpy()
    losses = {name: _ball_log_loss(clean[f"wp_{name}"].to_numpy(), y, clip) for name in candidates}
    rng = np.random.default_rng(config["seed"])
    rows = []
    for label, mask in _slices(clean, config).items():
        part = clean.loc[mask, ["match_id"]].copy()
        for name in candidates:
            part[name] = losses[name][mask]
        per_match = part.groupby("match_id").agg(["sum", "size"])
        counts = per_match[(reference, "size")].to_numpy()
        draws = rng.integers(0, len(per_match), (reps, len(per_match)))
        ref_sum = per_match[(reference, "sum")].to_numpy()
        for name in candidates:
            if name == reference:
                continue
            diff = (per_match[(name, "sum")].to_numpy() - ref_sum)[draws].sum(axis=1) / counts[draws].sum(axis=1)
            rows.append({"slice": label, "model": name, "reference": reference,
                         "log_loss_minus_reference": float((losses[name][mask] - losses[reference][mask]).mean()),
                         "ci_low": float(np.quantile(diff, 0.025)), "ci_high": float(np.quantile(diff, 0.975))})
    return pd.DataFrame(rows)


def model_comparison(scores: pd.DataFrame, candidates: tuple[str, ...], config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Held-out log-loss, Brier and calibration for every candidate: by slice, by season, plus bootstrap gaps."""
    clip, bins = config["models"]["prob_clip"], config["models"]["calibration_bins"]
    clean = scores[scores["wp_train_row"]]
    y_all = clean["batting_team_won"].to_numpy()
    masks = _slices(clean, config)
    overall, seasons, reliab = [], [], []
    for name in candidates:
        p_all = clean[f"wp_{name}"].to_numpy()
        row: dict[str, Any] = {"model": name}
        for label, mask in masks.items():
            for key, value in metric_row(p_all[mask], y_all[mask], clip, bins).items():
                row[f"{key}_{label}"] = value
        overall.append(row)
        for year, part in clean.groupby("year"):
            seasons.append({"model": name, "held_out_season": year,
                            **metric_row(part[f"wp_{name}"].to_numpy(), part["batting_team_won"].to_numpy(), clip, bins)})
        for innings in (1, 2):
            part = clean[clean["innings"] == innings]
            table = reliability_table(part[f"wp_{name}"].to_numpy(), part["batting_team_won"].to_numpy(), bins)
            reliab.append(table.assign(model=name, innings=innings))
    return {
        "overall": pd.DataFrame(overall), "by_season": pd.DataFrame(seasons), "reliability": pd.concat(reliab),
        "bootstrap": bootstrap_vs_reference(clean, candidates, config["pressure"]["wp_model"], config),
    }


def leverage_agreement(scores: pd.DataFrame, candidates: tuple[str, ...], share: float) -> pd.DataFrame:
    """Rank correlation of leverage between candidates, and overlap of their top-share sets with the reference."""
    scored = scores[scores["state_ok"]]
    lev = scored[[f"lev_{name}" for name in candidates]].rename(columns=lambda c: c.removeprefix("lev_"))
    corr = lev.corr(method="spearman")
    top = lev.rank(pct=True) > 1 - share
    rows = []
    for a in candidates:
        for b in candidates:
            if a < b:
                rows.append({"model_a": a, "model_b": b, "spearman": corr.loc[a, b],
                             "top_set_overlap": float((top[a] & top[b]).sum() / top[a].sum())})
    return pd.DataFrame(rows)


def last_ball_table(scores: pd.DataFrame, candidates: tuple[str, ...]) -> pd.DataFrame:
    """Final ball of a chase by runs still needed: what happened v what each candidate said."""
    clean = scores[scores["wp_train_row"] & (scores["innings"] == 2) & (scores["balls_remaining"] == 1)]
    need = clean["runs_required"].clip(upper=7).astype(int)
    table = clean.groupby(need).agg(balls=("batting_team_won", "size"), chasing_side_won=("batting_team_won", "mean"),
                                    **{name: (f"wp_{name}", "mean") for name in candidates})
    out = table.reset_index().rename(columns={"runs_required": "runs_needed"})
    out["runs_needed"] = out["runs_needed"].astype(str).replace({"7": "7 or more"})
    return out


def li_summary(balls: pd.DataFrame) -> pd.DataFrame:
    """Quantiles of LI overall and by innings, plus each group's share above the common-scale cut."""
    scored = balls[balls["state_ok"]]
    rows = []
    for label, part in (("all", scored), ("first innings", scored[scored["innings"] == 1]), ("chase", scored[scored["innings"] == 2])):
        q = part["li"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        rows.append({"group": label, "balls": len(part), "mean": part["li"].mean(), **{f"p{int(k * 100)}": v for k, v in q.items()},
                     "max": part["li"].max()})
    return pd.DataFrame(rows)


def heatmap_grid(balls: pd.DataFrame) -> pd.DataFrame:
    """Mean LI and ball count by innings, over and wickets already down."""
    scored = balls[balls["state_ok"]]
    grid = scored.groupby(["innings", "overid", "wickets_before"])["li"].agg(mean_li="mean", balls="size")
    return grid.reset_index()


def separation(balls: pd.DataFrame, flag: str) -> pd.DataFrame:
    """Outcome profile of high-pressure v ordinary balls, overall and within each bowling phase."""
    scored = balls[balls["state_ok"]].assign(high_pressure=lambda d: d[flag])
    overall = outcome_summary(scored, ["high_pressure"]).assign(phase="all")
    by_phase = outcome_summary(scored, ["phase", "high_pressure"])
    return pd.concat([overall, by_phase], ignore_index=True).assign(flag=flag)


def concentration(balls: pd.DataFrame) -> pd.DataFrame:
    """Share of all realised win-probability movement that happens on high-pressure balls."""
    scored = balls[balls["state_ok"] & balls["abs_wp_change"].notna()]
    total = scored["abs_wp_change"].sum()
    rows = [{"flag": flag, "share_of_balls": scored[flag].mean(),
             "share_of_wp_movement": scored.loc[scored[flag], "abs_wp_change"].sum() / total} for flag in FLAGS]
    return pd.DataFrame(rows)


def flag_contrast(balls: pd.DataFrame) -> pd.DataFrame:
    """How sharply each flag separates stakes within each innings.

    For every flag and innings: mean LI and mean realised WP swing of flagged v unflagged
    balls, and the innings' share of the total LI weight carried by the flagged set.
    """
    scored = balls[balls["state_ok"]]
    rows = []
    for flag in FLAGS:
        total_weight = scored.loc[scored[flag], "li"].sum()
        for innings, part in scored.groupby("innings"):
            hit, miss = part[part[flag]], part[~part[flag]]
            rows.append({
                "flag": flag, "innings": innings, "flagged_balls": len(hit), "li_cut": hit["li"].min(),
                "mean_li_flagged": hit["li"].mean(), "mean_li_unflagged": miss["li"].mean(),
                "li_ratio": hit["li"].mean() / miss["li"].mean(),
                "swing_flagged_pct": hit["abs_wp_change"].mean() * 100, "swing_unflagged_pct": miss["abs_wp_change"].mean() * 100,
                "share_of_flagged_li_weight": hit["li"].sum() / total_weight,
                "matches_with_flagged_balls": hit["match_id"].nunique(),
            })
    return pd.DataFrame(rows)


def innings_options(balls: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """How the two ways of drawing the high-pressure line distribute pressure balls."""
    scored = balls[balls["state_ok"]]
    return {
        "by_innings": share_table(scored, "innings", FLAGS),
        "by_batting_position": share_table(scored[scored["batting_order"] <= 11], "batting_order", FLAGS),
        "by_phase": share_table(scored, "phase", FLAGS).set_index("phase").loc[["powerplay", "middle", "death"]].reset_index(),
        "by_innings_phase": share_table(scored.assign(innings_phase=scored["innings"].astype(str) + "_" + scored["phase"]),
                                        "innings_phase", FLAGS),
    }


def phase_exposure(balls: pd.DataFrame, flag: str) -> pd.DataFrame:
    """High-pressure balls by innings and bowling phase: exposure rate in each cell and share of the pressure set."""
    scored = balls[balls["state_ok"]]
    cells = scored.groupby(["innings", "phase"])[flag].agg(balls="size", hp_balls="sum").reset_index()
    both = scored.groupby("phase")[flag].agg(balls="size", hp_balls="sum").reset_index().assign(innings="both")
    out = pd.concat([cells, both], ignore_index=True)
    out["hp_rate"] = out["hp_balls"] / out["balls"]
    out["share_of_hp_balls"] = out["hp_balls"] / scored[flag].sum()
    order = {"powerplay": 0, "middle": 1, "death": 2}
    return out.sort_values(["innings", "phase"], key=lambda s: s.map(order) if s.name == "phase" else s.astype(str))


def player_exposure(balls: pd.DataFrame, role: str, flag: str, min_balls: int) -> pd.DataFrame:
    """Per player: share of their balls that are high-pressure, and their mix of innings and phases.

    ``role`` is "batter" (balls faced as striker) or "bowler" (every delivery bowled).
    """
    scored = balls[balls["state_ok"]]
    key, name = ("striker_id", "striker_name") if role == "batter" else ("bowler_id", "bowler_name")
    if role == "batter":
        scored = scored[scored["is_ball_faced"]]
    d = scored.assign(chase=scored["innings"].eq(2), hp=scored[flag].astype(float))
    g = d.groupby(key)
    out = pd.DataFrame({"name": g[name].first(), "balls": g.size(), "hp_balls": g["hp"].sum().astype(int),
                        "hp_share": g["hp"].mean(), "chase_share": g["chase"].mean()})
    out["hp_rate_first_innings"] = d[~d["chase"]].groupby(key)["hp"].mean()
    out["hp_rate_chase"] = d[d["chase"]].groupby(key)["hp"].mean()
    for phase in ("powerplay", "middle", "death"):
        out[f"{phase}_share"] = d["phase"].eq(phase).groupby(d[key]).mean()
    out = out[out["balls"] >= min_balls].reset_index().rename(columns={key: "player_id"})
    return out.assign(role=role).sort_values("hp_share", ascending=False)


def _r_squared(y: np.ndarray, x: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(y)), x])
    resid = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return float(1 - resid.var() / y.var())


def exposure_dependence(players: pd.DataFrame) -> pd.DataFrame:
    """How much of the spread in players' high-pressure share is explained by innings mix and phase mix."""
    rows = []
    for role, part in players.groupby("role"):
        y = part["hp_share"].to_numpy()
        chase = part[["chase_share"]].to_numpy()
        slope = float(np.polyfit(part["chase_share"], y, 1)[0])
        q_hp = part["hp_share"].quantile([0.1, 0.5, 0.9]).to_numpy()
        q_chase = part["chase_share"].quantile([0.1, 0.5, 0.9]).to_numpy()
        rows.append({
            "role": role, "players": len(part),
            "hp_share_p10": q_hp[0], "hp_share_median": q_hp[1], "hp_share_p90": q_hp[2],
            "chase_share_p10": q_chase[0], "chase_share_median": q_chase[1], "chase_share_p90": q_chase[2],
            "median_hp_rate_first_innings": part["hp_rate_first_innings"].median(),
            "median_hp_rate_chase": part["hp_rate_chase"].median(),
            "hp_share_change_per_10pt_chase_share": slope * 0.1,
            "r2_chase_share": _r_squared(y, chase),
            "r2_phase_mix": _r_squared(y, part[["powerplay_share", "death_share"]].to_numpy()),
            "r2_chase_share_and_phase_mix": _r_squared(y, part[["chase_share", "powerplay_share", "death_share"]].to_numpy()),
        })
    return pd.DataFrame(rows)


def team_exposure(balls: pd.DataFrame, flag: str, min_matches: int) -> pd.DataFrame:
    """Per team and competition type: how often it chases, and its high-pressure share with bat and ball."""
    scored = balls[balls["state_ok"]]
    name_bat = np.where(scored["batting_team_id"] == scored["home_team_id"], scored["home_team"], scored["away_team"])
    name_bowl = np.where(scored["bowling_team_id"] == scored["home_team_id"], scored["home_team"], scored["away_team"])
    d = scored.assign(team_bat=name_bat, team_bowl=name_bowl, chase=scored["innings"].eq(2))
    bat = d.groupby(["comp_type", "team_bat"]).agg(matches=("match_id", "nunique"), batting_hp_share=(flag, "mean"))
    chasing = d[d["chase"]].groupby(["comp_type", "team_bat"])["match_id"].nunique()
    bat["share_of_matches_chasing"] = (chasing / bat["matches"]).fillna(0.0)
    bowl = d.groupby(["comp_type", "team_bowl"]).agg(bowling_hp_share=(flag, "mean"))
    bat.index.names = bowl.index.names = ["comp_type", "team"]
    out = bat.join(bowl).reset_index()
    return out[out["matches"] >= min_matches].sort_values(["comp_type", "share_of_matches_chasing"], ascending=[True, False])


def knockout_check(balls: pd.DataFrame, probs: np.ndarray, config: dict[str, Any]) -> pd.DataFrame:
    """Do knockout balls behave differently from league balls in the same state?

    Compares actual runs and wickets per ball with what the league-average outcome
    model expects for that state. Standard errors are across matches.
    """
    classes = config["outcome_classes"]
    runs = np.array([c[1] for c in classes])
    wicket = np.array([c[3] for c in classes])
    scored = balls[balls["state_ok"]]
    p = probs[scored.index]
    frame = pd.DataFrame({
        "match_id": scored["match_id"].to_numpy(), "knock_out": scored["knock_out"].to_numpy(),
        "runs_diff": scored["total_runs"].to_numpy() - p @ runs,
        "wicket_diff": scored["is_wicket"].to_numpy().astype(float) - p @ wicket,
        "li": scored["li"].to_numpy(),
    })
    per_match = frame.groupby(["knock_out", "match_id"]).agg(runs_diff=("runs_diff", "mean"), wicket_diff=("wicket_diff", "mean"),
                                                             li=("li", "mean"), balls=("li", "size")).reset_index()
    rows = []
    for flag, part in per_match.groupby("knock_out"):
        n = len(part)
        rows.append({
            "knockout": bool(flag), "matches": n, "balls": int(part["balls"].sum()),
            "runs_per_ball_vs_expected": part["runs_diff"].mean(), "runs_se": part["runs_diff"].std() / np.sqrt(n),
            "wickets_per_ball_vs_expected": part["wicket_diff"].mean(), "wickets_se": part["wicket_diff"].std() / np.sqrt(n),
            "mean_li": part["li"].mean(), "li_se": part["li"].std() / np.sqrt(n),
        })
    return pd.DataFrame(rows)


def rule_agreement(balls: pd.DataFrame, share: float) -> dict[str, pd.DataFrame]:
    """How the rule-based index and leverage agree, and what kind of ball they disagree on."""
    scored = balls[balls["state_ok"]].copy()
    scored["hp_rule"] = scored["rule_index"].rank(pct=True) > 1 - share
    scored["agreement"] = np.select(
        [scored["hp_common"] & scored["hp_rule"], scored["hp_common"] & ~scored["hp_rule"], ~scored["hp_common"] & scored["hp_rule"]],
        ["both", "leverage only", "rule only"], default="neither",
    )
    dist = np.abs(scored["wp"] - 0.5)
    profile = scored.assign(wp_distance_from_even=dist).groupby("agreement").agg(
        balls=("li", "size"), mean_li=("li", "mean"), mean_rule_index=("rule_index", "mean"),
        mean_wp_distance_from_even=("wp_distance_from_even", "mean"), share_chase=("innings", lambda s: (s == 2).mean()),
        mean_over=("overid", "mean"), run_rate=("total_runs", lambda s: s.mean() * 6), dismissal_pct=("is_wicket", lambda s: s.mean() * 100),
    ).reset_index()
    summary = pd.DataFrame([{
        "spearman": scored[["li", "rule_index"]].corr(method="spearman").iloc[0, 1],
        "spearman_first_innings": scored.loc[scored["innings"] == 1, ["li", "rule_index"]].corr(method="spearman").iloc[0, 1],
        "spearman_chase": scored.loc[scored["innings"] == 2, ["li", "rule_index"]].corr(method="spearman").iloc[0, 1],
        "share_of_leverage_hp_also_rule_hp": (scored["hp_common"] & scored["hp_rule"]).sum() / scored["hp_common"].sum(),
    }])
    return {"summary": summary, "profile": profile}


def leverage_drivers(balls: pd.DataFrame) -> pd.DataFrame:
    """Mean LI by how close the contest is and how late it is: the plain-language reading of leverage."""
    scored = balls[balls["state_ok"]].copy()
    gap = np.abs(scored["wp"] - 0.5)
    labels = ["even (35-65%)", "leaning (15-35% or 65-85%)", "one-sided (under 15% or over 85%)"]
    scored["contest"] = pd.cut(gap, [-0.01, 0.15, 0.35, 0.51], labels=labels)
    return scored.groupby(["innings", "contest", "phase"], observed=True)["li"].agg(mean_li="mean", balls="size").reset_index()


def pick_close_finishes(balls: pd.DataFrame, match_table: pd.DataFrame) -> pd.DataFrame:
    """Choose four close finishes by rule, not reputation.

    The pool is every decided match with a known target (rain matches with a revised
    target included). Tightness is the run margin for a successful defence and the balls
    left for a successful chase. One match is taken from each of: knockout, T20I chase,
    IPL chase, successful defence - the tightest first, then the most recent.
    """
    ok = match_table["has_winner"] & ~match_table["level_after_regulation"] & ~match_table["target_unknown"]
    m = match_table[ok & match_table["total2"].notna()].copy()
    m["defended"] = m["total2"] < m["target"] - 1
    m["run_margin"] = (m["target"] - 1 - m["total2"]).where(m["defended"])
    m["balls_left"] = (m["scheduled2"] - m["legal2"]).where(~m["defended"])
    m["tightness"] = m["run_margin"].fillna(m["balls_left"])
    close = m.sort_values(["tightness", "match_date"], ascending=[True, False])
    picks: list[pd.Series] = []
    taken: set[int] = set()
    rules = [
        ("Knockout", close["knock_out"].eq(1)),
        ("T20I chase", close["comp_type"].eq("T20I") & ~close["defended"]),
        ("IPL chase", close["comp_type"].eq("IPL") & ~close["defended"]),
        ("Successful defence", close["defended"]),
    ]
    for label, mask in rules:
        pool = close[mask & ~close["match_id"].isin(taken)]
        if not pool.empty:
            row = pool.iloc[0].copy()
            row["category"] = label
            picks.append(row)
            taken.add(int(row["match_id"]))
    return pd.DataFrame(picks)


def match_curve(balls: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """Ball-by-ball WP (from the chasing side's view) and LI for one match."""
    d = balls[(balls["match_id"] == match_id) & balls["state_ok"]].sort_values(["innings", "innings_ball_num"]).copy()
    d["wp_chasing"] = np.where(d["innings"] == 1, 1 - d["wp"], d["wp"])
    cols = ["match_id", "innings", "innings_ball_num", "overid", "striker_name", "bowler_name", "score_before",
            "wickets_before", "balls_remaining", "runs_required", "total_runs", "is_wicket", "wp_chasing", "li"]
    return d[cols].reset_index(drop=True)
