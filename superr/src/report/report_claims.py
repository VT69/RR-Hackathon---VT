"""The checked claims of the methodology report (see ``check_report`` for how they are checked).

Each claim is a phrase copied from the report with its numbers replaced by ``{}`` placeholders, and the
values recomputed here from the output tables or ``config.yaml``. If the report is reworded, the phrase
here must be updated too (the check reports it as MISSING rather than passing silently).
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from typing import Any

import pandas as pd

from src.report.check_report import Claim, Tables, fmt_num, fmt_signed
from src.robustness.sensitivity import counted_variants, variant_groups

WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}


def pct(x: float, digits: int = 0) -> str:
    """0.479 -> '47.9%' style (digits after the point)."""
    return f"{x * 100:.{digits}f}%"


def f2(x: float) -> str:
    """Two decimals, with -0.00 shown as 0.00 and a true minus sign otherwise."""
    s = f"{x:.2f}"
    return "0.00" if s == "-0.00" else s.replace("-", "−")


def ordinal(n: int) -> str:
    """1 -> 1st, 2 -> 2nd, 11 -> 11th."""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def log_minutes(path) -> float:
    """Minutes between the first and last timestamped line of a pipeline log."""
    lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line[:4].isdigit()]
    stamp = lambda line: datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")  # noqa: E731
    return (stamp(lines[-1]) - stamp(lines[0])).total_seconds() / 60


def build(T: Tables, config: dict[str, Any]) -> list[Claim]:
    """Return every checked claim."""
    out: list[Claim] = []

    def C(section: str, template: str, *values: Any) -> None:
        out.append(Claim(section, template, tuple(values)))

    # ------------------------------------------------------------------ shared values
    li_sum = T("phase2b/li_summary.csv").set_index("group")
    cut = float(T("phase2b/li_cuts.csv").query("flag == 'hp_common'")["li_cut"].iloc[0])
    fc = T("phase2b/flag_contrast.csv").query("flag == 'hp_common'").set_index("innings")
    share_inn2 = fc.loc[2, "flagged_balls"] / fc["flagged_balls"].sum()
    sep = T("phase2b/separation.csv").query("flag == 'hp_common' and phase == 'all'").set_index("high_pressure")
    rule = T("phase2b/rule_vs_leverage_summary.csv").iloc[0]
    prof = T("phase2b/rule_vs_leverage_profile.csv").set_index("agreement")
    sit = T("phase2b/situations.csv").set_index("situation")
    rel = T("phase5/split_half_reliability.csv").query("weighting == 'uniform'").set_index(["role", "component"])
    final = T("final_rankings_top15.csv")
    elig_final = T("final_rankings_eligible.csv")
    variants = T("phase6/variant_ranks.csv")
    groups = variant_groups(variants["variant"].unique().tolist())
    n_counted = len(counted_variants(groups))
    ovp = pd.concat([T("phase5/overall_v_pressure_batters.csv").assign(role="batter"),
                     T("phase5/overall_v_pressure_bowlers.csv").assign(role="bowler")])
    rob, rcfg = config["robustness"], config["rating"]

    # ------------------------------------------------------------------ 0. executive summary
    s = "0 summary"
    C(s, '{} of high-pressure balls are bowled in the second innings', pct(share_inn2))
    C(s, 'the top {:.0f}% of balls by expected win-probability swing are "high pressure" (Leverage Index above {:.2f})',
      config["pressure"]["high_pressure_share"] * 100, cut)
    assert rob["stability_cuts"][1] == 0.5        # "at least half" = the moderately-stable bar
    C(s, "under at least half of {} alternative choices", n_counted)
    C(s, "shares only {} of its high-pressure balls with leverage", pct(rule.share_of_leverage_hp_also_rule_hp))
    C(s, "(reliability {} batters, {} bowlers)", f2(rel.loc[("batter", "pressure"), "random_mean"]),
      f2(rel.loc[("bowler", "pressure"), "random_mean"]))
    C(s, "the gap between pressure and usual level does not ({} and {})",
      f2(rel.loc[("batter", "uplift"), "random_mean"]), f2(rel.loc[("bowler", "uplift"), "random_mean"]))
    drops = ovp[ovp["group"].str.startswith("drops")].groupby("role").size()
    C("4 overall v pressure", "{} batters and {} bowlers are not in the pressure top 10", int(drops["batter"]), int(drops["bowler"]))
    bat = final[(final["role"] == "batter") & (final["rank"] <= rcfg["top_n"])].sort_values("rank")
    bowl = final[(final["role"] == "bowler") & (final["rank"] <= rcfg["top_n"])].sort_values("rank")
    for a, b in zip(bat.itertuples(), bowl.itertuples()):
        C("0 summary table", "{}", f"| {a.rank} | {a.player_name} | {b.player_name} |")

    # ------------------------------------------------------------------ 1. pressure
    s = "1 pressure"
    C(s, "**LI above {:.2f}**", cut)
    C(s, "Of {:,} scored balls, {:,} are high pressure", int(li_sum.loc["all", "balls"]), int(sep.loc[True, "balls"]))
    for name in ["Chase: need 60 off 12 balls, 8 down", "Chase: need 9 off 6 balls, 5 down",
                 "Chase: need 20 off 30 balls, 9 down", "First innings: 25 for 4 after 5 overs"]:
        C(s, "| " + name + " | {} | {:.2f} |", pct(sit.loc[name, "win_probability"], 1), sit.loc[name, "leverage_index"])
    C(s, "Rank correlation over all balls is {:.2f}, and only **{}**", rule.spearman, pct(rule.share_of_leverage_hp_also_rule_hp))
    C(s, "on average their WP is {:.0f} points from even, against {:.0f} for balls that only leverage flags",
      prof.loc["rule only", "mean_wp_distance_from_even"] * 100, prof.loc["leverage only", "mean_wp_distance_from_even"] * 100)
    wp = T("phase2b/wp_model_comparison.csv").set_index("model")
    C(s, "({:,} held-out balls from", int(wp.loc["dp_platt", "n_balls_all"]))
    rows = {"logistic": "| Logistic regression | **{:.4f}** | {:.4f} | **{:.4f}** | **{:.4f}** |",
            "xgboost": "| XGBoost (monotone) | {:.4f} | {:.4f} | {:.4f} | {:.4f} |",
            "lightgbm": "| LightGBM (monotone) | {:.4f} | {:.4f} | {:.4f} | {:.4f} |",
            "dp": "| Dynamic programming (DP) | {:.4f} | **{:.4f}** | {:.4f} | {:.4f} |",
            "dp_platt": "| **DP + recalibration (chosen)** | {:.4f} | {:.4f} | {:.4f} | {:.4f} |"}
    for model, template in rows.items():
        r = wp.loc[model]
        C(s, template, r.log_loss_all, r.log_loss_chase_last12, r.ece_all, r.roughness_inn2)
    gap = T("phase2b/wp_model_bootstrap_gaps.csv").query("slice == 'all' and model == 'logistic'").iloc[0]
    C(s, "logistic is better by {:.4f}, 95% interval for the difference {} to {}", -gap.log_loss_minus_reference,
      fmt_signed(gap.ci_low, 4).replace("+", ""), fmt_signed(gap.ci_high, 4))
    last = T("phase2b/wp_last_ball_of_chase.csv").set_index("runs_needed").loc["1"]
    C(s, "the chasing side won all {} times in the held-out data; the DP said {}, logistic regression {}",
      int(last.balls), pct(last.dp_platt), pct(last.logistic))
    assert float(last.chasing_side_won) == 1.0
    C(s, "high-pressure balls go for {:.2f} runs per over against {:.2f}, with {:.2f} dismissals per 100 balls against {:.2f} "
         "and boundaries on {:.1f}% against {:.1f}%", sep.loc[True, "run_rate"], sep.loc[False, "run_rate"],
      sep.loc[True, "dismissal_pct"], sep.loc[False, "dismissal_pct"], sep.loc[True, "boundary_pct"], sep.loc[False, "boundary_pct"])
    C(s, "the average realised swing is {:.2f}% per ball against {:.2f}%", sep.loc[True, "mean_abs_wp_swing_pct"],
      sep.loc[False, "mean_abs_wp_swing_pct"])
    conc = T("phase2b/concentration.csv").set_index("flag").loc["hp_common"]
    C(s, "a quarter of the balls ({}) carry **{}**", pct(conc.share_of_balls, 1), pct(conc.share_of_wp_movement, 1))
    pos = T("phase2b/flag_options_by_batting_position.csv").set_index("batting_order").loc[1:8, "hp_common_rate"]
    C(s, "Batting positions 1 to 8 each have {:.0f}-{:.0f}% of their balls flagged", pos.min() * 100, pos.max() * 100)
    C(s, "{} of high-pressure balls are in the second innings", pct(share_inn2))
    C(s, "{} of chase balls qualify, against {} of first-innings balls",
      pct(fc.loc[2, "flagged_balls"] / li_sum.loc["chase", "balls"], 1),
      pct(fc.loc[1, "flagged_balls"] / li_sum.loc["first innings", "balls"], 1))
    ex = T("phase2b/exposure_by_innings_phase.csv")
    ex = ex[ex["innings"].astype(str) == "both"].set_index("phase")
    C(s, "The death overs supply only {} of high-pressure balls; the middle overs supply {}",
      pct(ex.loc["death", "share_of_hp_balls"], 1), pct(ex.loc["middle", "share_of_hp_balls"], 1))
    km = T("phase2b/knockout_check.csv")
    C("appendix A", "distinct values across {} matches)", int(km["matches"].sum()))

    # ------------------------------------------------------------------ 2. performance
    s = "2 performance"
    es = T("phase3/expectation_scores.csv").set_index("expectation")
    C(s, "a squared error of {:.3f} against {:.3f} for runs off the bat", es.loc["xr_bat", "full_model"],
      es.loc["xr_bat", "season_phase_average"])
    rg = T("phase3/raw_gap_v_value_gap.csv").query("role == 'batter'").set_index("pressure")
    C(s, "Raw strike rate is {:.1f} on ordinary balls and {:.1f} on high-pressure balls; run value is {} and {} runs per 100 balls",
      rg.loc[False, "runs_per_100"], rg.loc[True, "runs_per_100"], fmt_signed(rg.loc[False, "value_per_100"]),
      fmt_signed(rg.loc[True, "value_per_100"]))
    wc = T("phase3/wicket_cost_examples.csv").set_index("state")
    states = {
        "Chase, early: need 180 off 114, 10 in hand": "Chase, early: need 180 off 114, 10 in hand",
        "Chase, middle: need 90 off 60, 7 in hand": "Chase, middle: need 90 off 60, 7 in hand",
        "Chase, last 2 overs: need 20 off 12, 6 in hand": "Chase, last 2 overs: need 20 off 12, 6 in hand",
        "Chase, last 2 overs: need 20 off 12, last pair": "Chase, last 2 overs: need 20 off 12, last pair",
        "Chase, last ball: need 3 off 1": "Chase, last ball: need 3 off 1, 5 in hand",
        "First innings: 50 for 5 after 8 overs": "First innings: 50 for 5 after 8 overs",
        "First innings: 150 for 8 after 17 overs": "First innings: 150 for 8 after 17 overs",
    }
    for shown, key in states.items():
        C(s, "| " + shown + " | {} | {:.1f} |", pct(wc.loc[key, "win_probability"]), wc.loc[key, "wicket_cost_runs"])
    summ = T("phase3/wicket_cost_summary.csv").set_index("group").loc["all"]
    C(s, "is used instead ({} of balls)", pct(summ.fallback_share, 1))
    ev = T("phase3/wicket_cost_event_study.csv").set_index("overs_left")
    C(s, "it was {:.1f} runs too high with 16-20 overs left and {:.1f} runs too low with 1-2 left",
      ev.loc["16-20", "gap_before"], -ev.loc["1-2", "gap_before"])
    C(s, "across {:,} wickets, {:.2f} runs actually lost against {:.2f} from the calibrated model",
      int(ev.loc["all", "wickets"]), ev.loc["all", "runs_actually_lost"], ev.loc["all", "cost_after"])
    C(s, "**Capped** at {:.1f} runs (the {}th percentile); the average cost is {:.1f} runs",
      float(T("phase3/wicket_cost_cap.csv")["cap_after_calibration"].iloc[0]),
      round(config["value"]["cost_cap_quantile"] * 100), summ.mean_cost)
    wch = T("phase5/weighting_choice.csv").set_index("weighting")
    C(s, "(reliability {:.3f}, {:.3f} and {:.3f}, within one standard deviation)", wch.loc["uniform", "pressure_reliability_mean"],
      wch.loc["cap", "pressure_reliability_mean"], wch.loc["sqrt", "pressure_reliability_mean"])
    assert wch.loc["uniform", "chosen"]

    # ------------------------------------------------------------------ 3. fairness
    s = "3 fairness"
    el = T("phase4/eligibility.csv")
    n_el = el[el["status"] == "eligible"].groupby("role").size()
    C(s, "**at least {} high-pressure balls in at least {} matches** in that role: **{} batters and {} bowlers**",
      config["eligibility"]["min_pressure_balls"], config["eligibility"]["min_pressure_matches"],
      int(n_el["batter"]), int(n_el["bowler"]))
    cv = T("phase4/opposition_cv_batter.csv")
    chosen = cv[cv["penalty"] == cv["chosen_penalty"]].iloc[0]
    C(s, "(penalty {:,.0f} balls)", chosen.chosen_penalty)
    C(s, "(mean squared error {:.4f} against {:.4f})", chosen.held_out_mse,
      cv[cv["model"] == "intercept only"]["held_out_mse"].iloc[0])
    g = T("phase4/same_player_gaps.csv")
    gl = lambda role, a, b: g[(g["role"] == role) & (g["against"] == a) & (g["minus_against"] == b)].iloc[0]  # noqa: E731
    r = gl("batter", "T20I top-8", "IPL")
    C(s, "| {} batters: v T20I top-8 minus v IPL | {} | {} | {} |", r.players, fmt_signed(r.gap_raw), fmt_signed(r.gap_adj), pct(r.share_closed))
    r = gl("batter", "T20I other full member", "T20I top-8")
    C(s, "| {} batters: v other full members minus v top-8 | {} | {} |", r.players, fmt_signed(r.gap_raw), fmt_signed(r.gap_adj))
    assert abs(r.gap_adj) < r.se_adj
    r = gl("bowler", "T20I top-8", "IPL")
    C(s, "| {} bowlers: v T20I top-8 minus v IPL | {} | {} | {} |", r.players, fmt_signed(r.gap_raw), fmt_signed(r.gap_adj), pct(r.share_closed))
    r = gl("bowler", "T20I other full member", "T20I top-8")
    C(s, "| {} bowlers: v other full members minus v top-8 | {} | {} | {} |", r.players, fmt_signed(r.gap_raw), fmt_signed(r.gap_adj), pct(r.share_closed))
    ob = T("phase4/overall_value_adjusted_bowlers.csv").set_index("name")
    n_ranked = int(ob["rank_adjusted"].notna().sum())
    rk = lambda n, c: ordinal(int(ob.loc[n, c]))  # noqa: E731
    C(s, "Ranked on all-ball value among the {} bowlers with at least {} balls, Blessing Muzarabani falls from {} to {}, "
         "Mohammad Nawaz from {} to {}, and Mahedi Hasan from {} to {} (his value falls from {} to {} runs per 100 balls)",
      n_ranked, config["value"]["min_balls_overall"], rk("Blessing Muzarabani", "rank_raw"), rk("Blessing Muzarabani", "rank_adjusted"),
      rk("Mohammad Nawaz", "rank_raw"), rk("Mohammad Nawaz", "rank_adjusted"),
      rk("Mahedi Hasan", "rank_raw"), rk("Mahedi Hasan", "rank_adjusted"),
      fmt_signed(ob.loc["Mahedi Hasan", "value_raw"]), fmt_signed(ob.loc["Mahedi Hasan", "value_adjusted"]))
    C(s, "Jasprit Bumrah, who bowls mostly in the IPL, stays {} of the same {}", rk("Jasprit Bumrah", "rank_adjusted"), n_ranked)
    am = T("phase4/adjusted_means.csv").query("slice == 'competition'").set_index(["role", "level"])["adjusted"]
    ipl_gap_bat = am[("batter", "IPL")] - am[("batter", "T20I")]
    ipl_gap_bowl = am[("bowler", "IPL")] - am[("bowler", "T20I")]
    C(s, "worth {:.1f} runs per 100 balls more than an average T20I performance for batters, and {:.1f} for bowlers",
      ipl_gap_bat, ipl_gap_bowl)
    C("6 insights", "worth {:.1f} runs per 100 balls more than an average T20I performance for batters ({:.1f} for bowlers)",
      ipl_gap_bat, ipl_gap_bowl)
    before = T("phase3/value_by_season_before_recalibration.csv").query("role == 'batter' and level == '2026'")
    after = T("phase3/value_zero_mean_checks.csv").query("role == 'batter' and slice == 'season' and level == '2026'")
    C(s, "batting value would be {} runs per 100 balls in 2026 for nothing; with it, {}",
      fmt_signed(before["value_per_100"].iloc[0]), fmt_num(after["value_per_100"].iloc[0]))
    dep = T("phase2b/exposure_dependence.csv").set_index("role")
    C(s, "explains {} of the differences in their share of pressure balls ({} for bowlers)",
      pct(dep.loc["batter", "r2_chase_share"]), pct(dep.loc["bowler", "r2_chase_share"]))
    teams = T("phase2b/exposure_by_team.csv")
    t20i, ipl = teams[teams["comp_type"] == "T20I"], teams[teams["comp_type"] == "IPL"]
    assert t20i.loc[t20i["batting_hp_share"].idxmin(), "team"] == "India"
    lo, hi = ipl.loc[ipl["batting_hp_share"].idxmin()], ipl.loc[ipl["batting_hp_share"].idxmax()]
    C(s, "({}), and in the IPL the share ranges from {} ({}) to {} ({})",
      pct(t20i["batting_hp_share"].min(), 1), pct(lo.batting_hp_share, 1), lo.team, pct(hi.batting_hp_share, 1), hi.team)
    vol = T("phase5/volume_checks.csv").set_index("role")
    C(s, "({:.2f} against {:.2f} for batters)", vol.loc["batter", "top_n_mean_share_balls_in_chase"],
      vol.loc["batter", "eligible_median_share_balls_in_chase"])

    # ------------------------------------------------------------------ 4. rating and results
    s = "4 rating"
    br = T("phase5/consistency_base_rates.csv").query("weighting == 'uniform'").set_index("role")["base_rate"]
    C(s, "({} for batters, {} for bowlers;", pct(br["batter"], 1), pct(br["bowler"], 1))
    C(s, "repeated over {} random splits", rcfg["split_reps"])
    ew = T("phase5/effective_weights.csv").set_index(["role", "component"])
    w = lambda role, comp, col: ew.loc[(role, comp), col]  # noqa: E731
    C(s, "| Rating formula | {:.2f} × z(pressure value) + {:.2f} × z(consistency) | {:.2f} × z(pressure value) + {:.2f} × z(consistency) |",
      w("batter", "pressure", "share_of_weight"), w("batter", "consistency", "share_of_weight"),
      w("bowler", "pressure", "share_of_weight"), w("bowler", "consistency", "share_of_weight"))
    C(s, "| In original units | {:.3f} × pressure value + {:.3f} × consistency points | {:.3f} × pressure value + {:.3f} × consistency points |",
      w("batter", "pressure", "effective_coefficient"), w("batter", "consistency", "effective_coefficient") / 100,
      w("bowler", "pressure", "effective_coefficient"), w("bowler", "consistency", "effective_coefficient") / 100)
    C(s, "| Correlation of the two components | {:.2f} | {:.2f} |", vol.loc["batter", "corr_pressure_v_consistency_shrunk"],
      vol.loc["bowler", "corr_pressure_v_consistency_shrunk"])
    C(s, "| Uplift weight | {:.0f} (reliability {}) | {:.0f} (reliability {}) |", w("batter", "uplift", "share_of_weight"),
      f2(rel.loc[("batter", "uplift"), "random_mean"]), w("bowler", "uplift", "share_of_weight"),
      f2(rel.loc[("bowler", "uplift"), "random_mean"]))
    for comp, label, bold in (("pressure", "Pressure value", True), ("consistency", "Consistency", False),
                              ("uplift", 'Uplift ("clutch")', False), ("usual", "Usual value (ordinary balls, reference)", False)):
        b = "**" if bold else ""
        C(s, "| " + label + " | {} | " + b + "{}" + b + " | {} | " + b + "{}" + b + " |",
          f2(rel.loc[("batter", comp), "r_random_mean"]), f2(rel.loc[("batter", comp), "random_mean"]),
          f2(rel.loc[("bowler", comp), "r_random_mean"]), f2(rel.loc[("bowler", comp), "random_mean"]))
    C(s, "For each player, {:,} values are drawn", rcfg["posterior_draws"])
    lo_q, hi_q = rcfg["interval"]
    C(s, "This gives a **{:.0f}% interval**", (hi_q - lo_q) * 100)
    C(s, "whose P(top 10) differ by at most {:.2f}", rob["tie_break_max_p_gap"])
    C(s, "each player is re-ranked under {} alternative choices (§5.1). A player is **stable** if in the top 10 in at "
         "least {:.0f}% of them, **moderately stable** from {:.0f}%", n_counted, rob["stability_cuts"][0] * 100,
      rob["stability_cuts"][1] * 100)
    C(s, "(B rank {} or better)", config["report"]["framing_min_b_rank"])
    C(s, "B ranks them below {}", ordinal(config["report"]["framing_min_b_rank"]))
    C(s, "short record (fewer than {} pressure balls)", config["report"]["short_record_pressure_balls"])
    a_v_b = T("phase6/a_v_b_overlap.csv")
    assert (a_v_b["final_and_a"] == rcfg["top_n"]).all()      # "exactly the same ten players per role as option A"

    card = T("phase6/evidence_card_batter_50284.csv").set_index("field")["value"]
    num = lambda k: float(card[k])  # noqa: E731
    sc = T("phase6/stability_classes_batters.csv").set_index("player_id").loc[50284]
    assert int(card["final_rank"]) == int(card["rank_a"]) == int(card["rank_b"]) == 1
    C(s, "| Record | {:.0f} pressure balls in {:.0f} matches ({:.0f} matches in all; eligibility bar {} balls) |",
      num("pressure_balls"), num("pressure_matches"), num("matches"), config["eligibility"]["min_pressure_balls"])
    C(s, "| What happened | {:.0f} runs against {:.0f} expected ({} runs per 100 balls); out {:.0f} times against {:.1f} expected; "
         "boundaries on {:.0f}% of balls |", num("runs"), num("expected_runs"), fmt_signed(num("runs_v_expected_per_100"), 0),
      num("wickets"), num("expected_wickets"), num("boundary_pct"))
    C(s, "| Pressure value | raw {}, shrunk **{}** (90% interval {} to {}) |", fmt_signed(num("pressure_value_raw")),
      fmt_signed(num("pressure_value_shrunk")), fmt_signed(num("pressure_value_p05")), fmt_signed(num("pressure_value_p95")))
    C(s, "| P(top 10) | **{:.2f}** (A); {:.2f} (B); {:.2f} if every player is cut to {} pressure balls |", num("p_top_n"),
      num("p_top_n_hierarchical"), num("p_top_n_equal_balls"), config["eligibility"]["min_pressure_balls"])
    C(s, "| 90% rank interval | {} |", str(card["rank_interval"]).replace("-", " to "))
    assert float(sc.share_in_top_n) == 1.0
    C(s, "in the top 10 in all {} variants in which he is eligible (worst rank {:.0f})", int(sc.variants_eligible), sc.worst_rank)
    C(s, "| Usual level | {} on ordinary balls", fmt_signed(num("usual_value_raw")))
    parts = [num(f"{k}_part_per_100") for k in ("runs", "wickets", "non_striker", "opposition")]
    assert abs(sum(parts) - num("pressure_value_raw")) < 0.05            # the four parts add up to the raw value
    assert num("wickets") > num("expected_wickets") and parts[1] > 0   # "more outs, yet a positive wicket term"
    C(s, "| Breakdown of the raw value | runs above expectation {}, wicket term {} (his {:.0f} dismissals came where wickets "
         "were cheap, so they cost less than the {:.1f} expected ones would have), non-striker run-out term {}, opposition "
         "adjustment {}: total {} per 100 balls |", fmt_signed(parts[0]), fmt_signed(parts[1]), num("wickets"),
      num("expected_wickets"), fmt_signed(parts[2]), fmt_signed(parts[3]), fmt_signed(num("pressure_value_raw")))
    n_el_pool = elig_final.groupby("role").size()
    C("4 overall v pressure", "among the same eligible players as the pressure ranking ({} batters, {} bowlers)",
      int(n_el_pool["batter"]), int(n_el_pool["bowler"]))
    C(s, "heavy shrinkage ({:.1f} to {:.1f})", num("pressure_value_raw"), num("pressure_value_shrunk"))
    pb = final[final["role"] == "batter"].set_index("rank")["pressure_balls"]
    C(s, "behind #2 and #3 ({} and {} pressure balls)", int(pb[2]), int(pb[3]))

    usual = ovp.set_index("player_id")["usual_value_raw"]
    for name in ["Yashasvi Jaiswal", "Pathum Nissanka", "Andre Russell", "Rajat Patidar", "Ashutosh Sharma",
                 "Patrick Cummins", "Akeal Hosein", "Mark Wood", "Blessing Muzarabani", "Noor Ahmad"]:
        role = "bowler" if name in ("Patrick Cummins", "Akeal Hosein", "Mark Wood", "Blessing Muzarabani", "Noor Ahmad") else "batter"
        r = elig_final[(elig_final["role"] == role) & (elig_final["player_name"] == name)].iloc[0]
        C("4 overall v pressure", "| " + name + " | {} → {} | {} | {} | {} |", int(r.overall_rank), int(r["rank"]),
          fmt_signed(usual[r.player_id]), fmt_signed(r.pressure_value_raw), int(r.pressure_balls))

    ab = pd.concat([T("phase6/notable_absences_batters.csv"), T("phase6/notable_absences_bowlers.csv")]).set_index(["role", "name"])
    for (role, name), label in {("batter", "Virat Kohli"): "Virat Kohli (bat)", ("batter", "Rashid Khan"): "Rashid Khan (bat)",
                                ("batter", "Andre Russell"): "Andre Russell (bat)", ("bowler", "Jasprit Bumrah"): "Jasprit Bumrah (bowl)",
                                ("bowler", "Rashid Khan"): "Rashid Khan (bowl)", ("bowler", "Andre Russell"): "Andre Russell (bowl)"}.items():
        r = ab.loc[(role, name)]
        C("4 absences", "| " + label + " | {} | {} | {} → {} ({} to {}) | {:.2f} |", int(r.final_rank), int(r.pressure_balls),
          fmt_signed(r.pressure_value_raw), fmt_signed(r.pressure_value_shrunk), fmt_num(r.pressure_value_p05),
          fmt_num(r.pressure_value_p95), r.p_top_n)
    k = ab.loc[("batter", "Virat Kohli")]
    C("4 absences", "Scored {:.0f} run per 100 balls below expectation; out {:.0f} times against {:.1f} expected",
      -k.runs_v_expected_per_100, k.wickets, k.expected_wickets)
    k = ab.loc[("batter", "Andre Russell")]
    C("4 absences", "Scored {:.0f} runs per 100 above expectation but out {:.0f} times against {:.1f} expected",
      k.runs_v_expected_per_100, k.wickets, k.expected_wickets)
    C("4 absences", "Good record on only {} balls", int(ab.loc[("batter", "Rashid Khan"), "pressure_balls"]))
    k = ab.loc[("bowler", "Jasprit Bumrah")]
    C("4 absences", "Conceded {:.0f} runs per 100 balls fewer than expected (excellent) but took {:.0f} wickets against {:.1f} expected; {} on ordinary balls",
      -k.runs_v_expected_per_100, k.wickets, k.expected_wickets, fmt_signed(k.usual_value_raw))
    k = ab.loc[("bowler", "Rashid Khan")]
    # The pressure-ball boundary average is the one quoted in the absences table's own reason text.
    league_bowl = float(re.search(r"pressure-ball average (\d+)%", k.reason).group(1))
    C("4 absences", "Conceded {:.0f} runs per 100 fewer than expected and boundaries on {:.0f}% of balls (average {:.0f}%); {:.0f} wickets against {:.1f}",
      -k.runs_v_expected_per_100, k.boundary_pct, league_bowl, k.wickets, k.expected_wickets)
    k = ab.loc[("bowler", "Andre Russell")]
    C("4 absences", "Conceded {:.0f} runs per 100 more than expected; boundaries on {:.0f}% of balls", k.runs_v_expected_per_100, k.boundary_pct)
    k = ab.loc[("batter", "Virat Kohli")]
    kc = T("phase6/stability_classes_batters.csv").set_index("player_id").loc[int(k.player_id), "stability"]
    assert kc == "moderately stable" and int(k.final_rank) == rcfg["top_n"] + 1

    # ------------------------------------------------------------------ 5. reliability and limitations
    s = "5 reliability"
    for role in ("batter", "bowler"):
        top = final[(final["role"] == role) & (final["rank"] <= rcfg["top_n"])]
        n_stable = int((top["stability"] == "stable").sum())
        if role == "batter":
            C(s, "{} of the ten batters are in the top 10 in at least {:.0f}% of variants", WORDS[n_stable], rob["stability_cuts"][0] * 100)
            mod = top[top["stability"] == "moderately stable"]["player_name"].str.split().str[-1].tolist()
            C(s, "the three moderately stable batters ({})", ", ".join(mod[:-1]) + ", " + mod[-1])
        else:
            C(s, "{} of the ten bowlers are stable", WORDS[n_stable])
    head = {}
    for role in ("batter", "bowler"):
        r = variants[variants["role"] == role]
        top_ids = set(r[(r["variant"] == "headline") & (r["rank"] <= rcfg["top_n"])]["player_id"])
        head[role] = {v: len(top_ids & set(gv[gv["rank"] <= rcfg["top_n"]]["player_id"])) for v, gv in r.groupby("variant")}
        hb = set(r[(r["variant"] == "headline") & (r["rank_hierarchical"] <= rcfg["top_n"])]["player_id"])
        head[role]["option B"] = len(top_ids & hb)
    pairs = [("√LI weight", "weight: sqrt LI", "IPL only", "IPL only"), ("LI cap weight", "weight: LI cap", "T20I only", "T20I only"),
             ("Knockouts ×1.25", "knockouts x1.25", "Without 2022", "without 2022"),
             ("Eligibility 150", "eligibility 150", "Without 2023", "without 2023"),
             ("Eligibility 200", "eligibility 200", "Without 2024", "without 2024"),
             ("Associates excluded", "associates excluded", "Without 2025", "without 2025"),
             ("Position / phase relative", "role: position / phase relative", "Without 2026", "without 2026"),
             ("Logistic leverage", "pressure: logistic leverage", "Per-innings cut", "pressure: per-innings cut")]
    for la, va, lb, vb in pairs:
        C("5 sensitivity table", "| " + la + " | {} | {} | | " + lb + " | {} | {} |", head["batter"][va], head["bowler"][va],
          head["batter"][vb], head["bowler"][vb])
    C("5 sensitivity table", "| *Rule index (alternative)* | *{}* | *{}* | | *Option B (alternative)* | *{}* | *{}* |",
      head["batter"]["pressure: rule index"], head["bowler"]["pressure: rule index"], head["batter"]["option B"], head["bowler"]["option B"])
    rating_vars = ["weight: sqrt LI", "weight: LI cap", "knockouts x1.25", "eligibility 150", "role: position / phase relative"]
    vals = [head[r][v] for r in head for v in rating_vars] + [head["bowler"]["eligibility 200"]]
    C(s, "knockout weight, eligibility and role baselines leave {}-{} of the top 10 in place, and excluding associate matches {}-{}",
      min(vals), max(vals), min(head[r]["associates excluded"] for r in head), max(head[r]["associates excluded"] for r in head))
    e200 = variants[(variants["role"] == "batter") & (variants["variant"] == "eligibility 200")].set_index("player_id")["rank"]
    top_bat = variants[(variants["role"] == "batter") & (variants["variant"] == "headline") & (variants["rank"] <= rcfg["top_n"])]
    missing = sorted(set(top_bat["player_id"]) - set(e200.index))
    assert missing == [4736, 50284] and int(e200[4260]) == 11     # Inglis and Sooryavanshi ineligible; Klaasen 11th
    C(s, "eligibility at 200 balls for batters ({} of 10): Sooryavanshi and Inglis fall below the bar, and Klaasen slips to {}",
      head["batter"]["eligibility 200"], ordinal(int(e200[4260])))
    seasons = [v for v in groups["seasons"]]
    changes = [rcfg["top_n"] - head[r][v] for r in head for v in seasons]
    C(s, "dropping a season changes {}-{} names", min(changes), max(changes))
    pool = variants[variants["variant"].isin(["IPL only", "T20I only"])].groupby(["role", "variant"]).size()
    C(s, "shrink the pool to {}-{} players", int(pool.min()), int(pool.max()))
    pi = [rcfg["top_n"] - head[r]["pressure: per-innings cut"] for r in head]
    ri = [rcfg["top_n"] - head[r]["pressure: rule index"] for r in head]
    C(s, "the per-innings cut changes {}-{} names and the rule index {}-{}", min(pi), max(pi), min(ri), max(ri))
    C(s, "(rank correlation of rating with pressure balls {:.2f} for batters, {:.2f} for bowlers)",
      vol.loc["batter", "spearman_rating_v_pressure_balls"], vol.loc["bowler", "spearman_rating_v_pressure_balls"])
    C(s, "already in the raw records ({:.2f} and {:.2f})", vol.loc["batter", "spearman_raw_pressure_v_pressure_balls"],
      vol.loc["bowler", "spearman_raw_pressure_v_pressure_balls"])
    C(s, "Cutting every player to {} pressure balls, the top 10's average chance of staying in the top 10 is {:.2f} (batters) and {:.2f} (bowlers)",
      config["eligibility"]["min_pressure_balls"], vol.loc["batter", "top_n_mean_p_top_n_equal_balls"],
      vol.loc["bowler", "top_n_mean_p_top_n_equal_balls"])
    C(s, "the reliability of uplift is {} for batters and {} for bowlers", f2(rel.loc[("batter", "uplift"), "random_mean"]),
      f2(rel.loc[("bowler", "uplift"), "random_mean"]))
    C(s, "Pressure value itself does reproduce ({} and {})", f2(rel.loc[("batter", "pressure"), "random_mean"]),
      f2(rel.loc[("bowler", "pressure"), "random_mean"]))
    C(s, "**Reliability is modest** ({} and {})", f2(rel.loc[("batter", "pressure"), "random_mean"]),
      f2(rel.loc[("bowler", "pressure"), "random_mean"]))
    pb_role = elig_final.groupby("role")["pressure_balls"]
    C(s, "at {}-{} pressure balls per player (median {:.0f} for batters, {:.0f} for bowlers)", int(elig_final["pressure_balls"].min()),
      int(elig_final["pressure_balls"].max()), pb_role.median()["batter"], pb_role.median()["bowler"])
    r = gl("batter", "T20I other full member", "T20I top-8")
    C(s, "of batting against the four \"other full members\" ({}, within one standard error", fmt_num(r.gap_adj))
    facts = json.loads((T.outputs / "audit" / "facts.json").read_text(encoding="utf-8"))["reconciliation"]
    C(s, "({} of {} checkable matches reproduce)", facts["run_margin_exact"] + facts["wicket_margin_exact"],
      facts["run_margin_matches"] + facts["wicket_margin_matches"])
    ids = facts["run_margin_mismatch_ids"] + facts["wicket_margin_mismatch_ids"]
    C(s, "a missing delivery in match {} and a wrong result string in match {}", ids[0], ids[1])

    # ------------------------------------------------------------------ 6. insights
    s = "6 insights"
    C(s, "Only {} of the balls a required-rate-and-wickets rule calls", pct(rule.share_of_leverage_hp_also_rule_hp))
    C(s, "(win probability {})", pct(sit.loc["Chase: need 60 off 12 balls, 8 down", "win_probability"], 1))
    nb = T("phase4/overall_value_adjusted_batters.csv")
    n_bat = int(nb["rank_adjusted"].notna().sum())
    nar = nb[nb["player_id"] == 781].iloc[0]
    C(s, "Among the {} batters with at least {} balls he ranks {} on opposition-adjusted value ({} on raw value)", n_bat,
      config["value"]["min_balls_overall"], ordinal(int(nar.rank_adjusted)), ordinal(int(nar.rank_raw)))
    assert int(final[(final["role"] == "bowler") & (final["player_id"] == 781)]["rank"].iloc[0]) == 1
    C(s, "(reliability {} batters, {} bowlers)", f2(rel.loc[("batter", "uplift"), "random_mean"]),
      f2(rel.loc[("bowler", "uplift"), "random_mean"]))

    # ------------------------------------------------------------------ appendices
    s = "appendix A"
    ab_ = T("phase2b/ablations.csv")
    for model, label in (("lightgbm", "LightGBM"), ("xgboost", "XGBoost"), ("logistic", "Logistic regression")):
        base = ab_[(ab_["model"] == model) & (ab_["variant"] == "base")]["log_loss_all"].iloc[0]
        leak = ab_[(ab_["model"] == model) & ab_["variant"].str.contains("identity leak")]["log_loss_all"].iloc[0]
        bold = "**" if model != "logistic" else ""
        C(s, "| " + label + " | {:.4f} | " + bold + "{:.4f}" + bold + " | {} |", base, leak, fmt_signed(leak - base, 4))
    s = "appendix B"
    for model, label in (("logistic", "Logistic regression"), ("xgboost", "XGBoost"), ("lightgbm", "LightGBM"), ("dp", "DP"),
                         ("dp_platt", "DP + recalibration")):
        r = wp.loc[model]
        C(s, "| " + label + " | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |", r.log_loss_all, r.log_loss_inn1,
          r.log_loss_inn2, r.log_loss_chase_last30, r.brier_all, r.ece_all)
    dpp = ab_[ab_["model"] == "dp_platt"].set_index("variant")
    C(s, "raises log-loss from {:.4f} to {:.4f}. An Impact Player flag changes it by {}", dpp.loc["base", "log_loss_all"],
      dpp.loc["no scoring-environment adjustment", "log_loss_all"], fmt_signed(dpp.loc["+ Impact Player flag", "change_vs_base"], 4))
    for key, label, digits in (("xr_bat", "Runs off the bat | Squared error", 3), ("xw_bat", "Striker dismissed | Log-loss", 4),
                               ("xr_bowl", "Runs charged to bowler | Squared error", 3), ("xw_bowl", "Bowler-credited wicket | Log-loss", 4)):
        C(s, "| " + label + " | {:." + str(digits) + "f} | {:." + str(digits) + "f} |", es.loc[key, "season_phase_average"],
          es.loc[key, "full_model"])
    vt = T("phase3/value_v_traditional.csv").set_index("traditional")["pearson"]
    C(s, "run value correlates {} with strike rate and {} with batting average; {} with economy rate and {} with balls per wicket",
      fmt_signed(vt["strike_rate"], 2), fmt_signed(vt["average"], 2), fmt_num(vt["economy"], 2), fmt_num(vt["balls_per_wicket"], 2))
    s = "appendix E"
    C(s, "the last full `--force` run took about {:.0f} minutes", log_minutes(T.outputs / "pipeline_run.log"))
    cached = log_minutes(T.outputs / "pipeline_run_cached.log")       # varies with machine load (4-11 minutes seen)
    C(s, "a cached run takes under {} minutes", max(15, 5 * math.ceil(cached / 5)))
    C(s, "**Fixed seeds** (`seed: {}` in `config.yaml`)", config["seed"])
    return out
