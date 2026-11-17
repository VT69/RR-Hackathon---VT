"""End-to-end pipeline. Run from the project root:

    python -m src.pipeline            # reuses the cached model stage if present
    python -m src.pipeline --force    # recomputes everything from the raw CSVs

Stages so far: data audit -> match state -> cross-fitted win-probability candidates
-> pressure (leverage) -> Phase 2b validation -> expectations, wicket cost and run
value (Phase 3) -> Phase 3 sanity checks.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data.audit import run_audit
from src.data.loading import load_raw_tables
from src.features import pressure_validation as pv
from src.features.context_extras import add_context_extras
from src.features.credit import add_credit_columns, add_situation_features
from src.features.leverage import state_frame, win_probability
from src.features.match_state import build_ball_states, build_match_table
from src.features.pressure import add_pressure_columns, add_realised_swing
from src.features.rule_index import rule_pressure_index
from src.models.crossfit import (
    CANDIDATES, CrossFit, leverage_at, predict_candidate, roughness_candidate, run_crossfit, score_candidate,
)
from src.metrics import value_validation as vv
from src.metrics.player_value import overall_table, player_balls, pressure_split
from src.metrics.run_value import add_run_value, add_wpa
from src.metrics.wicket_cost import cost_parts, crossfit_calibration, crossfit_cost_parts, raw_cost, wicket_cost
from src.models.evaluate import log_loss
from src.rating import fairness_validation as fv
from src.rating import rating_report as rr
from src.rating import ratings as rt
from src.rating.opposition import adjust
from src.report import build_pdf, check_report
from src.report import final_outputs as fo
from src.viz.report_plots import ReportFigures
from src.robustness import sensitivity as sens
from src.models.expected import (
    baseline_expectations, crossfit_expectations, fingerprint_audit, recalibrate_by_season,
)
from src.viz import pressure_plots as plots
from src.viz import value_plots

logger = logging.getLogger(__name__)
COUNTERFACTUAL_CTX_OFFSET = 5


def _reference_states(cf: CrossFit, rows: list[dict[str, float]]) -> tuple[int, pd.DataFrame]:
    """State frame for hand-written situations, placed in the configured reference context."""
    ref = cf.config["pressure"]["reference_context"]
    full = cf.config["ball_semantics"]["scheduled_overs"] * cf.config["ball_semantics"]["balls_per_over"]
    pool = cf.balls[(cf.balls["year"] == ref["year"]) & (cf.balls["comp_type"] == ref["comp_type"]) & (cf.balls["involves_associate"] == 0)]
    context = pool[["ctx", "env_rpo", "is_ipl", "impact_rule", "involves_associate"]].mean().to_dict()
    frame = pd.DataFrame(rows).assign(scheduled_balls=full, scheduled2=full, **context)
    frame["ctx"] = int(context["ctx"])
    return int(ref["year"]), frame


def _leverage_slices(cf: CrossFit, mean_leverage: dict[str, float]) -> pd.DataFrame:
    """LI along a line of chase states (fixed balls and wickets) for every candidate."""
    cfg = cf.config["pressure"]["slice"]
    runs = np.arange(1, cfg["max_runs"] + 1)
    year, states = _reference_states(
        cf, [{"balls_remaining": cfg["balls_remaining"], "x": int(r), "wickets_in_hand": cfg["wickets_in_hand"]} for r in runs]
    )
    parts = []
    for name in CANDIDATES:
        _, lev = leverage_at(cf, name, year, 2, states)
        parts.append(pd.DataFrame({"model": name, "runs_required": runs, "li": lev / mean_leverage[name]}))
    return pd.concat(parts, ignore_index=True)


def _example_situations(cf: CrossFit, name: str, mean_leverage: float) -> pd.DataFrame:
    """WP and LI of the chosen model in the hand-written example situations."""
    rows = []
    for situation in cf.config["pressure"]["example_situations"]:
        year, states = _reference_states(cf, [{k: situation[k] for k in ("balls_remaining", "x", "wickets_in_hand")}])
        wp, lev = leverage_at(cf, name, year, situation["innings"], states)
        rows.append({"situation": situation["label"], "win_probability": wp[0], "leverage_index": lev[0] / mean_leverage})
    return pd.DataFrame(rows)


def _impact_wicket_cost(cf: CrossFit, name: str) -> pd.DataFrame:
    """Model-implied WP cost of losing a wicket in IPL 2023+ states, with and without the Impact Player flag.

    ``cf`` must be a cross-fit whose models include the flag (the ablation run, not the headline run).
    """
    scored = cf.scored()
    rows = scored[(scored["impact_rule"] == 1) & (scored["wickets_in_hand"] >= 2)]
    parts = []
    for year, fold in cf.models.items():
        for innings in (1, 2):
            part = rows[(rows["year"] == year) & (rows["innings"] == innings)]
            if part.empty:
                continue
            states = state_frame(part, innings)
            costs = {}
            for flag in (1, 0):
                s = states.copy()
                if flag == 0:
                    for ctx in s["ctx"].unique():
                        fold["dp"].contexts[int(ctx) + COUNTERFACTUAL_CTX_OFFSET] = {**fold["dp"].contexts[int(ctx)], "impact_rule": 0.0}
                    s["ctx"] = s["ctx"] + COUNTERFACTUAL_CTX_OFFSET
                    s["impact_rule"] = 0
                with_wicket = win_probability(fold[name], innings, s)
                without = win_probability(fold[name], innings, s.assign(wickets_in_hand=s["wickets_in_hand"] - 1))
                costs[flag] = with_wicket - without
            parts.append(pd.DataFrame({"innings": innings, "overid": part["overid"].to_numpy(),
                                       "cost_with_rule": costs[1], "cost_without_rule": costs[0]}))
    frame = pd.concat(parts, ignore_index=True)
    frame["phase"] = pd.cut(frame["overid"], [0, 6, 15, 20], labels=["powerplay", "middle", "death"])
    table = frame.groupby(["innings", "phase"], observed=True)[["cost_with_rule", "cost_without_rule"]].mean()
    total = frame[["cost_with_rule", "cost_without_rule"]].mean().to_frame().T.assign(innings="all", phase="all")
    out = pd.concat([table.reset_index(), total], ignore_index=True)
    out["relative_change_pct"] = (out["cost_with_rule"] / out["cost_without_rule"] - 1) * 100
    return out.assign(model=name)


def _with_environment(balls: pd.DataFrame, env: pd.Series, per_over: int) -> pd.DataFrame:
    """Copy of the ball table with a different scoring-environment column and the features derived from it."""
    out = balls.assign(env_rpo=env.to_numpy())
    out["rrr_rel"] = out["runs_required"] * per_over / out["balls_remaining"].clip(lower=1) / out["env_rpo"]
    out["runs_vs_env"] = out["score_before"] - out["env_rpo"] * out["legal_before"] / per_over
    return out


def _loss_row(model: str, label: str, clean: pd.DataFrame, p: np.ndarray, clip: float) -> dict[str, Any]:
    y = clean["batting_team_won"].to_numpy()
    first = clean["innings"].eq(1).to_numpy()
    return {"model": model, "variant": label, "log_loss_all": log_loss(p, y, clip),
            "log_loss_inn1": log_loss(p[first], y[first], clip), "log_loss_inn2": log_loss(p[~first], y[~first], clip)}


def _ablations(
    balls: pd.DataFrame, match_table: pd.DataFrame, config: dict[str, Any], chosen: str, scores: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Held-out log-loss of feature variants: what does each piece of context buy?

    Also returns the wicket-cost comparison from the variant that includes the Impact Player flag.
    """
    clip = config["models"]["prob_clip"]
    extras = add_context_extras(balls, match_table, config)
    per_over = config["ball_semantics"]["balls_per_over"]
    # One scoring level for every season and competition (no era adjustment at all).
    flat = _with_environment(balls, pd.Series(balls["env_rpo"].mean(), index=balls.index), per_over)
    # The originally approved version: season average excluding the match itself. Kept only to
    # document why it was dropped - the value is nearly unique per match (see DECISIONS.md D10).
    runs = match_table["total1"].fillna(0) + match_table["total2"].fillna(0)
    legal = match_table["legal1"].fillna(0) + match_table["legal2"].fillna(0)
    grp = [match_table["comp_type"], match_table["year"]]
    loo = (runs.groupby(grp).transform("sum") - runs) / (legal.groupby(grp).transform("sum") - legal) * per_over
    loo_by_match = pd.Series(loo.to_numpy(), index=match_table["match_id"].to_numpy())
    leaky = _with_environment(balls, balls["match_id"].map(loo_by_match), per_over)
    variants: list[tuple[str, str, pd.DataFrame, tuple[str, ...], tuple[str, ...]]] = [
        ("lightgbm", "base", balls, (), ()),
        ("lightgbm", "no scoring-environment adjustment", flat, ("env_rpo",), ()),
        ("lightgbm", "environment excluding the match itself (identity leak)", leaky, (), ()),
        ("xgboost", "base", balls, (), ()),
        ("xgboost", "environment excluding the match itself (identity leak)", leaky, (), ()),
        ("logistic", "base", balls, (), ()),
        ("logistic", "environment excluding the match itself (identity leak)", leaky, (), ()),
        ("lightgbm", "+ Impact Player flag", balls, (), ("impact_rule",)),
        ("lightgbm", "no associate flag", balls, ("involves_associate",), ()),
        ("lightgbm", "+ pre-match team strength (Elo)", extras, (), ("elo_diff",)),
        ("lightgbm", "+ venue factor", extras, (), ("venue_factor",)),
        ("logistic", "+ pre-match team strength (Elo)", extras, (), ("elo_diff",)),
        ("logistic", "+ venue factor", extras, (), ("venue_factor",)),
        ("logistic", "+ Impact Player flag", balls, (), ("impact_rule",)),
    ]
    rows = []
    wicket_cost = pd.DataFrame()
    if chosen.startswith("dp"):  # the base run already exists; only the variant needs new DP tables
        variants.append((chosen, "+ Impact Player flag", balls, (), ("impact_rule",)))
        variants.append((chosen, "no scoring-environment adjustment", flat, ("env_rpo",), ()))
        done = scores[scores["wp_train_row"]]
        rows.append(_loss_row(chosen, "base", done, done[f"wp_{chosen}"].to_numpy(), clip))
    for model, label, data, drop, extra in variants:
        is_dp = model.startswith("dp")
        cf = run_crossfit(data, config, drop=drop, fitted=() if is_dp else (model,), extra=extra, with_dp=is_dp)
        clean = cf.balls[cf.balls["wp_train_row"]]
        rows.append(_loss_row(model, label, clean, predict_candidate(cf, model, clean, extra), clip))
        logger.info("Ablation %s / %s: log-loss %.4f", model, label, rows[-1]["log_loss_all"])
        if is_dp and "impact_rule" in extra:
            wicket_cost = _impact_wicket_cost(cf, model)
    out = pd.DataFrame(rows)
    base = out[out["variant"] == "base"].set_index("model")["log_loss_all"]
    out["change_vs_base"] = out["log_loss_all"] - out["model"].map(base)
    return {"ablations": out, "impact_wicket_cost": wicket_cost}


def model_stage(balls: pd.DataFrame, match_table: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Everything that needs fitted models: cross-fit, score every candidate, diagnostics."""
    chosen = config["pressure"]["wp_model"]
    cf = run_crossfit(balls, config)
    scores = cf.balls.copy()
    mean_leverage: dict[str, float] = {}
    for name in CANDIDATES:
        scored = score_candidate(cf, name)
        scores[f"wp_{name}"] = scored["wp"]
        scores[f"lev_{name}"] = scored["leverage"]
        mean_leverage[name] = float(scored["leverage"].mean())
        logger.info("Scored %s", name)
    roughness = pd.DataFrame([{"model": name, **roughness_candidate(cf, name)} for name in CANDIDATES])
    return {
        "scores": scores,
        "outcome_probs": cf.outcome_probs,
        "roughness": roughness,
        "slices": _leverage_slices(cf, mean_leverage),
        "situations": _example_situations(cf, chosen, mean_leverage[chosen]),
        # Kept so later stages can ask the chosen model about any state without refitting.
        "fold_models": {year: {chosen: fold[chosen]} for year, fold in cf.models.items()},
        "outcome_models": cf.outcome_models,
    }


def pressure_table(results: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    """Ball table with the chosen model's WP, leverage, LI, flags and the rule-based index."""
    chosen = config["pressure"]["wp_model"]
    balls = results["scores"].copy()
    balls["wp"] = balls[f"wp_{chosen}"]
    balls["leverage"] = balls[f"lev_{chosen}"]
    balls = add_pressure_columns(balls, config)
    balls["rule_index"] = rule_pressure_index(balls, config)
    return add_realised_swing(balls)


def validation_stage(results: dict[str, Any], match_table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Write every Phase 2b table and figure; returns the ball-level pressure table."""
    out_root = resolve_path(config["paths"]["outputs_dir"])
    tables, figures = out_root / "tables" / "phase2b", out_root / "figures" / "phase2b"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    share = config["pressure"]["high_pressure_share"]

    comparison = pv.model_comparison(results["scores"], CANDIDATES, config)
    overall = comparison["overall"].merge(results["roughness"], on="model")
    overall.to_csv(tables / "wp_model_comparison.csv", index=False)
    comparison["by_season"].to_csv(tables / "wp_model_by_season.csv", index=False)
    comparison["reliability"].to_csv(tables / "wp_reliability.csv", index=False)
    comparison["bootstrap"].to_csv(tables / "wp_model_bootstrap_gaps.csv", index=False)
    pv.leverage_agreement(results["scores"], CANDIDATES, share).to_csv(tables / "leverage_agreement_between_models.csv", index=False)
    pv.last_ball_table(results["scores"], CANDIDATES).to_csv(tables / "wp_last_ball_of_chase.csv", index=False)
    for key in ("slices", "situations", "impact_wicket_cost", "ablations"):
        results[key].to_csv(tables / f"{key}.csv", index=False)

    balls = pressure_table(results, config)
    scored = balls[balls["state_ok"]]
    cut_common = float(scored.loc[scored["hp_common"], "li"].min())
    cuts = pd.DataFrame([
        {"flag": "hp_common", "innings": "both", "li_cut": cut_common},
        *[{"flag": "hp_within_innings", "innings": inn, "li_cut": float(scored.loc[(scored["innings"] == inn) & scored["hp_within_innings"], "li"].min())}
          for inn in (1, 2)],
    ])
    cuts.to_csv(tables / "li_cuts.csv", index=False)
    pv.li_summary(balls).to_csv(tables / "li_summary.csv", index=False)
    grid = pv.heatmap_grid(balls)
    grid.to_csv(tables / "li_by_over_wickets.csv", index=False)
    separation = pd.concat([pv.separation(balls, f) for f in pv.FLAGS], ignore_index=True)
    separation.to_csv(tables / "separation.csv", index=False)
    pv.concentration(balls).to_csv(tables / "concentration.csv", index=False)
    pv.leverage_drivers(balls).to_csv(tables / "leverage_drivers.csv", index=False)
    pv.flag_contrast(balls).to_csv(tables / "flag_contrast.csv", index=False)
    options = pv.innings_options(balls)
    for key, table in options.items():
        table.to_csv(tables / f"flag_options_{key}.csv", index=False)
    pv.knockout_check(balls, results["outcome_probs"], config).to_csv(tables / "knockout_check.csv", index=False)
    flag, exposure = config["pressure"]["high_pressure_flag"], config["pressure"]["exposure"]
    pv.phase_exposure(balls, flag).to_csv(tables / "exposure_by_innings_phase.csv", index=False)
    players = pd.concat([pv.player_exposure(balls, role, flag, exposure["min_balls"]) for role in ("batter", "bowler")],
                        ignore_index=True)
    players.to_csv(tables / "exposure_by_player.csv", index=False)
    dependence = pv.exposure_dependence(players)
    dependence.to_csv(tables / "exposure_dependence.csv", index=False)
    pv.team_exposure(balls, flag, exposure["min_team_matches"]).to_csv(tables / "exposure_by_team.csv", index=False)
    agreement = pv.rule_agreement(balls, share)
    agreement["summary"].to_csv(tables / "rule_vs_leverage_summary.csv", index=False)
    agreement["profile"].to_csv(tables / "rule_vs_leverage_profile.csv", index=False)

    picks = pv.pick_close_finishes(balls, match_table)
    picks[["category", "match_id", "comp_name", "match_date", "home_team", "away_team", "result", "total1", "total2"]].to_csv(
        tables / "close_finishes.csv", index=False)
    curves = [pv.match_curve(balls, int(mid)) for mid in picks["match_id"]]
    pd.concat(curves).to_csv(tables / "close_finish_curves.csv", index=False)
    titles = [f"{r.category}: {r.home_team} v {r.away_team}\n{r.comp_name}, {r.match_date}\n{r.result}" for r in picks.itertuples()]

    plots.plot_calibration(comparison["reliability"], figures / "wp_calibration.png")
    slice_cfg = config["pressure"]["slice"]
    plots.plot_leverage_slice(
        results["slices"], figures / "leverage_smoothness.png",
        f"Chase with {slice_cfg['balls_remaining']} balls left and {slice_cfg['wickets_in_hand']} wickets in hand "
        f"({config['pressure']['reference_context']['comp_type']} {config['pressure']['reference_context']['year']} conditions), one run at a time.",
    )
    plots.plot_li_distribution(balls, cut_common, figures / "li_distribution.png")
    plots.plot_li_heatmap(grid, figures / "li_heatmap_over_wickets.png", config["pressure"]["heatmap_min_balls"])
    definitions = {"hp_common": "top 25% of all balls by Leverage Index", "hp_within_innings": "top 25% of balls within each innings"}
    for name, text in definitions.items():
        plots.plot_separation(separation[separation["flag"] == name], figures / f"pressure_vs_ordinary_{name}.png", text)
    plots.plot_close_finishes(curves, titles, cut_common, figures / "close_finishes.png")
    plots.plot_role_shares(options["by_batting_position"], options["by_phase"], figures / "flag_options_roles.png")
    plots.plot_exposure(players, dependence, figures / "exposure_vs_innings_mix.png")

    keep = [
        "match_id", "innings", "innings_ball_num", "comp_type", "year", "overid", "phase", "batting_team_id", "bowling_team_id",
        "striker_id", "non_striker_id", "bowler_id", "batting_order", "balls_remaining", "score_before", "wickets_before",
        "runs_required", "env_rpo", "knock_out", "involves_associate", "rain_affected", "state_ok", "wp", "leverage", "li",
        "li_pct", "li_pct_innings", "hp_common", "hp_within_innings", "rule_index",
    ]
    balls[keep].to_csv(out_root / "ball_pressure.csv", index=False)
    logger.info("Phase 2b outputs written to %s and %s", tables, figures)
    return balls


def expectation_stage(balls: pd.DataFrame, config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Cross-fitted expectations: the full models, the same without the settledness features, and the plain average."""
    return {
        "full_model": crossfit_expectations(balls, config),
        "state_only": crossfit_expectations(balls, config, drop=tuple(config["value"]["settledness_features"])),
        "season_phase_average": baseline_expectations(balls),
    }


def _cost_examples(cf: CrossFit, name: str, fallback: pd.DataFrame, cap: float, factors: pd.DataFrame) -> pd.DataFrame:
    """The wicket cost in hand-written representative states (reference conditions), for a cricket-logic check.

    ``wicket_cost_before`` is the DP ratio (or fallback) alone; ``wicket_cost_runs`` adds the
    calibration factor of the reference season and the cap.
    """
    cfg, per_over = cf.config["value"], cf.config["ball_semantics"]["balls_per_over"]
    rows = []
    for example in cfg["cost_examples"]:
        year, states = _reference_states(cf, [{k: example[k] for k in ("balls_remaining", "x", "wickets_in_hand")}])
        model = cf.models[year][name]
        innings = example["innings"]
        cost = raw_cost(cost_parts(model, innings, states)).iloc[0]
        last_ball = example["balls_remaining"] <= 1
        settled = not last_ball and not cost["wp_per_run"] >= cfg["min_wp_per_run"]
        overs_left = min(int(np.ceil(example["balls_remaining"] / per_over)), int(fallback.index.max()))
        chosen = float(fallback.loc[overs_left, example["wickets_in_hand"]]) if settled else float(cost["c_raw"])
        chosen = 0.0 if last_ball else chosen
        factor = float(factors.loc[overs_left, year])
        rule = "last ball" if last_ball else "fallback (settled)" if settled else "capped" if chosen * factor > cap else "ratio"
        rows.append({
            "state": example["label"], "win_probability": float(win_probability(model, innings, states)[0]),
            "wp_lost_to_wicket": cost["wp_lost"], "wp_per_run": cost["wp_per_run"], "ratio": cost["c_raw"],
            "rule_used": rule, "wicket_cost_before": chosen, "calibration_factor": factor,
            "wicket_cost_runs": min(chosen * factor, cap),
        })
    return pd.DataFrame(rows)


def value_stage(
    results: dict[str, Any], pressure_balls: pd.DataFrame, config: dict[str, Any], cache_dir: Path, force: bool
) -> pd.DataFrame:
    """Phase 3: expectations, wicket exchange rate, run value per ball, player totals and every sanity check."""
    out_root = resolve_path(config["paths"]["outputs_dir"])
    tables, figures = out_root / "tables" / "phase3", out_root / "figures" / "phase3"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    cfg, chosen, flag = config["value"], config["pressure"]["wp_model"], config["pressure"]["high_pressure_flag"]
    per_over = config["ball_semantics"]["balls_per_over"]

    balls = add_situation_features(add_credit_columns(pressure_balls, config), config)
    fingerprint_audit(balls, config).to_csv(tables / "feature_fingerprint_audit.csv", index=False)
    variants = _cached(cache_dir / "phase3_expectations.pkl", force, lambda: expectation_stage(balls, config))
    held_out = variants["full_model"]
    vv.expectation_scores(balls, variants, config["models"]["prob_clip"]).to_csv(tables / "expectation_scores.csv", index=False)
    calibration = vv.calibration_table(balls, held_out, config["models"]["calibration_bins"])
    calibration.to_csv(tables / "expectation_calibration.csv", index=False)
    expected, factors = recalibrate_by_season(balls, held_out)
    factors.to_csv(tables / "expectation_season_factors.csv", index=False)
    vv.calibration_by_slice(balls, expected, flag).to_csv(tables / "expectation_calibration_by_slice.csv", index=False)

    parts = crossfit_cost_parts(balls, results["fold_models"], chosen)
    uncalibrated, _, cap_before = wicket_cost(balls, parts, config)
    cost_factors = crossfit_calibration(balls, uncalibrated["c_uncapped"], config)
    cost_factors.to_csv(tables / "wicket_cost_calibration_factors.csv")
    cost, fallback, cap = wicket_cost(balls, parts, config, cost_factors)
    cf = CrossFit(balls=results["scores"], models=results["fold_models"], outcome_probs=results["outcome_probs"],
                  config=config, outcome_models=results["outcome_models"])
    _cost_examples(cf, chosen, fallback, cap, cost_factors).to_csv(tables / "wicket_cost_examples.csv", index=False)
    fallback.rename_axis(index="overs_left", columns="wickets_in_hand").to_csv(tables / "wicket_cost_fallback_table.csv")
    vv.cost_summary(balls, cost, cap).to_csv(tables / "wicket_cost_summary.csv", index=False)

    valued = add_wpa(add_run_value(balls, expected, cost["c"]))
    valued["cost_rule"] = np.select([cost["last_ball"], cost["used_fallback"], cost["capped"]],
                                    ["last ball", "fallback", "capped"], default="ratio")
    grid = vv.cost_grid(valued, per_over)
    grid.to_csv(tables / "wicket_cost_by_overs_wickets.csv", index=False)
    vv.cost_event_study(valued, {"before": uncalibrated["c"], "after": cost["c"]}, config).to_csv(
        tables / "wicket_cost_event_study.csv", index=False)
    # Before/after the calibration: batting position and players (all else identical).
    old = add_run_value(balls, expected, uncalibrated["c"])
    valued["v_bat_before_calibration"] = old["v_bat"]
    valued["v_bowl_before_calibration"] = old["v_bowl"]
    vv.value_by_batting_position(valued, {"before": "v_bat_before_calibration", "after": "v_bat"}).to_csv(
        tables / "calibration_effect_by_batting_position.csv", index=False)
    faced = valued[valued["state_ok"] & valued["is_ball_faced"]]
    positions = faced.groupby("striker_id")["batting_order"].agg(lambda s: int(s.mode().iloc[0]))
    for role, column in (("batter", "v_bat"), ("bowler", "v_bowl")):
        before_rows = player_balls(old.assign(wpa=np.nan, li=valued["li"]), role, flag)
        after_table = overall_table(player_balls(valued, role, flag))
        vv.player_before_after(overall_table(before_rows), after_table, cfg["min_balls_overall"],
                               positions if role == "batter" else pd.Series(dtype=float)).to_csv(
            tables / f"calibration_effect_{role}s.csv", index=False)
    pd.DataFrame([{"cap_before_calibration": cap_before, "cap_after_calibration": cap}]).to_csv(
        tables / "wicket_cost_cap.csv", index=False)
    # The same season check without the recalibration, to show what it corrects.
    raw_value = add_run_value(balls, held_out, cost["c"])
    before = pd.concat([vv.zero_mean_table(player_balls(raw_value.assign(wpa=np.nan), role, flag)) for role in ("batter", "bowler")])
    before[before["slice"].isin(["overall", "season"])].to_csv(tables / "value_by_season_before_recalibration.csv", index=False)

    role_rows = {role: player_balls(valued, role, flag) for role in ("batter", "bowler")}
    players = {role: overall_table(rows) for role, rows in role_rows.items()}
    zero = pd.concat([vv.zero_mean_table(rows) for rows in role_rows.values()], ignore_index=True)
    zero.to_csv(tables / "value_zero_mean_checks.csv", index=False)
    pd.concat([vv.raw_gap_table(rows) for rows in role_rows.values()]).to_csv(tables / "raw_gap_v_value_gap.csv", index=False)
    pd.concat([vv.value_spread(rows) for rows in role_rows.values()]).to_csv(tables / "value_spread_by_phase.csv", index=False)
    floor = cfg["min_balls_overall"]
    pd.concat([vv.face_validity(p, floor) for p in players.values()]).to_csv(tables / "value_v_traditional.csv", index=False)
    pd.concat([vv.wpa_checks(role_rows[r], players[r], floor) for r in role_rows]).to_csv(tables / "wpa_checks.csv", index=False)
    for role, table in players.items():
        table.sort_values("value_per_100", ascending=False).to_csv(tables / f"player_overall_value_{role}s.csv", index=False)
        vv.extremes(table, floor, cfg["extremes_shown"]).to_csv(tables / f"overall_top_bottom_{role}s.csv", index=False)

    # Pressure split: built and checked for noise here; rankings wait for Phases 4-5.
    reliability = []
    for label, cells in (("phase", ["phase"]), ("phase_x_innings", ["phase", "innings"])):
        for role, rows in role_rows.items():
            split = pressure_split(rows, cells, config)
            reliability.append(vv.split_reliability(split, cfg["reliability_thresholds"]).assign(matched_on=label))
            if label == "phase":
                split.to_csv(tables / f"player_pressure_split_{role}s.csv", index=False)
    pd.concat(reliability, ignore_index=True).to_csv(tables / "pressure_split_reliability.csv", index=False)

    value_plots.plot_expectation_calibration(calibration, figures / "expectation_calibration.png")
    value_plots.plot_cost_grid(grid, figures / "wicket_cost_grid.png", config["pressure"]["heatmap_min_balls"])
    value_plots.plot_zero_mean(zero, figures / "value_zero_mean.png")
    value_plots.plot_face_validity(players["batter"], players["bowler"], figures / "value_v_traditional.png", floor)

    keep = [
        "match_id", "innings", "innings_ball_num", "comp_type", "year", "phase", "striker_id", "non_striker_id", "bowler_id",
        "balls_remaining", "wickets_in_hand", "runs_required", "striker_balls_faced", "free_hit", "in_powerplay", "state_ok",
        "li", flag, "wp", "wpa", "xr_bat", "xw_bat", "xw_ns", "xr_bowl", "xw_bowl", "c", "cost_rule",
        "bat_runs", "striker_out", "non_striker_out", "bowler_runs", "bowler_wicket", "v_bat", "v_ns", "v_bowl",
    ]
    valued[keep].to_csv(out_root / "ball_value.csv", index=False)
    logger.info("Phase 3 outputs written to %s and %s (wicket-cost cap %.2f runs)", tables, figures, cap)
    return valued


def fairness_stage(valued: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Phase 4: opposition and competition adjustment, its acceptance tests, and eligibility."""
    out_root = resolve_path(config["paths"]["outputs_dir"])
    tables, figures = out_root / "tables" / "phase4", out_root / "figures" / "phase4"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    cfg, flag = config["opposition"], config["pressure"]["high_pressure_flag"]
    balls, fits = adjust(valued, config)
    for role, fit in fits.items():
        fit.cv.assign(role=role, chosen_penalty=fit.penalty).to_csv(tables / f"opposition_cv_{role}.csv", index=False)
        fit.coefficients.to_csv(tables / f"opposition_coefficients_{role}.csv", index=False)
    gaps = fv.same_player_gaps(balls, cfg["same_player_comparisons"], cfg["same_player_min_balls"])
    gaps.to_csv(tables / "same_player_gaps.csv", index=False)
    floor = config["value"]["min_balls_overall"]
    moves = fv.overall_before_after(balls, flag, floor)
    for role, table in moves.items():
        table.to_csv(tables / f"overall_value_adjusted_{role}s.csv", index=False)
        table[table["player_id"].isin(cfg["test_cases"])].to_csv(tables / f"test_cases_{role}s.csv", index=False)
    fv.adjusted_means(balls).to_csv(tables / "adjusted_means.csv", index=False)
    fv.by_batting_position(balls).to_csv(tables / "adjusted_by_batting_position.csv", index=False)
    players = {role: overall_table(player_balls(balls, role, flag)) for role in ("batter", "bowler")}
    elig = fv.eligibility(balls, flag, players, config)
    elig.to_csv(tables / "eligibility.csv", index=False)
    plots.plot_same_player_gaps(gaps, figures / "same_player_gaps.png")
    # Fairness of the IPL split of the tiers for strong T20I-only players (D32).
    plain = {**config, "opposition": {**cfg, "split_groups_by_ipl": False}}
    unsplit, _ = adjust(valued, plain)
    named = list(config["rating"]["split_check_named"]) + [
        int(p) for p in balls.loc[balls["bowler_name"].isin(["Shaheen Shah Afridi", "Haris Rauf"]), "bowler_id"].unique()]
    rr.split_check(balls, unsplit, flag, config["rating"]["split_check_players"], floor, named).to_csv(
        tables / "tier_split_check_t20i_only.csv", index=False)
    keep = ["match_id", "innings", "innings_ball_num", "striker_id", "bowler_id", "batting_team", "bowling_team",
            "batting_tier", "bowling_tier", "batter_group", "bowler_group", "opp_part_batter", "opp_part_bowler",
            "v_bat", "v_bat_adj", "v_bowl", "v_bowl_adj"]
    balls[keep].to_csv(out_root / "ball_value_adjusted.csv", index=False)
    logger.info("Phase 4 outputs written to %s (%d eligible batters, %d eligible bowlers)", tables,
                int(((elig["role"] == "batter") & (elig["status"] == "eligible")).sum()),
                int(((elig["role"] == "bowler") & (elig["status"] == "eligible")).sum()))
    return balls


def rating_stage(balls: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Phase 5: leverage weighting chosen by split-half reliability, shrunk components, composite, bootstrap."""
    out_root = resolve_path(config["paths"]["outputs_dir"])
    tables, figures = out_root / "tables" / "phase5", out_root / "figures" / "phase5"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    cfg, flag = config["rating"], config["pressure"]["high_pressure_flag"]
    min_cell = config["value"]["baseline_min_cell_balls"]
    rng = np.random.default_rng(config["seed"])
    cap = float(balls.loc[balls["state_ok"], "li"].quantile(cfg["cap_quantile"]))
    config = {**config, "rating": {**cfg, "cap": cap}}
    elig = pd.read_csv(out_root / "tables" / "phase4" / "eligibility.csv")
    adjusted = balls.assign(v_bat=balls["v_bat_adj"], v_bowl=balls["v_bowl_adj"])

    prepared, reliability, base_rates = {}, [], []
    for role in ("batter", "bowler"):
        rows = player_balls(adjusted, role, flag)
        eligible = elig.loc[(elig["role"] == role) & (elig["status"] == "eligible"), "player_id"].to_numpy()
        agg = rt.match_aggregates(rows, cap, cfg["consistency_min_pressure_balls"])
        pm = rt.PlayerMatches.build(agg, eligible)
        prepared[role] = (rows, agg, pm)
        for weight in cfg["weightings"]:
            base = rt.base_rate(agg, weight)
            base_rates.append({"role": role, "weighting": weight, "base_rate": base,
                               "qualifying_player_matches": int(agg[f"q_{weight}"].sum())})
            reliability.append(rt.reliability_table(pm, weight, min_cell, base, cfg["split_reps"], rng).assign(role=role))
        logger.info("Phase 5: split-half reliability done for %ss", role)
    reliability = pd.concat(reliability, ignore_index=True)
    reliability.to_csv(tables / "split_half_reliability.csv", index=False)
    base_rates = pd.DataFrame(base_rates)
    base_rates.to_csv(tables / "consistency_base_rates.csv", index=False)
    chosen, choice = rt.choose_weighting({w: reliability[reliability["weighting"] == w] for w in cfg["weightings"]},
                                         cfg["weightings"])
    choice.assign(cap_value=cap).to_csv(tables / "weighting_choice.csv", index=False)
    logger.info("Phase 5: leverage weighting chosen = %s", chosen)

    effective, volume, rated_by_role, weights_by_role = [], [], {}, {}
    for role, (rows, agg, pm) in prepared.items():
        base = float(base_rates.loc[(base_rates["role"] == role) & (base_rates["weighting"] == chosen), "base_rate"].iloc[0])
        rel = reliability[(reliability["role"] == role) & (reliability["weighting"] == chosen)]
        result = rt.rate_role(pm, chosen, base, rel, config, rng)
        overall = overall_table(rows)
        overall = overall[overall["player_id"].isin(pm.players)]
        rated = rr.assemble(result["table"], rows, overall)
        rated.to_csv(tables / f"ratings_{role}s.csv", index=False)
        rated_by_role[role], weights_by_role[role] = rated, result["weights"]
        parts = rr.summary_rows(rated, cfg)
        parts["top"].to_csv(tables / f"draft_top10_{role}s.csv", index=False)
        parts["next"].to_csv(tables / f"ranks_11_15_{role}s.csv", index=False)
        rr.overall_v_pressure(rated, cfg["top_n"]).to_csv(tables / f"overall_v_pressure_{role}s.csv", index=False)
        effective.append(result["effective"].assign(role=role, weighting=chosen, base_rate=base))
        top = rated[rated["rank"] <= cfg["top_n"]]
        volume.append(result["volume"].assign(
            role=role, top_n_mean_share_balls_in_chase=top["share_balls_in_chase"].mean(),
            eligible_median_share_balls_in_chase=rated["share_balls_in_chase"].median(),
            top_n_mean_pressure_balls=top["pressure_balls"].mean(), eligible_median_pressure_balls=rated["pressure_balls"].median(),
            top_n_mean_p_top_n_equal_balls=top["p_top_n_equal_balls"].mean()))
        plots.plot_draft_top(rated, figures / f"draft_top_{role}s.png", role, cfg["top_n"] + cfg["show_next"])
    pd.concat(effective, ignore_index=True).to_csv(tables / "effective_weights.csv", index=False)
    pd.concat(volume, ignore_index=True).to_csv(tables / "volume_checks.csv", index=False)
    logger.info("Phase 5 outputs written to %s", tables)
    return {"rated": rated_by_role, "weights": weights_by_role, "weighting": chosen, "config": config}


def robustness_stage(balls: pd.DataFrame, phase5: dict[str, Any]) -> dict[str, Any]:
    """Phase 6: option B, tie-break, absences, evidence cards, sensitivity variants and stability."""
    config = phase5["config"]
    out_root = resolve_path(config["paths"]["outputs_dir"])
    tables, figures = out_root / "tables" / "phase6", out_root / "figures" / "phase6"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    cfg, top_n, flag = config["robustness"], config["rating"]["top_n"], config["pressure"]["high_pressure_flag"]
    names = {**balls.drop_duplicates("striker_id").set_index("striker_id")["striker_name"].to_dict(),
             **balls.drop_duplicates("bowler_id").set_index("bowler_id")["bowler_name"].to_dict()}
    variants = sens.build_variants(balls, config)
    ranks = sens.run_variants(variants, config, phase5["weighting"], phase5["weights"])
    ranks.to_csv(tables / "variant_ranks.csv", index=False)
    groups = sens.variant_groups(list(variants))
    counted = sens.counted_variants(groups)
    cuts = tuple(cfg["stability_cuts"])
    overlaps, finals, all_classes = [], {}, {}
    for role, rated in phase5["rated"].items():
        role_ranks = ranks[ranks["role"] == role]
        classes = sens.stability_classes(role_ranks, counted, top_n, cuts)
        all_classes[role] = classes
        classes.to_csv(tables / f"stability_classes_{role}s.csv")
        plain, _ = sens.apply_tie_break(rated, cfg["tie_break_zone"], cfg["tie_break_max_p_gap"])
        final, checks = sens.apply_tie_break(rated, cfg["tie_break_zone"], cfg["tie_break_max_p_gap"],
                                             classes["stability"], top_n)
        final = final.merge(plain[["player_id", "final_rank"]].rename(columns={"final_rank": "unguarded_rank"}), on="player_id")
        final.to_csv(tables / f"final_order_{role}s.csv", index=False)
        finals[role] = final
        checks.to_csv(tables / f"tie_break_checks_{role}s.csv", index=False)
        overlaps.append({"role": role, **sens.overlap_summary(final, top_n)})
        sens.absences(final, balls, role, cfg["absences"][role], cfg["overall_top_checked"], top_n, flag, names).to_csv(
            tables / f"notable_absences_{role}s.csv", index=False)
        for pid in cfg["evidence_cards"].get(role, []):
            sens.evidence_card(final, balls, pid, role, flag).to_csv(tables / f"evidence_card_{role}_{pid}.csv", index=False)
        table = sens.stability_table(final, role_ranks, classes, cfg["tie_break_zone"], groups)
        table.to_csv(tables / f"stability_{role}s.csv", index=False)
        plots.plot_stability(table, groups, figures / f"rank_stability_{role}s.png", role, top_n)
    pd.DataFrame(overlaps).to_csv(tables / "a_v_b_overlap.csv", index=False)
    logger.info("Phase 6 outputs written to %s", tables)
    return {"finals": finals, "classes": all_classes}


def final_stage(balls: pd.DataFrame, phase6: dict[str, Any], mapping: pd.DataFrame, config: dict[str, Any]) -> None:
    """Phase 7: the submission files, with registry names, player_id and teams on every row."""
    out_root = resolve_path(config["paths"]["outputs_dir"])
    tables = fo.final_tables(phase6["finals"], phase6["classes"], balls, fo.registry_names(mapping), config)
    fo.write(tables, out_root)


def report_stage(config: dict[str, Any]) -> None:
    """Phase 7: the report figures (from the output tables only), then the report's number cross-check."""
    out_root = resolve_path(config["paths"]["outputs_dir"])
    report_dir = out_root / config["report"]["dir"]
    ReportFigures(out_root, report_dir / "figures", config).all()
    if (report_dir / "methodology_report.md").exists():
        check_report.run(out_root, config)
        build_pdf.run(config)
    else:
        logger.warning("No methodology report yet at %s; cross-check skipped", report_dir)


def _cached(path: Path, force: bool, build: Any) -> Any:
    """Return the pickled result at ``path``, building and saving it first if needed."""
    if path.exists() and not force:
        logger.info("Reusing cache %s (use --force to recompute)", path)
        with open(path, "rb") as handle:
            return pickle.load(handle)
    result = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(result, handle)
    return result


def run(force: bool = False) -> None:
    """Run every stage; the two slow model steps are cached under outputs/cache unless ``force`` is set."""
    config = load_config()
    np.random.seed(config["seed"])
    run_audit(config)
    raw = load_raw_tables(config)
    match_table = build_match_table(raw, config)
    balls = build_ball_states(raw, match_table, config)
    cache_dir: Path = resolve_path(config["paths"]["outputs_dir"]) / "cache"
    results = _cached(cache_dir / "phase2_model_stage.pkl", force, lambda: model_stage(balls, match_table, config))
    results.update(_cached(
        cache_dir / "phase2_ablations.pkl", force,
        lambda: _ablations(balls, match_table, config, config["pressure"]["wp_model"], results["scores"]),
    ))
    pressure_balls = validation_stage(results, match_table, config)
    valued = value_stage(results, pressure_balls, config, cache_dir, force)
    adjusted = fairness_stage(valued, config)
    phase5 = rating_stage(adjusted, config)
    phase6 = robustness_stage(adjusted, phase5)
    final_stage(adjusted, phase6, raw.player_mapping, phase5["config"])
    report_stage(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute the cached model stage")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(force=parser.parse_args().force)
