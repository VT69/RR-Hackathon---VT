"""Phase 1 data audit. Run with ``python -m src.data.audit`` from the project root.

Writes ``outputs/audit/facts.json`` (every number quoted in ``outputs/data_audit.md``)
and a handful of CSV tables next to it. Nothing here changes the data.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, resolve_path
from src.data.cleaning import add_ball_flags, classify_competitions
from src.data.loading import RawTables, load_raw_tables

logger = logging.getLogger(__name__)

MARGIN_PATTERN = re.compile(r"(\d+)\s*(run|wicket|wkt)", re.IGNORECASE)
DLS_PATTERN = re.compile(r"dls|d/l", re.IGNORECASE)


def inventory(raw: RawTables) -> dict[str, Any]:
    """Row/column counts, null counts and exact-duplicate rows per table."""
    out = {}
    for name, frame in vars(raw).items():
        nulls = frame.isna().sum()
        out[name] = {
            "rows": int(len(frame)),
            "cols": int(frame.shape[1]),
            "duplicate_rows": int(frame.duplicated().sum()),
            "null_counts": {k: int(v) for k, v in nulls[nulls > 0].items()},
        }
    return out


def key_checks(raw: RawTables) -> dict[str, Any]:
    """Primary keys and referential integrity between tables."""
    balls, matches, xi, mapping = raw.balls, raw.matches, raw.playing_eleven, raw.player_mapping
    ball_key = ["match_id", "innings", "innings_ball_num"]
    seq = balls.groupby(["match_id", "innings"])["innings_ball_num"].agg(["min", "max", "count"])
    ball_ids = set(balls["striker_id"]) | set(balls["non_striker_id"]) | set(balls["bowler_id"])
    return {
        "match_id_unique": bool(matches["match_id"].is_unique),
        "ball_key_duplicates": int(balls.duplicated(ball_key).sum()),
        "innings_with_ball_gaps": int(((seq["max"] - seq["min"] + 1) != seq["count"]).sum()),
        "file_already_in_ball_order": bool(
            balls.sort_values(ball_key, kind="stable").index.equals(balls.index)
        ),
        "matches_total": int(len(matches)),
        "matches_with_balls": int(balls["match_id"].nunique()),
        "ball_matches_missing_from_matches": int((~balls["match_id"].isin(matches["match_id"])).sum()),
        "matches_missing_from_xi": sorted(set(matches["match_id"]) - set(xi["match_id"])),
        "ball_player_ids_missing_from_mapping": len(ball_ids - set(mapping["player_id"])),
        "xi_player_ids_missing_from_mapping": len(set(xi["player_id"]) - set(mapping["player_id"])),
        "xi_duplicate_match_player_rows": int(xi.duplicated(["match_id", "player_id"]).sum()),
        "xi_matches_with_duplicates": sorted(
            xi.loc[xi.duplicated(["match_id", "player_id"]), "match_id"].unique().tolist()
        ),
    }


def coverage(balls: pd.DataFrame, matches: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Matches and balls by competition type and year, plus T20I opposition mix."""
    per_match = balls.groupby("match_id").size().rename("balls")
    m = matches.merge(per_match, on="match_id", how="left").fillna({"balls": 0})
    by = m.groupby(["comp_type", "year"]).agg(matches=("match_id", "size"), balls=("balls", "sum"))
    full = set(config["competitions"]["full_members"])
    t20i = m[m["comp_type"] == "T20I"].copy()
    n_full = t20i["home_team"].isin(full).astype(int) + t20i["away_team"].isin(full).astype(int)
    t20i["opposition_mix"] = n_full.map({2: "full_v_full", 1: "full_v_associate", 0: "associate_v_associate"})
    mix = t20i.groupby("opposition_mix").agg(matches=("match_id", "size"), balls=("balls", "sum"))
    teams = pd.concat(
        [
            m[["comp_type", "home_team"]].set_axis(["comp_type", "team"], axis=1),
            m[["comp_type", "away_team"]].set_axis(["comp_type", "team"], axis=1),
        ]
    )
    return {
        "date_range": [m["match_date"].min(), m["match_date"].max()],
        "by_type_year": by.astype(int).reset_index().to_dict("records"),
        "other_t20_competitions": sorted(m.loc[m["comp_type"] == "OtherT20", "comp_name"].unique()),
        "world_cup_competitions": sorted(
            m.loc[m["comp_name"].str.contains("world", case=False), "comp_name"].unique()
        ),
        "t20i_opposition_mix": mix.astype(int).reset_index().to_dict("records"),
        "team_match_counts": teams.value_counts().reset_index(name="matches"),
        "stage_by_type": pd.crosstab(m["stage"], [m["comp_type"], m["knock_out"]]),
    }


def ball_semantics_checks(flagged: pd.DataFrame) -> dict[str, Any]:
    """Evidence for the wide / no-ball / extras / dismissal conventions."""
    reg = flagged[~flagged["is_super_over"]]
    overs = reg.groupby(["match_id", "innings", "overid"])["is_legal"].sum().reset_index()
    last_over = overs.groupby(["match_id", "innings"])["overid"].transform("max")
    completed = overs[overs["overid"] < last_over]
    wk = flagged[flagged["dismissed_player_id"].notna()]
    who = np.where(wk["dismissed_player_id"] == wk["striker_id"], "striker", "non_striker")
    extras = flagged["extras_type"].fillna("none")
    return {
        "extras_type_counts": extras.value_counts().to_dict(),
        "completed_overs": int(len(completed)),
        "completed_overs_with_6_legal": int((completed["is_legal"] == 6).sum()),
        "completed_overs_not_6": completed[completed["is_legal"] != 6].to_dict("records"),
        "wides_with_bat_runs": int((flagged["is_wide"] & (flagged["runs"] > 0)).sum()),
        "byes_with_bat_runs": int((extras.isin(["byes", "legbyes"]) & (flagged["runs"] > 0)).sum()),
        "untyped_rows_with_extras": int(
            (extras.eq("none") & ((flagged["extras1_runs"] + flagged["extras_2_runs"]) > 0)).sum()
        ),
        "dismissal_type_counts": flagged["dismissal_type"].value_counts().to_dict(),
        "dismissed_who_by_type": pd.crosstab(wk["dismissal_type"].fillna("retired_hurt"), who)
        .reset_index()
        .to_dict("records"),
        "wicket_flag_without_dismissed_id": int((flagged["is_wicket"] & flagged["dismissed_player_id"].isna()).sum()),
        "retired_hurt_rows": int((~flagged["is_wicket"] & flagged["dismissed_player_id"].notna()).sum()),
        "wickets_on_wides": int((flagged["is_wicket"] & flagged["is_wide"]).sum()),
        "wickets_on_noballs": int((flagged["is_wicket"] & flagged["is_noball"]).sum()),
        "non_boundary_fours_or_sixes": int(
            ((flagged["runs"] == 4) & ~flagged["is_four"]).sum()
            + ((flagged["runs"] == 6) & ~flagged["is_six"]).sum()
        ),
        "overs_with_two_bowlers": int(
            (reg.groupby(["match_id", "innings", "overid"])["bowler_id"].nunique() > 1).sum()
        ),
        "bowler_innings_over_24_legal_balls": int(
            (reg.groupby(["match_id", "innings", "bowler_id"])["is_legal"].sum() > 24).sum()
        ),
        "bowl_speed_missing_share": float((flagged["bowl_speed"] <= 0).mean()),
    }


def _parse_margin(result: str) -> tuple[float, str | None]:
    found = MARGIN_PATTERN.search(result)
    if not found:
        return np.nan, None
    return float(found.group(1)), "runs" if found.group(2).lower() == "run" else "wickets"


def innings_summary(flagged: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """One row per match with first/second-innings totals, wickets and legal balls."""
    reg = flagged[~flagged["is_super_over"]]
    inn = reg.groupby(["match_id", "innings"]).agg(
        total=("total_runs", "sum"), wickets=("is_wicket", "sum"), legal=("is_legal", "sum")
    )
    wide = inn.unstack("innings")
    wide.columns = [f"{stat}{innings}" for stat, innings in wide.columns]
    out = matches.merge(wide.reset_index(), on="match_id", how="left")
    out["has_super_over_balls"] = out["match_id"].isin(flagged.loc[flagged["is_super_over"], "match_id"])
    margins = out["result"].apply(_parse_margin)
    out["margin"] = [m[0] for m in margins]
    out["margin_type"] = [m[1] for m in margins]
    revised = out[["first_inn_revised_overs", "second_inn_revised_overs", "revised_target"]].notna().any(axis=1)
    out["dls_or_revised"] = revised | out["result"].str.contains(DLS_PATTERN)
    return out


def reconciliation(summary: pd.DataFrame) -> dict[str, Any]:
    """Do ball-level totals reproduce the official results? (Checks the run semantics.)"""
    # Tied matches are excluded: their result text quotes the super-over margin, not the match margin.
    level = summary["total1"] == summary["total2"]
    clean = summary[~summary["dls_or_revised"] & summary["total2"].notna() & ~level]
    by_runs = clean[clean["margin_type"] == "runs"]
    by_wkts = clean[clean["margin_type"] == "wickets"]
    runs_ok = (by_runs["total1"] - by_runs["total2"]) == by_runs["margin"]
    wkts_ok = (10 - by_wkts["wickets2"]) == by_wkts["margin"]
    return {
        "run_margin_matches": int(len(by_runs)),
        "run_margin_exact": int(runs_ok.sum()),
        "run_margin_mismatch_ids": by_runs.loc[~runs_ok, "match_id"].tolist(),
        "wicket_margin_matches": int(len(by_wkts)),
        "wicket_margin_exact": int(wkts_ok.sum()),
        "wicket_margin_mismatch_ids": by_wkts.loc[~wkts_ok, "match_id"].tolist(),
    }


def match_status(summary: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """Abandoned / no-result / tied / rain-affected / reduced-overs classification."""
    max_legal = config["ball_semantics"]["scheduled_overs"] * config["ball_semantics"]["balls_per_over"]
    s = summary
    no_balls = s["total1"].isna()
    first_only = s["total1"].notna() & s["total2"].isna()
    no_winner_two_inn = s["total2"].notna() & s["win_team_id"].isna()
    tied = s["total2"].notna() & (s["total1"] == s["total2"])
    dls_no_target = s["result"].str.contains(DLS_PATTERN) & s["revised_target"].isna()
    short_first = (
        ~s["dls_or_revised"] & s["total2"].notna() & (s["legal1"] <= max_legal - 6) & (s["wickets1"] < 10)
    )
    return {
        "no_ball_data": int(no_balls.sum()),
        "first_innings_only": int(first_only.sum()),
        "two_innings_no_winner_ids": s.loc[no_winner_two_inn, "match_id"].tolist(),
        "tied_after_regular_innings_ids": s.loc[tied, "match_id"].tolist(),
        "tied_with_super_over_balls_ids": s.loc[tied & s["has_super_over_balls"], "match_id"].tolist(),
        "any_revised_field_or_dls_label": int(s["dls_or_revised"].sum()),
        "with_revised_target": int(s["revised_target"].notna().sum()),
        "dls_label_but_no_revised_target_ids": s.loc[dls_no_target, "match_id"].tolist(),
        "reduced_overs_unflagged_ids": s.loc[short_first, "match_id"].tolist(),
        "completed_two_innings_with_winner": int((s["total2"].notna() & s["win_team_id"].notna()).sum()),
        "knockout_matches": int((s["knock_out"] == 1).sum()),
    }


def player_identity(raw: RawTables, matches: pd.DataFrame) -> dict[str, Any]:
    """ID/name consistency: the pack has no players table, so the mapping file is the registry."""
    balls, mapping = raw.balls, raw.player_mapping
    roles = [("striker_id", "striker_name"), ("non_striker_id", "non_striker_name"), ("bowler_id", "bowler_name")]
    ball_names = pd.concat([balls[[i, n]].set_axis(["player_id", "name"], axis=1) for i, n in roles]).drop_duplicates()
    registry = mapping[["player_id", "player_name"]].drop_duplicates()
    joined = ball_names.merge(registry, on="player_id", how="left")
    shared = registry.groupby("player_name")["player_id"].nunique()
    shared_names = registry[registry["player_name"].isin(shared[shared > 1].index)]
    comp_type = balls[["match_id", "striker_id"]].merge(matches[["match_id", "comp_type"]], on="match_id")
    in_both = comp_type[comp_type["comp_type"].isin(["IPL", "T20I"])].groupby("striker_id")["comp_type"].nunique()
    return {
        "registry_player_ids": int(registry["player_id"].nunique()),
        "registry_distinct_names": int(registry["player_name"].nunique()),
        "ball_data_player_ids": int(ball_names["player_id"].nunique()),
        "ids_with_multiple_names_in_registry": int((registry.groupby("player_id")["player_name"].nunique() > 1).sum()),
        "ids_with_multiple_names_in_balls": int((ball_names.groupby("player_id")["name"].nunique() > 1).sum()),
        "ball_name_differs_from_registry": joined.loc[joined["name"] != joined["player_name"]].to_dict("records"),
        "names_shared_by_two_ids": shared_names.sort_values("player_name").to_dict("records"),
        "batters_seen_in_both_ipl_and_t20i": int((in_both > 1).sum()),
    }


def playing_eleven_checks(raw: RawTables, matches: pd.DataFrame) -> dict[str, Any]:
    """Playing XI sizes and Impact Player usage."""
    xi = raw.playing_eleven.merge(matches[["match_id", "comp_type", "year"]], on="match_id")
    played = xi[xi["is_played"]]
    sizes = played.groupby(["comp_type", "match_id", "team_id"]).size().reset_index(name="n")
    impact = xi.groupby(["comp_type", "year"])[["is_impact", "is_impact_off"]].sum()
    played_pairs = set(zip(played["match_id"], played["player_id"]))
    ball_pairs = set(zip(raw.balls["match_id"], raw.balls["striker_id"])) | set(
        zip(raw.balls["match_id"], raw.balls["bowler_id"])
    )
    return {
        "team_sizes": pd.crosstab(sizes["comp_type"], sizes["n"]),
        "impact_subs": impact[impact["is_impact"] > 0].reset_index().to_dict("records"),
        "batter_or_bowler_not_in_played_xi": len(ball_pairs - played_pairs),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Not JSON serialisable: {type(value)}")


def run_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Run every check, write tables + ``facts.json`` and return the facts."""
    raw = load_raw_tables(config)
    matches = classify_competitions(raw.matches, raw.competitions, config)
    flagged = add_ball_flags(raw.balls, config)
    summary = innings_summary(flagged, matches)

    out_dir: Path = resolve_path(config["paths"]["outputs_dir"]) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    cov = coverage(raw.balls, matches, config)
    cov.pop("team_match_counts").to_csv(out_dir / "team_match_counts.csv", index=False)
    cov.pop("stage_by_type").to_csv(out_dir / "stage_by_type.csv")
    xi = playing_eleven_checks(raw, matches)
    xi.pop("team_sizes").to_csv(out_dir / "playing_xi_team_sizes.csv")
    summary.to_csv(out_dir / "match_innings_summary.csv", index=False)

    facts = {
        "inventory": inventory(raw),
        "keys": key_checks(raw),
        "coverage": cov,
        "ball_semantics": ball_semantics_checks(flagged),
        "reconciliation": reconciliation(summary),
        "match_status": match_status(summary, config),
        "player_identity": player_identity(raw, matches),
        "playing_eleven": xi,
    }
    with open(out_dir / "facts.json", "w", encoding="utf-8") as handle:
        json.dump(facts, handle, indent=2, default=_json_default)
    logger.info("Audit written to %s", out_dir)
    return facts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_audit(load_config())
