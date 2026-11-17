"""The opposition model: tiers, groups, and recovery of a known opponent effect on synthetic balls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.rating.opposition import build_design, home_groups, match_folds, side_tier, solve

CONFIG = load_config()


def test_side_tier_labels_ipl_and_the_three_t20i_tiers() -> None:
    teams = pd.Series(["Mumbai Indians", "India", "Zimbabwe", "Nepal"])
    comps = pd.Series(["IPL", "T20I", "T20I", "T20I"])
    assert side_tier(teams, comps, CONFIG).tolist() == ["IPL", "T20I top-8", "T20I other full member", "T20I associate"]


def test_home_group_is_the_main_t20i_tier_split_by_ipl_participation() -> None:
    balls = pd.DataFrame({
        "striker_id": [1, 1, 2, 3],
        "comp_type": ["IPL", "T20I", "T20I", "IPL"],
        "batting_tier": ["IPL", "T20I top-8", "T20I top-8", "IPL"],
    })
    groups = home_groups(balls, "striker_id", "batting_tier", split_by_ipl=True)
    assert groups[1] == "T20I top-8 (also IPL)" and groups[2] == "T20I top-8" and groups[3] == "IPL only"


def test_whole_matches_stay_in_one_fold() -> None:
    ids = pd.Series([5, 5, 5, 9, 9, 11])
    folds = match_folds(ids, 2, seed=0)
    assert len(set(folds[:3])) == 1 and folds[3] == folds[4]


def test_a_weak_tier_is_recovered_and_does_not_leak_into_player_effects() -> None:
    """Batters 0-19 face a strong tier and a weak tier equally; the weak tier gives +0.3 runs a ball."""
    rng = np.random.default_rng(0)
    n = 20000
    rows = pd.DataFrame({
        "striker_id": rng.integers(0, 20, n), "bowler_id": rng.integers(100, 140, n), "match_id": rng.integers(0, 400, n),
    })
    rows["bowling_tier"] = np.where(rows["bowler_id"] < 120, "IPL", "T20I associate")
    skill = rng.normal(0, 0.1, 20)
    rows["v"] = skill[rows["striker_id"]] + np.where(rows["bowling_tier"] == "IPL", 0.0, 0.3) + rng.normal(0, 1, n)
    groups = {"striker_id": pd.Series("IPL only", index=range(20)), "bowler_id": pd.Series("IPL only", index=range(100, 140))}
    design = build_design(rows, "striker_id", "bowler_id", "bowling_tier", groups)
    beta = solve(design, rows["v"].to_numpy(), lam=1000.0)
    effects = design.columns.assign(effect=beta)
    tier = effects.loc[(effects["kind"] == "opponent tier") & (effects["key"] == "T20I associate"), "effect"].iloc[0]
    bowlers = effects.loc[effects["kind"] == "opponent", "effect"]
    assert abs(tier - 0.3) < 0.05
    assert bowlers.abs().max() < 0.05          # the tier effect is not smeared over the individual bowlers
