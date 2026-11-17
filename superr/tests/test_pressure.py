"""Leverage Index, pressure flags and the rule-based index."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import load_config
from src.features.pressure import add_pressure_columns
from src.features.rule_index import rule_pressure_index

CONFIG = load_config()


def _balls(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    innings = np.repeat([1, 2], n // 2)
    return pd.DataFrame(
        {
            "innings": innings,
            "state_ok": True,
            # The chase is given systematically higher leverage, as in the real data.
            "leverage": rng.gamma(2.0, 1.0, n) * np.where(innings == 1, 0.5, 1.5),
            "overid": rng.integers(1, 21, n),
        }
    )


def test_leverage_index_averages_one() -> None:
    out = add_pressure_columns(_balls(), CONFIG)
    assert np.isclose(out["li"].mean(), 1.0)


def test_both_flags_mark_the_configured_share() -> None:
    out = add_pressure_columns(_balls(), CONFIG)
    share = CONFIG["pressure"]["high_pressure_share"]
    assert np.isclose(out["hp_common"].mean(), share, atol=0.01)
    assert np.isclose(out["hp_within_innings"].mean(), share, atol=0.01)


def test_within_innings_flag_gives_each_innings_its_own_quota() -> None:
    out = add_pressure_columns(_balls(), CONFIG)
    share = CONFIG["pressure"]["high_pressure_share"]
    by_innings = out.groupby("innings")["hp_within_innings"].mean()
    assert np.allclose(by_innings, share, atol=0.01)
    assert out.groupby("innings")["hp_common"].mean().loc[2] > share  # common scale favours the chase


def test_unscored_balls_are_never_high_pressure() -> None:
    balls = _balls()
    balls.loc[0, ["state_ok", "leverage"]] = [False, np.nan]
    out = add_pressure_columns(balls, CONFIG)
    assert not out.loc[0, "hp_common"] and not out.loc[0, "hp_within_innings"]


def _rule_row(**overrides: float) -> pd.DataFrame:
    base = {
        "innings": 2, "legal_before": 90, "scheduled_balls": 120, "wickets_before": 5, "rrr_rel": 1.2,
        "score_before": 120, "env_rpo": 9.0, "runs_required": 54, "balls_remaining": 30, "wickets_in_hand": 5,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_rule_index_is_bounded_and_zero_for_a_lost_chase() -> None:
    live = rule_pressure_index(_rule_row(), CONFIG)[0]
    lost = rule_pressure_index(_rule_row(rrr_rel=3.0, runs_required=135), CONFIG)[0]
    assert 0 < live <= 1
    assert lost == 0


def test_rule_index_rises_with_the_required_rate() -> None:
    easy = rule_pressure_index(_rule_row(rrr_rel=0.9), CONFIG)[0]
    hard = rule_pressure_index(_rule_row(rrr_rel=1.6), CONFIG)[0]
    assert hard > easy
