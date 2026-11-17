"""Cricket-correct derived columns for the ball-by-ball table.

These are the rules that are easy to get silently wrong, so they live in one
place and are unit-tested (``tests/test_ball_flags.py``). Each rule was verified
against the data in ``outputs/data_audit.md``.

Raw column semantics (verified):
  * ``runs``          - runs off the bat, credited to the striker.
  * ``extras1_runs``  - the primary extra: wides (incl. any run on the wide),
                        the 1-run no-ball penalty, byes, leg-byes or penalty runs.
  * ``extras_2_runs`` - byes/leg-byes run on top of a no-ball.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def add_ball_flags(balls: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Return a copy of ``balls`` with cricket-correct helper columns.

    Added columns:
      * ``is_wide``, ``is_noball``
      * ``is_legal``      - counts towards the six balls of an over.
      * ``is_ball_faced`` - counts as a ball faced by the striker (no-balls do, wides do not).
      * ``total_runs``    - everything added to the team total by this delivery.
      * ``batter_runs``   - runs credited to the striker.
      * ``bowler_runs``   - runs charged to the bowler (bat + wides + no-ball penalty;
                            never byes, leg-byes or penalty runs).
      * ``is_bowler_wicket`` - dismissal credited to the bowler.
      * ``is_super_over`` - innings 3+.
    """
    sem = config["ball_semantics"]
    out = balls.copy()
    extras = out["extras_type"].fillna("")

    out["is_wide"] = extras.eq(sem["wide_label"])
    out["is_noball"] = extras.isin(sem["noball_labels"])
    out["is_legal"] = ~(out["is_wide"] | out["is_noball"])
    out["is_ball_faced"] = ~out["is_wide"]

    out["total_runs"] = out["runs"] + out["extras1_runs"] + out["extras_2_runs"]
    out["batter_runs"] = out["runs"]
    bowler_extras = np.where(out["is_wide"] | out["is_noball"], out["extras1_runs"], 0)
    out["bowler_runs"] = out["runs"] + bowler_extras

    out["is_bowler_wicket"] = out["is_wicket"] & out["dismissal_type"].isin(
        sem["bowler_wicket_types"]
    )
    out["is_super_over"] = ~out["innings"].isin(sem["regular_innings"])
    return out


def classify_competitions(
    matches: pd.DataFrame, competitions: pd.DataFrame, config: dict[str, Any]
) -> pd.DataFrame:
    """Attach competition info to matches and label each as IPL / T20I / OtherT20.

    ``match_format`` alone is not enough: the two A-team Asia Cup events share the
    ``T20`` format label with the IPL but are neither IPL nor full internationals.
    """
    cfg = config["competitions"]
    merged = matches.merge(competitions, on="comp_id", how="left", validate="many_to_one")
    is_ipl = merged["comp_name"].str.startswith(cfg["ipl_name_prefix"])
    is_t20i = merged["match_format"].eq(cfg["t20i_format"])
    merged["comp_type"] = np.select([is_ipl, is_t20i], ["IPL", "T20I"], default="OtherT20")
    merged["year"] = merged["match_date"].str[:4].astype(int)
    return merged
