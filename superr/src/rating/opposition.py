"""Opposition and competition strength: a "who faced whom" ridge model over every ball.

Batting fit (one row per delivery to a striker):

    v_bat = intercept + opponent tier + batter's group + bowler's group + batter + bowler + noise

Bowling fit (one row per delivery) mirrors it with ``v_bowl``.

* **Opponent tier** (ball level): the tier of the side the player is facing on that ball -
  IPL, T20I top-8, other full member or associate. It carries the strength of the
  competition environment.
* **Group** (player level): the player's home tier - the tier of the side they play
  T20Is for, or "IPL only" for players with no T20I in the data.
* **Batter / bowler**: individual effects, pulled towards zero by a ridge penalty. Because
  tiers and groups are not penalised, each player is pulled towards their own group's
  level, not the global average (DECISIONS.md D29).

The model is used only for the **opponent's** part on each ball: a player's adjusted value
adds back what their opponent and the opposition tier are worth, so the player's own
rating is still built from their own balls and shrunk once, in Phase 5.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

logger = logging.getLogger(__name__)

IPL_TIER = "IPL"
REFERENCE_GROUP = "IPL only"
TIERS: tuple[str, ...] = ("IPL", "T20I top-8", "T20I other full member", "T20I associate")


def side_tier(team: pd.Series, comp_type: pd.Series, config: dict[str, Any]) -> pd.Series:
    """Tier of a side on each ball: IPL, or the T20I tier of the team."""
    cfg = config["competitions"]
    top, full = set(cfg["top_tier"]), set(cfg["full_members"])
    t20i = np.where(team.isin(top), TIERS[1], np.where(team.isin(full), TIERS[2], TIERS[3]))
    return pd.Series(np.where(comp_type.eq("IPL"), IPL_TIER, t20i), index=team.index)


def add_tiers(balls: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Add team names and tiers of the batting and bowling sides."""
    out = balls.copy()
    home = out["batting_team_id"] == out["home_team_id"]
    out["batting_team"] = np.where(home, out["home_team"], out["away_team"])
    out["bowling_team"] = np.where(home, out["away_team"], out["home_team"])
    out["batting_tier"] = side_tier(out["batting_team"], out["comp_type"], config)
    out["bowling_tier"] = side_tier(out["bowling_team"], out["comp_type"], config)
    return out


def home_groups(balls: pd.DataFrame, player: str, tier: str, split_by_ipl: bool) -> pd.Series:
    """Each player's home tier: the T20I tier they appear in most, else "IPL only".

    With ``split_by_ipl`` a T20I tier is split into players who also play in the IPL and
    those who do not, so that the many T20I-only players of a tier do not pull the
    level of the IPL-linked players (who carry the IPL-T20I comparison) towards them.
    """
    t20i = balls[balls["comp_type"] == "T20I"]
    main = t20i.groupby(player)[tier].agg(lambda s: s.mode().iloc[0])
    players = balls[player].unique()
    groups = pd.Series(main.reindex(players).fillna(REFERENCE_GROUP).to_numpy(), index=players)
    if split_by_ipl:
        in_ipl = set(balls.loc[balls["comp_type"] == "IPL", player])
        linked = groups.ne(REFERENCE_GROUP) & groups.index.isin(in_ipl)
        groups[linked] = groups[linked] + " (also IPL)"
    return groups


@dataclass
class Design:
    """Sparse design matrix plus the meaning of every column."""

    matrix: sparse.csr_matrix
    columns: pd.DataFrame          # kind, key, penalised
    n_rows: int


def _block(codes: np.ndarray, n_levels: int, n_rows: int) -> sparse.csr_matrix:
    ok = codes >= 0
    return sparse.csr_matrix((np.ones(ok.sum()), (np.flatnonzero(ok), codes[ok])), shape=(n_rows, n_levels))


def build_design(rows: pd.DataFrame, player: str, opponent: str, opp_tier: str,
                 groups: dict[str, pd.Series]) -> Design:
    """Columns: intercept, opponent tier (IPL is the reference), the two players' groups
    ("IPL only" is the reference), then one column per player and per opponent."""
    n = len(rows)
    blocks, meta = [sparse.csr_matrix(np.ones((n, 1)))], [("intercept", "intercept", False)]
    tiers = [t for t in TIERS if t != IPL_TIER]
    blocks.append(_block(pd.Categorical(rows[opp_tier], categories=tiers).codes, len(tiers), n))
    meta += [("opponent tier", t, False) for t in tiers]
    for role, column in (("player group", player), ("opponent group", opponent)):
        levels = sorted(g for g in groups[column].unique() if g != REFERENCE_GROUP)
        codes = pd.Categorical(rows[column].map(groups[column]), categories=levels).codes
        blocks.append(_block(codes, len(levels), n))
        meta += [(role, g, False) for g in levels]
    for role, column in (("player", player), ("opponent", opponent)):
        ids = np.sort(rows[column].unique())
        codes = pd.Categorical(rows[column], categories=ids).codes
        blocks.append(_block(codes, len(ids), n))
        meta += [(role, int(i), True) for i in ids]
    columns = pd.DataFrame(meta, columns=["kind", "key", "penalised"])
    return Design(sparse.hstack(blocks, format="csr"), columns, n)


def solve(design: Design, y: np.ndarray, lam: float, rows: np.ndarray | None = None, floor: float = 1e-6) -> np.ndarray:
    """Ridge solution with a penalty on player columns only (a tiny floor elsewhere keeps it stable)."""
    x = design.matrix if rows is None else design.matrix[rows]
    target = y if rows is None else y[rows]
    gram = (x.T @ x).toarray()
    penalty = np.where(design.columns["penalised"].to_numpy(), lam, floor)
    return np.linalg.solve(gram + np.diag(penalty), x.T @ target)


def match_folds(match_ids: pd.Series, k: int, seed: int) -> np.ndarray:
    """Fold number per row, assigning whole matches to folds at random."""
    unique = np.sort(match_ids.unique())
    fold_of = dict(zip(unique, np.random.default_rng(seed).permutation(len(unique)) % k))
    return match_ids.map(fold_of).to_numpy()


def cross_validate(design: Design, y: np.ndarray, folds: np.ndarray, grid: list[float]) -> pd.DataFrame:
    """Held-out mean squared error for each penalty, plus two reference models.

    "no player effects" keeps only the intercept, tiers and groups; "intercept only" is
    the league-average baseline. Matches never appear in both training and test rows.
    """
    kinds = design.columns["kind"]
    no_players = Design(design.matrix[:, np.flatnonzero(~kinds.isin(["player", "opponent"]).to_numpy())],
                        design.columns[~kinds.isin(["player", "opponent"])].reset_index(drop=True), design.n_rows)
    out = []
    for label, lam, d in [("intercept only", np.nan, None), ("no player effects", np.nan, no_players),
                          *[(f"ridge, penalty {g:g}", g, design) for g in grid]]:
        errors, count = 0.0, 0
        for fold in np.unique(folds):
            test, train = folds == fold, folds != fold
            if d is None:
                pred = np.full(test.sum(), y[train].mean())
            else:
                beta = solve(d, y, 0.0 if np.isnan(lam) else lam, rows=train)
                pred = d.matrix[test] @ beta
            errors += float(((y[test] - pred) ** 2).sum())
            count += int(test.sum())
        out.append({"model": label, "penalty": lam, "held_out_mse": errors / count})
        logger.info("Opposition model CV: %s -> %.5f", label, errors / count)
    table = pd.DataFrame(out)
    table["gain_v_intercept_only"] = table.loc[0, "held_out_mse"] - table["held_out_mse"]
    return table


@dataclass
class OppositionFit:
    """One fitted role: coefficients, chosen penalty, cross-validation table and per-ball opponent part."""

    role: str
    penalty: float
    coefficients: pd.DataFrame
    cv: pd.DataFrame
    opponent_part: pd.Series


def fit_role(rows: pd.DataFrame, role: str, value: str, groups: dict[str, pd.Series], config: dict[str, Any]) -> OppositionFit:
    """Fit one role, choosing the penalty by match-level cross-validation, then refit on all balls."""
    cfg = config["opposition"]
    player, opponent, opp_tier = ("striker_id", "bowler_id", "bowling_tier") if role == "batter" else \
        ("bowler_id", "striker_id", "batting_tier")
    design = build_design(rows, player, opponent, opp_tier, groups)
    y = rows[value].to_numpy(float)
    cv = cross_validate(design, y, match_folds(rows["match_id"], cfg["cv_folds"], config["seed"]), cfg["penalty_grid"])
    ridge = cv[cv["model"].str.startswith("ridge")]
    penalty = float(ridge.loc[ridge["held_out_mse"].idxmin(), "penalty"])
    beta = solve(design, y, penalty)
    coefficients = design.columns.assign(effect=beta)
    opp_cols = coefficients["kind"].isin(["opponent tier", "opponent group", "opponent"]).to_numpy()
    part = design.matrix[:, np.flatnonzero(opp_cols)] @ beta[opp_cols]
    return OppositionFit(role, penalty, coefficients, cv, pd.Series(part, index=rows.index))


def adjust(valued: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, OppositionFit]]:
    """Add ``v_bat_adj`` and ``v_bowl_adj``: run value with the opponent's part taken out.

    The opponent part is centred on its ball-weighted mean, so adjusted value keeps the
    overall mean of run value while competitions and opponents move relative to each other.
    The non-striker's run-out value (``v_ns``) is not adjusted: no bowler skill is involved.
    """
    balls = add_tiers(valued, config)
    scored = balls[balls["state_ok"]]
    split = config["opposition"]["split_groups_by_ipl"]
    groups = {"striker_id": home_groups(scored, "striker_id", "batting_tier", split),
              "bowler_id": home_groups(scored, "bowler_id", "bowling_tier", split)}
    fits: dict[str, OppositionFit] = {}
    for role, value, adj in (("batter", "v_bat", "v_bat_adj"), ("bowler", "v_bowl", "v_bowl_adj")):
        rows = scored[scored[value].notna()]
        fit = fit_role(rows, role, value, groups, config)
        part = fit.opponent_part - fit.opponent_part.mean()
        balls[f"opp_part_{role}"] = part.reindex(balls.index)
        balls[adj] = balls[value] - balls[f"opp_part_{role}"]
        fits[role] = fit
        logger.info("Opposition model (%s): penalty %g", role, fit.penalty)
    balls["batter_group"] = balls["striker_id"].map(groups["striker_id"])
    balls["bowler_group"] = balls["bowler_id"].map(groups["bowler_id"])
    return balls, fits
