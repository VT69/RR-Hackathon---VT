"""Phase 5: one rating per eligible player from opposition-adjusted run value on pressure balls.

Everything is computed from **per-match aggregates** for each player, so that one routine
evaluates the components for any choice of matches: all matches once (the estimate),
odd/even or random halves (split-half reliability), or a fixed number of pressure balls
(volume check). Intervals and P(top 10) come from posterior simulation: each player is
drawn from their empirical-Bayes posterior.
A "multiplicity matrix" C (replicates x matches) says how many times each match counts.

Components (runs per 100 balls unless stated):
  * pressure value - weighted mean adjusted value on high-pressure balls (the headline);
  * uplift         - pressure value minus the player's own ordinary-ball value, re-weighted
                     to the bowling-phase mix of their pressure balls;
  * consistency    - share of matches with enough pressure balls in which the player beat
                     same-state expectation, minus the league base rate (percentage points).

Leverage enters only through the pressure flag and the weight on pressure balls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

PHASES: tuple[str, ...] = ("powerplay", "middle", "death")
PER_100 = 100.0


def weight_options(li: pd.Series, cap: float) -> dict[str, np.ndarray]:
    """The three candidate weights for a pressure ball (D33)."""
    return {"uniform": np.ones(len(li)), "cap": np.minimum(li.to_numpy(), cap), "sqrt": np.sqrt(li.to_numpy())}


def match_aggregates(rows: pd.DataFrame, cap: float, min_balls: int) -> pd.DataFrame:
    """One row per (player, match) with every sum the components need.

    ``rows`` is the long player-ball table (``player_value.player_balls`` on adjusted values):
    value, count (ball counted for the player), pressure flag, li, phase, match_id, match_date.
    """
    hot = rows["pressure"].to_numpy()
    out = rows[["player_id", "match_id", "match_date"]].copy()
    value, count = rows["value"].to_numpy(), rows["count"].to_numpy()
    mult = rows["weight_mult"].to_numpy() if "weight_mult" in rows else 1.0   # e.g. knockout x1.25 (Phase 6)
    for name, w in weight_options(rows["li"], cap).items():
        w = w * mult
        out[f"pn_{name}"] = np.where(hot, w * value, 0.0)
        out[f"pd_{name}"] = np.where(hot, w * count, 0.0)
        for phase in PHASES:
            out[f"hd_{phase}_{name}"] = np.where(hot & (rows["phase"] == phase).to_numpy(), w * count, 0.0)
    out["pc"] = np.where(hot, count, 0.0)
    out["on"], out["od"] = np.where(hot, 0.0, value), np.where(hot, 0.0, count)
    for phase in PHASES:
        cold = ~hot & (rows["phase"] == phase).to_numpy()
        out[f"on_{phase}"], out[f"od_{phase}"] = np.where(cold, value, 0.0), np.where(cold, count, 0.0)
    agg = out.groupby(["player_id", "match_id"], sort=False).agg({"match_date": "first",
                                                                   **{c: "sum" for c in out.columns[3:]}})
    for name in weight_options(rows["li"].iloc[:1], cap):
        agg[f"pn2_{name}"] = agg[f"pn_{name}"] ** 2
        agg[f"pnd_{name}"] = agg[f"pn_{name}"] * agg[f"pd_{name}"]
        agg[f"pd2_{name}"] = agg[f"pd_{name}"] ** 2
        qualifies = agg["pc"] >= min_balls
        agg[f"q_{name}"] = qualifies.astype(float)
        agg[f"win_{name}"] = (qualifies & (agg[f"pn_{name}"] > 0)).astype(float)
    agg["on2"], agg["ond"], agg["od2"] = agg["on"] ** 2, agg["on"] * agg["od"], agg["od"] ** 2
    agg["has_p"] = (agg["pd_uniform"] > 0).astype(float)
    agg["has_o"] = (agg["od"] > 0).astype(float)
    return agg.reset_index().sort_values(["player_id", "match_date", "match_id"])


@dataclass
class PlayerMatches:
    """Per-player match aggregate matrices, in date order, for the eligible players."""

    players: np.ndarray
    features: list[str]
    blocks: list[np.ndarray]           # each (matches_i x features)
    match_ids: list[np.ndarray]

    @classmethod
    def build(cls, agg: pd.DataFrame, players: np.ndarray) -> "PlayerMatches":
        features = [c for c in agg.columns if c not in ("player_id", "match_id", "match_date")]
        blocks, ids = [], []
        grouped = dict(tuple(agg[agg["player_id"].isin(players)].groupby("player_id", sort=False)))
        for p in players:
            part = grouped[p]
            blocks.append(part[features].to_numpy(float))
            ids.append(part["match_id"].to_numpy())
        return cls(np.asarray(players), features, blocks, ids)

    def sums(self, multiplicity: list[np.ndarray]) -> np.ndarray:
        """Summed features for every player and replicate: array (players x replicates x features)."""
        return np.stack([c @ block for c, block in zip(multiplicity, self.blocks)])

    def ones(self) -> list[np.ndarray]:
        return [np.ones((1, len(b))) for b in self.blocks]


def raw_components(s: np.ndarray, features: list[str], weight: str, min_cell: float) -> dict[str, np.ndarray]:
    """Raw components and their match-clustered standard errors from summed features.

    The standard error of a ratio of match totals (a / b) is the cluster-robust one:
    sqrt(M / (M - 1) * sum_m (a_m - r b_m)^2) / sum_m b_m, which is what a match bootstrap
    estimates; it is computed in closed form so it can be recomputed in every resample.
    """
    f = {name: s[..., i] for i, name in enumerate(features)}
    with np.errstate(divide="ignore", invalid="ignore"):
        pn, pd_ = f[f"pn_{weight}"], f[f"pd_{weight}"]
        p = pn / pd_
        m = f["has_p"]
        sse = f[f"pn2_{weight}"] - 2 * p * f[f"pnd_{weight}"] + p**2 * f[f"pd2_{weight}"]
        p_se = np.sqrt(np.clip(sse, 0, None) * m / np.maximum(m - 1, 1)) / pd_
        usual = f["on"] / f["od"]
        mo = f["has_o"]
        u_sse = f["on2"] - 2 * usual * f["ond"] + usual**2 * f["od2"]
        u_se = np.sqrt(np.clip(u_sse, 0, None) * mo / np.maximum(mo - 1, 1)) / f["od"]
        matched = np.zeros_like(p)
        for phase in PHASES:
            cell = np.where(f[f"od_{phase}"] >= min_cell, f[f"on_{phase}"] / f[f"od_{phase}"], usual)
            matched += f[f"hd_{phase}_{weight}"] / pd_ * cell
        wins, n = f[f"win_{weight}"], f[f"q_{weight}"]
    return {"pressure": p * PER_100, "pressure_se": p_se * PER_100, "usual": usual * PER_100,
            "matched_usual": matched * PER_100, "usual_se": u_se * PER_100,
            "uplift": (p - matched) * PER_100, "uplift_se": np.sqrt(p_se**2 + u_se**2) * PER_100,
            "wins": wins, "matches_counted": n, "pressure_balls": f["pc"]}


def pooled_se(se: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Replace each player's own standard error by a pooled one: a common noise level / sqrt(balls).

    The common level is the median of se^2 * n across players, so match clustering is kept,
    but a player whose own standard error is low by luck (it rests on only 20-50 matches)
    does not escape shrinkage because of it.
    """
    ok = np.isfinite(se) & (n > 0)
    k = float(np.median(se[ok] ** 2 * n[ok]))
    return np.sqrt(k / np.where(n > 0, n, np.nan))


def eb_shrink(x: np.ndarray, se: np.ndarray, prior_mean: float | None = None) -> tuple[np.ndarray, float, float]:
    """Normal-normal empirical Bayes across players: returns (shrunk, prior mean, real variance tau^2).

    ``prior_mean`` None uses the mean of the observed values (the population mean).
    tau^2 = observed variance minus the mean squared standard error, floored at zero.
    """
    ok = np.isfinite(x) & np.isfinite(se)
    mean = float(np.mean(x[ok])) if prior_mean is None else prior_mean
    tau2 = max(float(np.var(x[ok], ddof=1) - np.mean(se[ok] ** 2)), 0.0)
    factor = tau2 / (tau2 + se**2) if tau2 > 0 else np.zeros_like(se)
    return mean + (x - mean) * factor, mean, tau2


def eb_posterior_var(se: np.ndarray, tau2: float) -> np.ndarray:
    """Posterior variance of each player's true value under the normal-normal model: tau^2 s^2 / (tau^2 + s^2)."""
    return tau2 * se**2 / (tau2 + se**2) if tau2 > 0 else np.zeros_like(se)


def beta_binomial_shrink(wins: np.ndarray, n: np.ndarray, base: float) -> tuple[np.ndarray, float]:
    """Shrink match win rates towards the league base rate with a beta prior fitted by moments.

    Returns (shrunk rate minus base rate, prior strength kappa in matches). Between-player
    variance beyond binomial noise sets kappa; no excess variance means everyone gets the base rate.
    """
    ok = n > 0
    rate = np.where(ok, wins / np.where(ok, n, 1), base)
    excess = float(np.var(rate[ok], ddof=1) - np.mean(base * (1 - base) / n[ok]))
    if excess <= 0:
        return np.zeros_like(rate), float("inf")
    kappa = max(base * (1 - base) / excess - 1, 0.0)
    return (wins + kappa * base) / (n + kappa) - base, kappa


def spearman_brown(r: float) -> float:
    """Reliability of the full measure from the correlation between two halves."""
    return 2 * r / (1 + r) if r > -1 else np.nan


def half_multiplicities(pm: PlayerMatches, rng: np.random.Generator | None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Two halves of each player's matches: odd/even in date order, or a random half if ``rng`` is given."""
    first, second = [], []
    for ids in pm.match_ids:
        n = len(ids)
        mask = (np.arange(n) % 2 == 0) if rng is None else np.isin(np.arange(n), rng.permutation(n)[: n // 2])
        first.append(mask[None, :].astype(float))
        second.append((~mask)[None, :].astype(float))
    return first, second


def truncation_multiplicities(pm: PlayerMatches, balls_index: int, target: float, reps: int,
                              rng: np.random.Generator) -> list[np.ndarray]:
    """Keep whole matches in random order until each player reaches ``target`` pressure balls."""
    out = []
    for block in pm.blocks:
        counts = block[:, balls_index]
        c = np.zeros((reps, len(counts)))
        for r in range(reps):
            order = rng.permutation(len(counts))
            keep = order[: int(np.searchsorted(np.cumsum(counts[order]), target) + 1)]
            c[r, keep] = 1.0
        out.append(c)
    return out


def score(comp: dict[str, np.ndarray], base_rate: float, weights: dict[str, float]) -> dict[str, np.ndarray]:
    """Shrink every component across players (one replicate at a time) and form both ratings.

    ``comp`` arrays are (players x replicates). Standard errors are pooled (``pooled_se``).

    * composite (option A, the headline): reliability-weighted mean of z-standardised shrunk
      components; a component without real spread contributes 0.
    * hierarchical (option B): the player's own phase-matched usual level, shrunk towards the
      population mean, plus their uplift shrunk towards zero by its estimated variance.
    """
    n_players, reps = comp["pressure"].shape
    keys = ("pressure_shrunk", "uplift_shrunk", "consistency_shrunk", "composite", "usual_shrunk", "hierarchical")
    out = {k: np.full((n_players, reps), np.nan) for k in keys}
    total = sum(weights.values())
    for r in range(reps):
        n = comp["pressure_balls"][:, r]
        p, _, _ = eb_shrink(comp["pressure"][:, r], pooled_se(comp["pressure_se"][:, r], n))
        u, _, _ = eb_shrink(comp["uplift"][:, r], pooled_se(comp["uplift_se"][:, r], n), prior_mean=0.0)
        c, _ = beta_binomial_shrink(comp["wins"][:, r], comp["matches_counted"][:, r], base_rate)
        theta, _, _ = eb_shrink(comp["matched_usual"][:, r], pooled_se(comp["usual_se"][:, r], n))
        out["usual_shrunk"][:, r], out["hierarchical"][:, r] = theta, theta + u
        composite = np.zeros(n_players)
        for key, values in (("pressure", p), ("uplift", u), ("consistency", c)):
            sd = np.nanstd(values)
            if weights.get(key, 0) > 0 and sd > 0:
                composite += weights[key] * (values - np.nanmean(values)) / sd
        out["pressure_shrunk"][:, r], out["uplift_shrunk"][:, r], out["consistency_shrunk"][:, r] = p, u, c
        out["composite"][:, r] = composite / total if total > 0 else composite
    return out


def ranks(values: np.ndarray) -> np.ndarray:
    """Rank 1 = highest, per replicate (column)."""
    return (-values).argsort(axis=0).argsort(axis=0) + 1


def reliability_table(pm: PlayerMatches, weight: str, min_cell: float, base_rate: float, reps: int,
                      rng: np.random.Generator) -> pd.DataFrame:
    """Split-half reliability of each raw component: odd/even, and over random halves.

    ``r_*`` columns are the raw correlation between the two halves; the others are
    Spearman-Brown corrected, 2r / (1 + r): the reliability of the full-sample measure.
    """
    def one_split(halves: tuple[list[np.ndarray], list[np.ndarray]]) -> dict[str, float]:
        a = raw_components(pm.sums(halves[0])[:, 0, :], pm.features, weight, min_cell)
        b = raw_components(pm.sums(halves[1])[:, 0, :], pm.features, weight, min_cell)
        out = {}
        for key in ("pressure", "uplift", "usual"):
            ok = np.isfinite(a[key]) & np.isfinite(b[key])
            out[key] = float(np.corrcoef(a[key][ok], b[key][ok])[0, 1])
        ra = np.where(a["matches_counted"] > 0, a["wins"] / np.maximum(a["matches_counted"], 1), np.nan) - base_rate
        rb = np.where(b["matches_counted"] > 0, b["wins"] / np.maximum(b["matches_counted"], 1), np.nan) - base_rate
        ok = np.isfinite(ra) & np.isfinite(rb)
        out["consistency"] = float(np.corrcoef(ra[ok], rb[ok])[0, 1])
        return out

    odd_even = one_split(half_multiplicities(pm, None))
    random = pd.DataFrame([one_split(half_multiplicities(pm, rng)) for _ in range(reps)])
    corrected = random.apply(lambda col: col.map(spearman_brown))
    rows = []
    for key in ("pressure", "uplift", "consistency", "usual"):
        rows.append({"component": key, "r_odd_even": odd_even[key], "r_random_mean": random[key].mean(),
                     "odd_even": spearman_brown(odd_even[key]), "random_mean": corrected[key].mean(),
                     "random_p05": corrected[key].quantile(0.05), "random_p95": corrected[key].quantile(0.95),
                     "random_sd": corrected[key].std()})
    return pd.DataFrame(rows).assign(weighting=weight)


def choose_weighting(tables: dict[str, pd.DataFrame], order: list[str]) -> tuple[str, pd.DataFrame]:
    """Pick the weighting with the most reproducible pressure value; the simplest within one SD of the best.

    ``tables`` maps weighting -> reliability tables stacked over roles. ``order`` lists
    weightings from simplest to most complex.
    """
    rows = []
    for name in order:
        t = tables[name]
        p = t[t["component"] == "pressure"]
        rows.append({"weighting": name, "pressure_reliability_mean": p["random_mean"].mean(),
                     "pressure_reliability_sd": p["random_sd"].mean(), "odd_even": p["odd_even"].mean()})
    summary = pd.DataFrame(rows)
    best = summary.loc[summary["pressure_reliability_mean"].idxmax()]
    tied = summary[summary["pressure_reliability_mean"] >= best["pressure_reliability_mean"] - best["pressure_reliability_sd"]]
    chosen = next(name for name in order if name in set(tied["weighting"]))
    return chosen, summary.assign(chosen=summary["weighting"] == chosen)


def base_rate(agg: pd.DataFrame, weight: str) -> float:
    """League base rate: share of all qualifying player-matches (every player) that beat expectation."""
    return float(agg[f"win_{weight}"].sum() / agg[f"q_{weight}"].sum())


def rate_role(pm: PlayerMatches, weight: str, base: float, reliability: pd.DataFrame, config: dict[str, Any],
              rng: np.random.Generator) -> dict[str, Any]:
    """Point estimates, posterior intervals and top-n probabilities for both ratings, and the volume checks.

    Posterior simulation: each player's true pressure value, usual level and uplift are drawn
    from their normal empirical-Bayes posteriors (mean = shrunk value, variance =
    tau^2 s^2 / (tau^2 + s^2)) and consistency from its beta posterior; every draw is turned
    into both ratings and ranked. Hyperparameters (tau^2, kappa) are held at their estimates.
    """
    cfg, min_cell = config["rating"], config["value"]["baseline_min_cell_balls"]
    rel = reliability.set_index("component")["random_mean"]
    weights = {k: max(float(rel[k]), 0.0) for k in ("pressure", "uplift", "consistency")}
    raw = raw_components(pm.sums(pm.ones()), pm.features, weight, min_cell)
    raw = {k: v[:, 0] for k, v in raw.items()}
    point = {k: v[:, 0] for k, v in score({k: v[:, None] for k, v in raw.items()}, base, weights).items()}
    n, draws, (lo, hi), top_n = raw["pressure_balls"], cfg["posterior_draws"], cfg["interval"], cfg["top_n"]

    def normal_draws(x: np.ndarray, se: np.ndarray, prior: float | None) -> np.ndarray:
        shrunk, _, tau2 = eb_shrink(x, se, prior_mean=prior)
        return shrunk[:, None] + np.sqrt(eb_posterior_var(se, tau2))[:, None] * rng.standard_normal((len(x), draws))

    p_draw = normal_draws(raw["pressure"], pooled_se(raw["pressure_se"], n), None)
    u_draw = normal_draws(raw["uplift"], pooled_se(raw["uplift_se"], n), 0.0)
    t_draw = normal_draws(raw["matched_usual"], pooled_se(raw["usual_se"], n), None)
    _, kappa = beta_binomial_shrink(raw["wins"], raw["matches_counted"], base)
    if np.isfinite(kappa):
        a = raw["wins"] + kappa * base
        b = raw["matches_counted"] - raw["wins"] + kappa * (1 - base)
        c_draw = rng.beta(a[:, None].repeat(draws, 1), b[:, None].repeat(draws, 1)) - base
    else:
        c_draw = np.zeros((len(n), draws))
    total = sum(weights.values())
    composite_draw = np.zeros_like(p_draw)
    for key, d in (("pressure", p_draw), ("uplift", u_draw), ("consistency", c_draw)):
        sd = float(np.nanstd(point[f"{key}_shrunk"]))
        if weights[key] > 0 and sd > 0:
            composite_draw += weights[key] * (d - np.nanmean(point[f"{key}_shrunk"])) / sd
    composite_draw /= total
    hier_draw = t_draw + u_draw
    a_rank, b_rank = ranks(composite_draw), ranks(hier_draw)

    trunc_raw = raw_components(
        pm.sums(truncation_multiplicities(pm, pm.features.index("pc"), config["eligibility"]["min_pressure_balls"],
                                          cfg["truncation_reps"], rng)), pm.features, weight, min_cell)
    trunc = score(trunc_raw, base, weights)

    table = pd.DataFrame({
        "player_id": pm.players,
        "pressure_balls": n,
        "pressure_value_raw": raw["pressure"],
        "pressure_value_se_own": raw["pressure_se"],
        "pressure_value_se": pooled_se(raw["pressure_se"], n),
        "pressure_value_shrunk": point["pressure_shrunk"],
        "pressure_value_p05": np.quantile(p_draw, lo, axis=1),
        "pressure_value_p95": np.quantile(p_draw, hi, axis=1),
        "usual_value_raw": raw["usual"],
        "matched_usual_raw": raw["matched_usual"],
        "matched_usual_shrunk": point["usual_shrunk"],
        "uplift_raw": raw["uplift"],
        "uplift_shrunk": point["uplift_shrunk"],
        "consistency_matches": raw["matches_counted"],
        "consistency_raw": np.where(raw["matches_counted"] > 0, raw["wins"] / np.maximum(raw["matches_counted"], 1), np.nan) - base,
        "consistency_shrunk": point["consistency_shrunk"],
        "composite": point["composite"],
        "composite_p05": np.quantile(composite_draw, lo, axis=1),
        "composite_p95": np.quantile(composite_draw, hi, axis=1),
        "rank_p05": np.quantile(a_rank, lo, axis=1),
        "rank_p95": np.quantile(a_rank, hi, axis=1),
        "p_top_n": (a_rank <= top_n).mean(axis=1),
        "p_top_n_equal_balls": (ranks(trunc["composite"]) <= top_n).mean(axis=1),
        "hierarchical": point["hierarchical"],
        "hierarchical_p05": np.quantile(hier_draw, lo, axis=1),
        "hierarchical_p95": np.quantile(hier_draw, hi, axis=1),
        "p_top_n_hierarchical": (b_rank <= top_n).mean(axis=1),
    })
    table["rank"] = table["composite"].rank(ascending=False, method="first").astype(int)
    table["rank_hierarchical"] = table["hierarchical"].rank(ascending=False, method="first").astype(int)

    sds = {k: float(np.nanstd(point[f"{k}_shrunk"])) for k in ("pressure", "uplift", "consistency")}
    effective = pd.DataFrame([{
        "component": k, "reliability_weight": weights[k], "share_of_weight": weights[k] / total if total else np.nan,
        "sd_of_shrunk_component": sds[k],
        "effective_coefficient": weights[k] / total / sds[k] if total and sds[k] > 0 else 0.0,
        "unit": "per unit share (x0.01 per percentage point)" if k == "consistency" else "runs per 100 balls",
    } for k in weights])
    volume = pd.DataFrame([{
        "spearman_rating_v_pressure_balls": table[["composite", "pressure_balls"]].corr(method="spearman").iloc[0, 1],
        "spearman_raw_pressure_v_pressure_balls": table[["pressure_value_raw", "pressure_balls"]].corr(method="spearman").iloc[0, 1],
        "spearman_shrunk_pressure_v_pressure_balls": table[["pressure_value_shrunk", "pressure_balls"]].corr(method="spearman").iloc[0, 1],
        "corr_pressure_v_consistency_shrunk": table[["pressure_value_shrunk", "consistency_shrunk"]].corr().iloc[0, 1],
        "corr_pressure_v_consistency_raw": table[["pressure_value_raw", "consistency_raw"]].corr().iloc[0, 1],
        "corr_pressure_v_uplift_shrunk": table[["pressure_value_shrunk", "uplift_shrunk"]].corr().iloc[0, 1],
    }])
    return {"table": table.sort_values("rank"), "effective": effective, "volume": volume, "weights": weights,
            "base_rate": base}


def point_scores(rows: pd.DataFrame, config: dict[str, Any], weight: str, weights: dict[str, float],
                 min_balls: int, min_matches: int) -> pd.DataFrame:
    """Point estimates of both ratings for one variant of the data (Phase 6 sensitivity).

    Eligibility is recomputed from ``rows`` with the given thresholds; the component weights
    are held at the headline values so that only the data or definition changes.
    """
    cfg = config["rating"]
    agg = match_aggregates(rows, cfg["cap"], cfg["consistency_min_pressure_balls"])
    per_player = agg.groupby("player_id").agg(pc=("pc", "sum"), pm=("pc", lambda s: int((s > 0).sum())))
    eligible = per_player[(per_player["pc"] >= min_balls) & (per_player["pm"] >= min_matches)].index.to_numpy()
    if len(eligible) < cfg["top_n"]:
        return pd.DataFrame(columns=["player_id", "rank", "rank_hierarchical"])
    pm = PlayerMatches.build(agg, eligible)
    raw = raw_components(pm.sums(pm.ones()), pm.features, weight, config["value"]["baseline_min_cell_balls"])
    point = score(raw, base_rate(agg, weight), weights)
    out = pd.DataFrame({"player_id": pm.players, "composite": point["composite"][:, 0], "hierarchical": point["hierarchical"][:, 0]})
    out["rank"] = out["composite"].rank(ascending=False, method="first").astype(int)
    out["rank_hierarchical"] = out["hierarchical"].rank(ascending=False, method="first").astype(int)
    return out
