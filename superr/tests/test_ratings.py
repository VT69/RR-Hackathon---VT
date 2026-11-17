"""Phase 5 rating arithmetic: shrinkage, beta-binomial consistency, split-half reliability, weighting choice."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.rating.ratings import beta_binomial_shrink, choose_weighting, eb_shrink, ranks, spearman_brown


def test_eb_shrink_pulls_noisy_players_harder_towards_the_mean() -> None:
    x = np.array([10.0, 10.0, -10.0, -10.0, 0.0])
    se = np.array([1.0, 5.0, 1.0, 5.0, 1.0])
    shrunk, mean, tau2 = eb_shrink(x, se)
    assert np.isclose(mean, 0.0) and tau2 > 0
    assert abs(shrunk[0]) > abs(shrunk[1])         # same observed value, more noise -> pulled further in
    assert abs(shrunk[0]) < 10.0


def test_eb_shrink_with_no_real_spread_returns_the_mean_for_everyone() -> None:
    x = np.array([1.0, -1.0, 0.5, -0.5])
    shrunk, mean, tau2 = eb_shrink(x, np.full(4, 10.0))
    assert tau2 == 0 and np.allclose(shrunk, mean)


def test_beta_binomial_centres_on_the_base_rate_and_shrinks_small_samples() -> None:
    wins = np.array([18.0, 2.0, 9.0, 1.0, 10.0, 3.0])
    n = np.array([20.0, 20.0, 20.0, 2.0, 20.0, 20.0])
    rel, kappa = beta_binomial_shrink(wins, n, base=0.5)
    assert 0 < kappa < np.inf
    assert rel[0] > 0 > rel[1]                      # a good and a poor record keep their sign
    assert abs(rel[3]) < abs(rel[0])               # 1 of 2 matches stays close to the base rate


def test_spearman_brown_doubles_a_half_test_correctly() -> None:
    assert np.isclose(spearman_brown(0.5), 2 / 3)
    assert np.isclose(spearman_brown(0.0), 0.0)


def test_choose_weighting_prefers_the_simplest_option_when_reliabilities_tie() -> None:
    def table(mean: float, sd: float) -> pd.DataFrame:
        return pd.DataFrame([{"component": "pressure", "random_mean": mean, "random_sd": sd, "odd_even": mean}])

    tied, _ = choose_weighting({"uniform": table(0.40, 0.05), "cap": table(0.43, 0.05), "sqrt": table(0.42, 0.05)},
                               ["uniform", "cap", "sqrt"])
    clear, _ = choose_weighting({"uniform": table(0.30, 0.02), "cap": table(0.45, 0.02), "sqrt": table(0.44, 0.02)},
                                ["uniform", "cap", "sqrt"])
    assert tied == "uniform" and clear == "cap"


def test_ranks_are_one_for_the_highest_value_in_each_replicate() -> None:
    values = np.array([[3.0, 1.0], [5.0, 2.0], [4.0, 9.0]])
    assert ranks(values).tolist() == [[3, 3], [1, 2], [2, 1]]


def test_tie_break_reorders_only_indistinguishable_neighbours_by_option_b() -> None:
    from src.robustness.sensitivity import apply_tie_break

    rated = pd.DataFrame({
        "rank": [1, 2, 3, 4], "name": ["A", "B", "C", "D"], "composite": [4.0, 3.0, 2.9, 1.0],
        "composite_p05": [3.5, 2.0, 2.0, 0.5], "composite_p95": [4.5, 4.0, 4.0, 1.5],
        "p_top_n": [0.95, 0.60, 0.55, 0.10], "hierarchical": [1.0, 0.0, 5.0, 9.0],
        "player_id": [1, 2, 3, 4], "rank_hierarchical": [3, 4, 2, 1],
    })
    final, checks = apply_tie_break(rated, zone=4, max_p_gap=0.15)
    order = final.sort_values("final_rank")["name"].tolist()
    assert checks["tied"].tolist() == [False, True, False]
    assert order == ["A", "C", "B", "D"]          # only B and C are tied; C wins on option B, D stays below both


def test_stability_guard_keeps_fragile_players_out_of_the_top_n() -> None:
    from src.robustness.sensitivity import apply_tie_break

    rated = pd.DataFrame({
        "player_id": [1, 2, 3, 4], "rank": [1, 2, 3, 4], "name": ["A", "B", "C", "D"],
        "composite_p05": [0.0] * 4, "composite_p95": [5.0] * 4, "p_top_n": [0.50, 0.45, 0.40, 0.35],
        "hierarchical": [1.0, 9.0, 2.0, 3.0], "rank_hierarchical": [4, 1, 3, 2],
    })
    classes = pd.Series({1: "stable", 2: "fragile", 3: "moderately stable", 4: "stable"})
    final, _ = apply_tie_break(rated, zone=4, max_p_gap=0.15, classes=classes, top_n=2)
    order = final.sort_values("final_rank")["name"].tolist()
    assert "B" not in order[:2]                    # fragile B never reaches the top 2, despite the best B
    assert order.index("B") > order.index("A")     # and is never promoted above a non-fragile player ranked higher in A
