"""Probability-quality metrics for win-probability models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_loss(p: np.ndarray, y: np.ndarray, clip: float) -> float:
    """Mean negative log-likelihood (lower is better)."""
    q = np.clip(p, clip, 1 - clip)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of the probability (lower is better)."""
    return float(((p - y) ** 2).mean())


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int) -> pd.DataFrame:
    """Predicted v observed win rate in equal-width probability bins."""
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    frame = pd.DataFrame({"bin": which, "p": p, "y": y})
    table = frame.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return table.reset_index()


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int) -> float:
    """Average |predicted - observed| across bins, weighted by bin size (lower is better)."""
    table = reliability_table(p, y, bins)
    return float((np.abs(table["predicted"] - table["observed"]) * table["n"]).sum() / table["n"].sum())


def metric_row(p: np.ndarray, y: np.ndarray, clip: float, bins: int) -> dict[str, float]:
    """Log-loss, Brier score, calibration error and sample size for one slice."""
    return {
        "log_loss": log_loss(p, y, clip),
        "brier": brier(p, y),
        "ece": expected_calibration_error(p, y, bins),
        "n_balls": int(len(y)),
    }
