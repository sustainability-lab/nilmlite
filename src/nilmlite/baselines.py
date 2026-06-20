"""Trivial, dependency-free baselines.

These are the always-available references on every leaderboard. Heavier
classic algorithms (Combinatorial Optimisation, FHMM) and deep models slot in
behind the same `fit`/`predict` interface; deep models additionally export to
ONNX for in-browser inference.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Mean"]


class Mean:
    """Predicts each appliance's mean training power for every timestep.

    A genuinely informative floor: any model worse than Mean has learned nothing.
    """

    def __init__(self):
        self.value_: float | None = None

    def fit(self, mains, target) -> "Mean":   # mains unused; kept for interface parity
        self.value_ = float(np.mean(np.asarray(target, dtype=np.float64)))
        return self

    def predict(self, mains) -> np.ndarray:
        if self.value_ is None:
            raise RuntimeError("call fit() first")
        return np.full(np.asarray(mains).shape[0], self.value_, dtype=np.float64)
