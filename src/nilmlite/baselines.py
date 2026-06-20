"""Trivial, dependency-free baselines.

These are the always-available references on every leaderboard. Heavier
classic algorithms (Combinatorial Optimisation, FHMM) and deep models slot in
behind the same `fit`/`predict` interface; deep models additionally export to
ONNX for in-browser inference.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Mean", "Linear"]


class Mean:
    """Predicts each appliance's mean training power for every timestep.

    A genuinely informative floor: any model worse than Mean has learned nothing.
    """

    n_params = 1

    def __init__(self):
        self.value_: float | None = None

    def fit(self, X, y) -> "Mean":            # X unused; kept for interface parity
        self.value_ = float(np.mean(np.asarray(y, dtype=np.float64)))
        return self

    def predict(self, X) -> np.ndarray:
        if self.value_ is None:
            raise RuntimeError("call fit() first")
        return np.full(np.asarray(X).shape[0], self.value_, dtype=np.float64)


class Linear:
    """Ridge-regularised linear map from a mains window to midpoint appliance power.

    Pure NumPy (closed-form normal equations), no extra dependencies — a learned
    reference that sits between Mean and the deep models.
    """

    def __init__(self, l2: float = 1.0):
        self.l2 = l2
        self.w_: np.ndarray | None = None
        self.n_params: int | None = None

    def fit(self, X, y) -> "Linear":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])      # bias column
        d = Xb.shape[1]
        A = Xb.T @ Xb + self.l2 * np.eye(d)
        self.w_ = np.linalg.solve(A, Xb.T @ y)
        self.n_params = int(d)
        return self

    def predict(self, X) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("call fit() first")
        X = np.asarray(X, dtype=np.float64)
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])
        return np.clip(Xb @ self.w_, 0, None)
