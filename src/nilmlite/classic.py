"""Classic NILM baselines that NILMTK users expect.

Unlike the per-appliance seq2point models, these are *joint* disaggregators: from
the aggregate they recover all appliances at once. Pure NumPy, no extra deps.

  CombinatorialOptimization — Hart (1985): learn each appliance's power states,
  then pick the on/off combination that best explains each aggregate sample.
"""
from __future__ import annotations

import itertools

import numpy as np

from .metrics import report

__all__ = ["CombinatorialOptimization", "evaluate_joint"]


class CombinatorialOptimization:
    """Classic CO. `fit` takes {appliance: training power}; `disaggregate` takes
    the aggregate and returns {appliance: predicted power}."""

    def __init__(self, n_states: int = 2):
        self.n_states = n_states
        self.appliances: list[str] = []
        self.states: dict[str, np.ndarray] = {}
        self._combos = None
        self._sums = None

    @staticmethod
    def _learn_states(y, k):
        y = np.asarray(y, dtype=np.float64)
        on = y[y > 15]
        if k <= 1 or on.size == 0:
            return np.array([0.0, float(on.mean()) if on.size else 0.0])
        qs = np.quantile(on, np.linspace(0, 1, k))      # k representative on-levels
        return np.unique(np.concatenate([[0.0], qs]))

    def fit(self, train: dict[str, np.ndarray]) -> "CombinatorialOptimization":
        self.appliances = list(train)
        self.states = {a: self._learn_states(train[a], self.n_states) for a in self.appliances}
        self._combos = np.array(list(itertools.product(*[self.states[a] for a in self.appliances])))
        self._sums = self._combos.sum(axis=1)
        return self

    def disaggregate(self, mains) -> dict[str, np.ndarray]:
        if self._combos is None:
            raise RuntimeError("call fit() first")
        mains = np.asarray(mains, dtype=np.float64)
        # nearest state-combination per sample (chunked to bound memory)
        out = np.empty((mains.shape[0], len(self.appliances)))
        for i in range(0, mains.shape[0], 20000):
            chunk = mains[i:i + 20000]
            idx = np.abs(chunk[:, None] - self._sums[None, :]).argmin(axis=1)
            out[i:i + 20000] = self._combos[idx]
        return {a: out[:, j] for j, a in enumerate(self.appliances)}

    @property
    def n_params(self):
        return int(sum(len(s) for s in self.states.values()))


def evaluate_joint(model, mains, truth: dict[str, np.ndarray], threshold: float = 15.0):
    """Fit-free scoring: run `model.disaggregate` and report metrics per appliance."""
    pred = model.disaggregate(mains)
    return {a: report(truth[a], pred[a], threshold) for a in truth if a in pred}
