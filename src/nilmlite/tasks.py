"""Benchmark task definitions.

A *task* is a sealed train/test specification over NILM-Parquet buildings. The
three canonical generalization regimes (the point of the NILMBench paper) are:

  * same-building   : train and test on the same home, different periods
  * cross-building  : train on home(s) A, test on unseen home B (same dataset)
  * cross-dataset   : train on dataset X, test on dataset Y (the hardest)

A model is anything with ``fit(X, y)`` / ``predict(X) -> (n,)`` and an optional
``n_params`` attribute, where X is a (n, window) array of mains windows and y is
the midpoint appliance power. The same interface covers Mean, classic
algorithms, and deep models (whose weights also export to ONNX for the browser).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Split:
    """One building of one dataset (a NILM-Parquet directory + building id)."""
    dataset: str
    building: int


@dataclass
class Task:
    name: str
    appliances: list[str]
    train: list[Split]
    test: list[Split]
    kind: str = "cross_building"          # same_building | cross_building | cross_dataset
    sample_period_s: int = 60
    window: int = 99
    on_threshold: float = 15.0
    note: str = field(default="")

    def __post_init__(self):
        if self.window % 2 == 0:
            raise ValueError("window must be odd (seq2point predicts the midpoint)")
