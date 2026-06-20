"""Run models against a Task and produce a leaderboard-ready result dict."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from .io import load_building
from .resample import resample
from .metrics import report
from .schema import MAINS_COL
from .tasks import Split, Task
from .windows import seq2point_xy

__all__ = ["evaluate", "build_xy"]


def _xy_for_split(task: Task, split: Split, appliance: str):
    path = Path(split.dataset) / f"building{split.building}.parquet"
    df = load_building(path)
    rs = resample(df, task.sample_period_s)
    mains = rs[MAINS_COL].to_numpy().astype(np.float64)
    target = rs[appliance].to_numpy().astype(np.float64)
    return seq2point_xy(mains, target, task.window)


def build_xy(task: Task, splits: list[Split], appliance: str):
    """Concatenate seq2point windows/targets across splits for one appliance."""
    parts = [_xy_for_split(task, s, appliance) for s in splits]
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    return X, y


def evaluate(model_factories: dict[str, Callable[[], object]], task: Task) -> dict:
    """Train + score each model on `task`, averaged over the task's appliances.

    `model_factories` maps a model name to a zero-arg callable returning a fresh
    model (re-fit per appliance). Returns a dict ready for `leaderboard`.
    """
    acc = {name: {"mae": [], "f1": [], "infer_ms": 0.0, "params": None}
           for name in model_factories}

    for app in task.appliances:
        Xtr, ytr = build_xy(task, task.train, app)
        Xte, yte = build_xy(task, task.test, app)
        for name, factory in model_factories.items():
            model = factory()
            model.fit(Xtr, ytr)
            t0 = perf_counter()
            pred = np.asarray(model.predict(Xte)).ravel()
            acc[name]["infer_ms"] += (perf_counter() - t0) * 1000
            r = report(yte, pred, task.on_threshold)
            acc[name]["mae"].append(r["mae"])
            acc[name]["f1"].append(r["f1"])
            acc[name]["params"] = getattr(model, "n_params", None)

    models = {}
    for name, a in acc.items():
        models[name] = {
            "mae": float(np.mean(a["mae"])),
            "f1": float(np.nanmean(a["f1"])),
            "params": a["params"],
            "infer_ms": round(a["infer_ms"], 1),
        }
    return {
        "task": task.name,
        "kind": task.kind,
        "appliances": task.appliances,
        "note": task.note,
        "models": models,
    }
