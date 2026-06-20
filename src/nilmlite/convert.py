"""Dataset generation and ingestion into NILM-Parquet.

`make_synthetic` builds a realistic multi-appliance household trace so the repo
is runnable with zero downloads. Real converters (REDD/UK-DALE/REFIT CSV and
NILMTK-HDF5 migration) plug in here and emit the same wide tables.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .io import save_building, save_manifest
from .schema import MAINS_COL, TIME_COL, Manifest

__all__ = ["make_synthetic", "write_synthetic_dataset"]


def make_synthetic(days: int = 30, period_s: int = 6, seed: int = 0,
                   start: str = "2013-01-01", appliance_scale: float = 1.0,
                   base_w: float = 55.0, fridge_cycle_s: int = 2700,
                   name: str = "synthetic",
                   source: str = "synthetic") -> tuple[pl.DataFrame, Manifest]:
    """Generate a single-building trace: mains + fridge + microwave + dish washer.

    `seed` varies homes within a dataset (cross-building). `appliance_scale`,
    `base_w` and `fridge_cycle_s` shift the *distribution* so two datasets differ
    (cross-dataset). Returns (wide DataFrame, Manifest).
    """
    rng = np.random.default_rng(seed)
    n = int(days * 24 * 3600 // period_s)
    t_s = np.arange(n) * period_s  # seconds since start
    s = appliance_scale

    # fridge: compressor cycle, on ~20 min, with a startup surge
    phase = t_s % fridge_cycle_s
    fridge = np.where(phase < 1200, 145.0 * s, 0.0)
    fridge += np.where(phase < 30, 380.0 * s, 0.0)        # compressor inrush
    fridge += rng.normal(0, 4, n) * (fridge > 0)

    # microwave: a handful of short high-power bursts per day
    microwave = np.zeros(n)
    for _ in range(int(days * 4)):
        i = int(rng.integers(0, n))
        dur = int(rng.integers(max(1, 30 // period_s), max(2, 180 // period_s)))
        microwave[i:i + dur] = rng.uniform(1100, 1500) * s

    # dish washer: ~1 long cycle/day with a heating spike then a wash plateau
    dish = np.zeros(n)
    clen = max(1, int(5400 // period_s))                  # ~90 min
    for _ in range(int(days)):
        i = int(rng.integers(0, max(1, n - clen)))
        seg = dish[i:i + clen]
        k = seg.shape[0]
        heat = max(1, k // 8)
        seg[:heat] = 2050 * s                             # heating element
        seg[heat:] = 110 * s                              # wash/circulation
        if k > clen // 2:
            seg[k // 2: k // 2 + heat] = 1850 * s         # second heat (dry)
        dish[i:i + clen] = seg

    base = base_w + 18 * np.sin(2 * np.pi * t_s / 86400) + rng.normal(0, 6, n)
    mains = np.clip(base + fridge + microwave + dish + rng.normal(0, 9, n), 0, None)

    start_ns = np.datetime64(start).astype("datetime64[s]")
    ts = (start_ns + t_s.astype("timedelta64[s]")).astype("datetime64[us]")

    df = pl.DataFrame({
        TIME_COL: ts,
        MAINS_COL: mains.astype(np.float32),
        "fridge": fridge.astype(np.float32),
        "microwave": microwave.astype(np.float32),
        "dish_washer": dish.astype(np.float32),
    })
    man = Manifest(
        name=name,
        sample_period_s=period_s,
        appliances=["fridge", "microwave", "dish_washer"],
        buildings=[1],
        source=source,
    )
    return df, man


def write_synthetic_dataset(directory: str | Path, buildings: int = 2,
                            days: int = 20, period_s: int = 6, seed0: int = 0,
                            **variant) -> Path:
    """Write a multi-building synthetic dataset to `directory` (NILM-Parquet).

    Extra keyword args (e.g. ``appliance_scale``, ``base_w``, ``name``,
    ``source``) are forwarded to :func:`make_synthetic` to create a distinct
    *dataset variant* shared across its buildings.
    """
    directory = Path(directory)
    man = None
    for b in range(1, buildings + 1):
        df, man = make_synthetic(days=days, period_s=period_s, seed=seed0 + b,
                                 **variant)
        save_building(df, directory / f"building{b}.parquet")
    man.buildings = list(range(1, buildings + 1))
    save_manifest(directory, man)
    return directory
