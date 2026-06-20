"""Head-to-head: nilmlite (Polars/Parquet) vs the pandas/PyTables (NILMTK-style)
data layer, on the identical end-to-end pipeline:

    read all meters  ->  align to a wide table  ->  resample  ->  window mains  ->  metrics

Run `python bench/gen_data.py` first.

This measures the *data layer* a benchmark harness actually spends wall-clock in;
it is not a comparison of NILMTK's modelling code.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nilmlite import metrics, resample, windows  # noqa: E402
from nilmlite.schema import MAINS_COL, TIME_COL  # noqa: E402

OUT = Path("data/bench")
RESAMPLE_S = 60
WIDTH = 99
REPS = 3
METERS = ["mains", "fridge", "microwave", "dish_washer"]


def _t(fn):
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def run_polars():
    stages = {}
    df, stages["read+align"] = _t(lambda: pl.read_parquet(OUT / "building1.parquet"))
    rs, stages["resample"] = _t(lambda: resample.resample(df, RESAMPLE_S))
    main = rs[MAINS_COL].to_numpy()
    fridge = rs["fridge"].to_numpy()
    _, stages["window"] = _t(lambda: windows.seq2point_xy(main, fridge, WIDTH))
    pred = fridge * 0.9 + 5.0
    _, stages["metrics"] = _t(lambda: metrics.report(fridge, pred))
    return stages


def _read_align_pandas(path: Path, fmt_keys=True):
    with pd.HDFStore(path, mode="r") as store:
        frames = []
        for i, col in enumerate(METERS, start=1):
            s = store.get(f"/building1/elec/meter{i}")
            s.columns = [col]
            frames.append(s)
        return pd.concat(frames, axis=1)


def run_pandas(path: Path):
    stages = {}
    df, stages["read+align"] = _t(lambda: _read_align_pandas(path))
    rs, stages["resample"] = _t(lambda: df.resample(f"{RESAMPLE_S}s").mean())
    main = rs[MAINS_COL].to_numpy()
    fridge = rs["fridge"].to_numpy()
    _, stages["window"] = _t(lambda: windows.seq2point_xy(main, fridge, WIDTH))
    pred = fridge * 0.9 + 5.0
    _, stages["metrics"] = _t(lambda: metrics.report(fridge, pred))
    return stages


def bench(label, fn):
    runs = [fn() for _ in range(REPS)]
    agg = {k: median(r[k] for r in runs) for k in runs[0]}
    agg["total"] = sum(agg.values())
    return label, agg


def main():
    if not (OUT / "building1.parquet").exists():
        sys.exit("run `python bench/gen_data.py` first")

    rows = [
        bench("nilmlite (Polars/Parquet)", run_polars),
        bench("pandas/HDF5 table (NILMTK)", lambda: run_pandas(OUT / "redd_like.h5")),
        bench("pandas/HDF5 fixed", lambda: run_pandas(OUT / "redd_like_fixed.h5")),
    ]

    cols = ["read+align", "resample", "window", "metrics", "total"]
    head = f"{'backend':<28}" + "".join(f"{c:>13}" for c in cols)
    line = "-" * len(head)
    print("\n" + head)
    print(line)
    for label, agg in rows:
        print(f"{label:<28}" + "".join(f"{agg[c]*1000:>10.1f}ms" for c in cols))
    print(line)

    base = dict(rows[1][1])  # NILMTK table as reference
    sp = base["total"] / rows[0][1]["total"]
    print(f"\nnilmlite total speedup vs NILMTK-style HDF5(table): {sp:.1f}x")

    def mb(p):
        return p.stat().st_size / 1e6 if p.exists() else float("nan")

    print("\non-disk size:")
    print(f"  parquet (zstd)        {mb(OUT/'building1.parquet'):8.1f} MB")
    print(f"  hdf5 table  (NILMTK)  {mb(OUT/'redd_like.h5'):8.1f} MB")
    print(f"  hdf5 fixed            {mb(OUT/'redd_like_fixed.h5'):8.1f} MB")

    _write_results_md(rows, cols, sp)


def _write_results_md(rows, cols, sp):
    def mb(p):
        return p.stat().st_size / 1e6 if p.exists() else float("nan")

    lines = ["# Benchmark: data layer\n",
             "Identical pipeline (read → align → resample → window → metrics) over the same",
             "synthetic trace. See `bench/bench_load.py`.\n",
             "| backend | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for label, agg in rows:
        lines.append("| " + label + " | " +
                     " | ".join(f"{agg[c]*1000:.1f} ms" for c in cols) + " |")
    lines += [
        f"\n**nilmlite is {sp:.1f}× faster end-to-end** than the NILMTK-style "
        "pandas/PyTables(table) layer.\n",
        "| format | on-disk |",
        "|---|---|",
        f"| Parquet (zstd) | {mb(OUT/'building1.parquet'):.1f} MB |",
        f"| HDF5 table (NILMTK-style) | {mb(OUT/'redd_like.h5'):.1f} MB |",
        f"| HDF5 fixed | {mb(OUT/'redd_like_fixed.h5'):.1f} MB |",
    ]
    Path("bench/results.md").write_text("\n".join(lines) + "\n")
    print("\nwrote bench/results.md")


if __name__ == "__main__":
    main()
