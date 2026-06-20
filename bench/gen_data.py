"""Materialise the same synthetic trace into both storage formats:

  * NILM-Parquet (nilmlite)                          -> data/bench/building1.parquet
  * pandas + PyTables HDF5, table format (NILMTK-like) -> data/bench/redd_like.h5
  * pandas + PyTables HDF5, fixed format               -> data/bench/redd_like_fixed.h5

NILMTK stores one DataFrame per meter under a key hierarchy and uses the
queryable *table* format; we reproduce that shape so the benchmark compares the
real data layers, not a strawman.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nilmlite.convert import make_synthetic  # noqa: E402
from nilmlite.io import save_building, save_manifest  # noqa: E402
from nilmlite.schema import MAINS_COL, TIME_COL  # noqa: E402

DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
PERIOD = int(sys.argv[2]) if len(sys.argv) > 2 else 6
OUT = Path("data/bench")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, man = make_synthetic(days=DAYS, period_s=PERIOD, seed=0)
    print(f"generated {len(df):,} rows  ({DAYS} days @ {PERIOD}s)")

    # --- nilmlite: one wide parquet ---
    save_manifest(OUT, man)
    pq = OUT / "building1.parquet"
    save_building(df, pq)

    # --- pandas / PyTables: per-meter DataFrames, NILMTK-style key hierarchy ---
    pdf = df.to_pandas().set_index(TIME_COL)
    meters = [MAINS_COL, *man.appliances]

    h5_table = OUT / "redd_like.h5"
    with pd.HDFStore(h5_table, mode="w", complevel=0) as store:
        for i, col in enumerate(meters, start=1):
            store.put(f"/building1/elec/meter{i}", pdf[[col]], format="table")

    h5_fixed = OUT / "redd_like_fixed.h5"
    with pd.HDFStore(h5_fixed, mode="w", complevel=0) as store:
        for i, col in enumerate(meters, start=1):
            store.put(f"/building1/elec/meter{i}", pdf[[col]], format="fixed")

    def mb(p: Path) -> float:
        return p.stat().st_size / 1e6

    print(f"  parquet (zstd)        {mb(pq):8.1f} MB  {pq}")
    print(f"  hdf5 table  (NILMTK)  {mb(h5_table):8.1f} MB  {h5_table}")
    print(f"  hdf5 fixed            {mb(h5_fixed):8.1f} MB  {h5_fixed}")


if __name__ == "__main__":
    main()
