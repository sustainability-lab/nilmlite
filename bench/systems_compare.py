"""Same-machine, same-data systems comparison: NILMTK's own API vs nilmlite.

Run inside the nilmtk container (has nilmtk) after installing nilmlite, with
REDD HDF5 at /in/redd.h5 and the migrated Parquet at /pq. Identical pipeline:
load mains + 3 appliances @ 6 s -> align -> seq2point window -> metric.
"""
import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

H5, PARQ, B, PERIOD, W, REPS = "/in/redd.h5", "/pq", 1, 6, 99, 3
APPS = ["fridge", "microwave", "dish washer"]


def best(fn):
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)


def nilmtk_run():
    import pandas as pd
    from nilmtk import DataSet
    ds = DataSet(H5); elec = ds.buildings[B].elec
    cols = {"mains": elec.mains().power_series_all_data(sample_period=PERIOD)}
    for a in APPS:
        try:
            cols[a] = elec[a].power_series_all_data(sample_period=PERIOD)
        except Exception:
            pass
    df = pd.concat(cols, axis=1).dropna(how="all").fillna(0.0)
    m, f = df["mains"].to_numpy(), df["fridge"].to_numpy()
    X = sliding_window_view(m, W); y = f[W // 2: W // 2 + X.shape[0]]
    _ = float(np.mean(np.abs(y - y)))
    ds.store.close()


def nilmlite_run():
    import nilmlite as nl
    df = nl.resample_to(nl.load_building(f"{PARQ}/building{B}.parquet"), PERIOD)
    m, f = df["mains"].to_numpy(), df["fridge"].to_numpy()
    X, y = nl.seq2point_xy(m, f, W); _ = nl.metrics.mae(y, y)


if __name__ == "__main__":
    import os
    a = best(nilmtk_run)
    b = best(nilmlite_run)
    print(f"nilmtk   (HDF5 + pandas)      : {a:7.3f} s")
    print(f"nilmlite (Parquet + Polars)   : {b:7.3f} s   -> {a/b:.1f}x faster")
    h5 = os.path.getsize(H5) / 1e6
    pq = sum(os.path.getsize(f"{PARQ}/{x}") for x in os.listdir(PARQ) if x.endswith(".parquet")) / 1e6
    print(f"storage  REDD: HDF5 {h5:.0f} MB  vs  Parquet {pq:.0f} MB  -> {h5/pq:.1f}x smaller")
