"""Migrate a NILMTK HDF5 dataset to NILM-Parquet — run ONCE in your old nilmtk env.

    python tools/nilmtk_to_parquet.py redd.h5 data/redd --period 6 \
        --appliances fridge,microwave,"dish washer"

This is deliberately NOT part of the `nilmlite` package: it depends on the heavy
legacy stack (nilmtk + pandas + PyTables). You run it once to escape that stack;
afterwards `nilmlite` reads the Parquet with no HDF5/conda/Docker ever again.

Works for any NILMTK dataset: REDD, UK-DALE, REFIT, iAWE, ECO, ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from nilmtk import DataSet

# nilmtk appliance label -> clean column name
DEFAULT_APPLIANCES = ["fridge", "microwave", "dish washer"]
COLNAME = {"dish washer": "dish_washer", "washer dryer": "washer_dryer"}


def _series(meter, period):
    s = meter.power_series_all_data(sample_period=period)
    return s[~s.index.duplicated(keep="first")].sort_index()


def convert(h5: str, out: str, period: int, appliances: list[str], name: str):
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = DataSet(h5)
    written, cols_seen = [], set()

    for b in ds.buildings:
        elec = ds.buildings[b].elec
        try:
            cols = {"mains": _series(elec.mains(), period)}
        except Exception as e:
            print(f"  building{b}: no mains ({e}); skipping")
            continue
        for app in appliances:
            try:
                cols[COLNAME.get(app, app)] = _series(elec[app], period)
            except Exception:
                pass  # appliance absent in this home
        df = pd.concat(cols, axis=1).dropna(how="all").fillna(0.0)
        df.index.name = "timestamp"
        df = df.reset_index()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
        for c in df.columns:
            if c != "timestamp":
                df[c] = df[c].astype(np.float32)
        path = out_dir / f"building{b}.parquet"
        df.to_parquet(path, compression="zstd", index=False)
        rows, mb = len(df), path.stat().st_size / 1e6
        appcols = [c for c in df.columns if c not in ("timestamp", "mains")]
        cols_seen.update(appcols)
        written.append(b)
        print(f"  building{b}: {rows:,} rows, {mb:.1f} MB, appliances={appcols}")

    manifest = {
        "name": name, "sample_period_s": period,
        "appliances": sorted(cols_seen), "buildings": written,
        "units": "W", "timezone": "UTC", "source": Path(h5).name,
        "version": "nilm-parquet/1",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_dir}/manifest.json  buildings={written} appliances={sorted(cols_seen)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("h5")
    p.add_argument("out")
    p.add_argument("--period", type=int, default=6, help="resample period (s)")
    p.add_argument("--appliances", default=",".join(DEFAULT_APPLIANCES))
    p.add_argument("--name", default=None)
    a = p.parse_args()
    apps = [s.strip() for s in a.appliances.split(",") if s.strip()]
    convert(a.h5, a.out, a.period, apps, a.name or Path(a.h5).stem)


if __name__ == "__main__":
    main()
