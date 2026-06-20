"""Patch docs/demo_data.json with richer dataset-exploration payloads — longer
multi-day slices + per-dataset stats — WITHOUT retraining (reuses models/matrix/zoo).

  python examples/build_viz.py
"""
import json
from pathlib import Path

import numpy as np

import nilmlite as nl

DOCS = Path("docs")
P, L = 60, 3000               # 60 s period, ~2-day slice (zoom/pan to explore)

VIZ = [
    ("REDD · US",    "data/redd",   3, ["fridge", "microwave", "dish_washer"]),
    ("UK-DALE · UK", "data/ukdale", 2, ["fridge", "microwave", "dish_washer", "washing_machine", "kettle"]),
    ("iAWE · India", "data/iawe",   1, ["fridge", "air_conditioner", "washing_machine"]),
]


def active_slice(a, L):
    bs, best = 0, -1
    for s in range(0, max(1, len(a) - L), 200):
        sc = int((a[s:s + L] > 15).sum())
        if sc > best:
            best, bs = sc, s
    return bs, min(L, len(a) - bs)


data = json.loads((DOCS / "demo_data.json").read_text())
datasets = {}
for name, path, b, viz in VIZ:
    bp = Path(path) / f"building{b}.parquet"
    if not bp.exists():
        continue
    full = nl.load_building(bp)
    span_days = (full["timestamp"].max() - full["timestamp"].min()).total_seconds() / 86400
    nb = len(list(Path(path).glob("building*.parquet")))
    df = nl.resample_to(nl.Dataset(path).load(b), P)
    apps = [a for a in viz if a in df.columns]
    fr = df["fridge"].to_numpy().astype(float) if "fridge" in df.columns else df[apps[0]].to_numpy().astype(float)
    st, Ls = active_slice(fr, L)
    appslice = {a: np.round(df[a].to_numpy()[st:st + Ls], 1).tolist() for a in apps}
    mains = np.round(df["mains"].to_numpy()[st:st + Ls], 1)
    tot = sum(sum(v) for v in appslice.values()) or 1.0
    share = {a: round(100 * sum(appslice[a]) / tot, 1) for a in apps}
    datasets[name] = {
        "mains": mains.tolist(), "appliances": appslice,
        "stats": {"days": round(span_days, 1), "buildings": nb, "period_s": P,
                  "mains_mean": round(float(mains.mean()), 0), "share": share},
    }
    print(f"{name}: {Ls} pts, {span_days:.0f} days, {nb} buildings, share={share}")

data["datasets"] = datasets
data["period_s"] = P
(DOCS / "demo_data.json").write_text(json.dumps(data))
print(f"\npatched docs/demo_data.json ({(DOCS/'demo_data.json').stat().st_size/1e3:.0f} KB)")
