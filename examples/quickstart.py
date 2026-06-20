"""End-to-end in ~20 lines: generate -> save -> load -> window -> baseline -> metrics."""
from pathlib import Path

import nilmlite as nl
from nilmlite.baselines import Mean
from nilmlite.convert import make_synthetic

OUT = Path("data/quickstart")

# 1. generate a synthetic household and persist it as NILM-Parquet
df, manifest = make_synthetic(days=10, period_s=6, seed=0)
nl.save_manifest(OUT, manifest)
nl.save_building(df, OUT / "building1.parquet")

# 2. reload through the Dataset API
ds = nl.Dataset(OUT)
print(ds)
data = ds.load(1)

# 3. resample 6s -> 60s and pull the arrays
rs = nl.resample_to(data, 60)
mains = rs["mains"].to_numpy()
fridge = rs["fridge"].to_numpy()

# 4. seq2point windows (what an ONNX/torch model would consume)
X, y = nl.seq2point_xy(mains, fridge, width=99)
print("windows:", X.shape, "targets:", y.shape)

# 5. a baseline + the full metric suite
pred = Mean().fit(mains, fridge).predict(mains)
for k, v in nl.report(fridge, pred).items():
    print(f"  {k:10s} {v:8.3f}")
