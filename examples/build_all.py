"""Build everything for the explorer: cross-dataset matrix + model zoo + browser
payload. Auto-includes whatever datasets are present (REDD, iAWE, UK-DALE, REFIT).

  python examples/build_all.py        # needs [dl,onnx] + migrated datasets

Trains once and reuses: Seq2Point per source dataset (matrix rows) is reused for
the REDD zoo/browser model; DAE/GRU/PatchTST train on REDD for the zoo + browser.
Writes docs/demo_data.json (v3) + per-model ONNX into docs/.
"""
import json
import shutil
from pathlib import Path

import numpy as np

import nilmlite as nl
from nilmlite.baselines import Linear, Mean
from nilmlite.matrix import _concat
from nilmlite.models_torch import ZOO

W = 99
P = 60
DOCS = Path("docs"); DOCS.mkdir(exist_ok=True)

# dataset spec: name -> (path, train_buildings, test_building, viz_appliances)
ALL = [
    ("REDD · US",      "data/redd",   [1, 2], 3, ["fridge", "microwave", "dish_washer"]),
    ("UK-DALE · UK",   "data/ukdale", [5],    2, ["fridge", "microwave", "kettle"]),
    ("iAWE · India",   "data/iawe",   [1],    1, ["fridge", "air_conditioner", "washing_machine"]),
]
SPECS = [dict(name=n, path=p, train=tr, test=te, viz=v)
         for (n, p, tr, te, v) in ALL if Path(p, f"building{te}.parquet").exists()]
print("datasets present:", [s["name"] for s in SPECS])

REDD = next(s for s in SPECS if s["path"] == "data/redd")


def fmt(p):
    return f"{p/1e6:.1f}M" if p >= 1e6 else (f"{p/1e3:.0f}K" if p >= 1e3 else str(int(p)))


def active_slice(df, key, L=1000):
    a = df[key].to_numpy().astype(float)
    bs, best = 0, -1
    for s in range(0, max(1, len(a) - L), 120):
        sc = int((a[s + W // 2: s + W // 2 + (L - W + 1)] > 15).sum())
        if sc > best:
            best, bs = sc, s
    return bs, min(L, len(a) - bs)


# ---------------- cross-dataset matrix (fridge): Mean, Linear, Seq2Point ----------------
matrices = []
for mname, mk in [("Mean", Mean), ("Linear", lambda: Linear(l2=5.0)),
                  ("Seq2Point", lambda: ZOO["Seq2Point"](window=W, epochs=6))]:
    print(f"matrix · {mname} …")
    res = nl.cross_dataset_matrix(SPECS, "fridge", mk, period_s=P, window=W)
    res["model"] = mname
    matrices.append(res)
    print(nl.matrix_text(res))

# ---------------- model zoo on REDD fridge -> ONNX + cross-building/cross-dataset ----------------
Xtr, ytr = _concat(REDD["path"], REDD["train"], "fridge", P, W)
test_sets = {s["name"]: _concat(s["path"], [s["test"]], "fridge", P, W) for s in SPECS}
browser_models, zoo_rows = {}, []
for name, cls in ZOO.items():
    print(f"zoo · training {name} on REDD …")
    m = cls(window=W, epochs=6).fit(Xtr, ytr)          # metrics via torch (no onnxruntime here)
    cb = nl.metrics.mae(test_sets[REDD["name"]][1], m.predict(test_sets[REDD["name"]][0]))
    cds = {s["name"]: nl.metrics.mae(test_sets[s["name"]][1], m.predict(test_sets[s["name"]][0]))
           for s in SPECS if s is not REDD}
    onnx_path = DOCS / f"{name.lower()}_fridge_redd.onnx"
    m.export_onnx(onnx_path)                            # torch.onnx.export only
    zoo_rows.append({"name": name, "params": fmt(m.n_params),
                     "cb_mae": round(float(cb), 2),
                     "cd_mae": {k: round(float(v), 2) for k, v in cds.items()}})
    browser_models[f"{name} · ONNX"] = {"type": "onnx", "file": onnx_path.name,
                                        "appliance": "fridge", "params": fmt(m.n_params),
                                        "trained_on": "REDD"}
zoo_rows.sort(key=lambda r: r["cb_mae"])

# Linear + Mean as browser-runnable (no onnx needed)
lin = Linear(l2=5.0).fit(Xtr, ytr); mean = Mean().fit(Xtr, ytr)
browser_models["Linear · ridge"] = {"type": "linear", "w": np.round(lin.w_, 5).tolist(),
                                    "appliance": "fridge", "params": "100", "trained_on": "REDD"}
browser_models["Mean · constant"] = {"type": "const", "value": round(float(mean.value_), 2),
                                     "appliance": "fridge", "params": "1", "trained_on": "REDD"}

# ---------------- dataset viz slices ----------------
datasets = {}
for s in SPECS:
    df = nl.resample_to(nl.Dataset(s["path"]).load(s["test"]), P)
    apps = [a for a in s["viz"] if a in df.columns]
    st, L = active_slice(df, "fridge")
    datasets[s["name"]] = {
        "mains": np.round(df["mains"].to_numpy()[st:st + L], 1).tolist(),
        "appliances": {a: np.round(df[a].to_numpy()[st:st + L], 1).tolist() for a in apps},
    }

payload = {"window": W, "on_threshold": 15.0, "datasets": datasets,
           "models": browser_models, "matrices": matrices,
           "zoo": {"redd_name": REDD["name"], "rows": zoo_rows}}
(DOCS / "demo_data.json").write_text(json.dumps(payload))
print(f"\nwrote docs/demo_data.json ({(DOCS/'demo_data.json').stat().st_size/1e3:.0f} KB)")
print("browser models:", list(browser_models))
print("zoo:", [(r["name"], r["cb_mae"], r["cd_mae"]) for r in zoo_rows])
