"""Build the multi-appliance explorer payload: per-appliance cross-dataset matrix
+ per-appliance in-browser models + the architecture zoo (fridge).

  python examples/build_all.py     # needs [dl] + migrated datasets (REDD/UK-DALE/iAWE)

Compute is reused: each appliance's Seq2Point matrix model (trained per source
dataset) doubles as that appliance's in-browser inference model. Only fridge gets
the full 4-architecture zoo. Writes docs/demo_data.json + per-(model,appliance) ONNX.
"""
import json
from pathlib import Path

import numpy as np

import nilmlite as nl
from nilmlite.baselines import Linear, Mean
from nilmlite.matrix import _concat
from nilmlite.models_torch import ZOO

W, P = 99, 60
EP, MAXTR = 5, 80_000
DOCS = Path("docs"); DOCS.mkdir(exist_ok=True)

ALL = [
    ("REDD · US",    "data/redd",   [1, 2], 3, ["fridge", "microwave", "dish_washer"]),
    ("UK-DALE · UK", "data/ukdale", [5],    2, ["fridge", "microwave", "dish_washer", "washing_machine", "kettle"]),
    ("iAWE · India", "data/iawe",   [1],    1, ["fridge", "air_conditioner", "washing_machine"]),
]
SPECS = [dict(name=n, path=p, train=tr, test=te, viz=v)
         for (n, p, tr, te, v) in ALL if Path(p, f"building{te}.parquet").exists()]
APPS = ["fridge", "microwave", "dish_washer", "washing_machine"]
HOME = {"fridge": "REDD · US", "microwave": "REDD · US",
        "dish_washer": "REDD · US", "washing_machine": "UK-DALE · UK"}
SHORT = {"Seq2Point": "seq2point", "DAE": "dae", "GRU": "gru", "PatchTST": "patchtst"}


def fmt(p):
    return f"{p/1e6:.1f}M" if p >= 1e6 else (f"{p/1e3:.0f}K" if p >= 1e3 else str(int(p)))


def cols(path, b):
    p = Path(path) / f"building{b}.parquet"
    return set(nl.load_building(p).columns) if p.exists() else set()


def has(spec, app):
    return app in cols(spec["path"], spec["test"]) and all(app in cols(spec["path"], b) for b in spec["train"])


def active_slice(df, key, L=1000):
    a = df[key].to_numpy().astype(float)
    bs, best = 0, -1
    for s in range(0, max(1, len(a) - L), 120):
        sc = int((a[s + W // 2: s + W // 2 + (L - W + 1)] > 15).sum())
        if sc > best:
            best, bs = sc, s
    return bs, min(L, len(a) - bs)


def mae_of(model, xy):
    return round(float(nl.metrics.mae(xy[1], model.predict(xy[0]))), 2)


# ---------------- per-appliance matrix (Mean/Linear/Seq2Point) + reusable models ----------------
matrices, seq_models = {}, {}
for app in APPS:
    dsets = [s for s in SPECS if has(s, app)]
    if len(dsets) < 2:
        print(f"skip matrix {app}: <2 datasets"); continue
    names = [s["name"] for s in dsets]
    test_xy = {s["name"]: _concat(s["path"], [s["test"]], app, P, W) for s in dsets}
    app_mats = []
    for mname, mk in [("Mean", Mean), ("Linear", lambda: Linear(l2=5.0)),
                      ("Seq2Point", lambda: ZOO["Seq2Point"](window=W, epochs=EP, max_train=MAXTR))]:
        print(f"matrix · {app} · {mname} …")
        M = []
        for src in dsets:
            model = mk().fit(*_concat(src["path"], src["train"], app, P, W))
            if mname == "Seq2Point":
                seq_models[(app, src["name"])] = model
            M.append([mae_of(model, test_xy[t["name"]]) for t in dsets])
        app_mats.append({"model": mname, "names": names, "matrix": M})
    matrices[app] = app_mats

# ---------------- per-appliance in-browser models (+ fridge zoo) ----------------
models_by_app, zoo_rows = {}, []
for app in APPS:
    home = HOME[app]
    if (app, home) not in seq_models:
        continue
    dsets = [s for s in SPECS if has(s, app)]
    test_xy = {s["name"]: _concat(s["path"], [s["test"]], app, P, W) for s in dsets}
    hs = next(s for s in SPECS if s["name"] == home)
    Xtr, ytr = _concat(hs["path"], hs["train"], app, P, W)

    def metrics_for(m):
        cb = mae_of(m, test_xy[home])
        cd = {s["name"]: mae_of(m, test_xy[s["name"]]) for s in dsets if s["name"] != home}
        return cb, cd

    mm = {}
    sp = seq_models[(app, home)]
    sp.export_onnx(DOCS / f"seq2point_{app}.onnx")
    mm["Seq2Point · ONNX"] = {"type": "onnx", "file": f"seq2point_{app}.onnx",
                              "appliance": app, "params": fmt(sp.n_params), "trained_on": home}
    if app == "fridge":
        cb, cd = metrics_for(sp)
        zoo_rows.append({"name": "Seq2Point", "params": fmt(sp.n_params), "cb_mae": cb, "cd_mae": cd})
        for name in ["DAE", "GRU", "PatchTST"]:
            print(f"zoo · {name} · fridge …")
            m = ZOO[name](window=W, epochs=EP, max_train=MAXTR).fit(Xtr, ytr)
            m.export_onnx(DOCS / f"{SHORT[name]}_{app}.onnx")
            cb, cd = metrics_for(m)
            zoo_rows.append({"name": name, "params": fmt(m.n_params), "cb_mae": cb, "cd_mae": cd})
            mm[f"{name} · ONNX"] = {"type": "onnx", "file": f"{SHORT[name]}_{app}.onnx",
                                    "appliance": app, "params": fmt(m.n_params), "trained_on": home}
    lin = Linear(l2=5.0).fit(Xtr, ytr); mean = Mean().fit(Xtr, ytr)
    mm["Linear · ridge"] = {"type": "linear", "w": np.round(lin.w_, 5).tolist(),
                            "appliance": app, "params": "100", "trained_on": home}
    mm["Mean · constant"] = {"type": "const", "value": round(float(mean.value_), 2),
                             "appliance": app, "params": "1", "trained_on": home}
    models_by_app[app] = mm
zoo_rows.sort(key=lambda r: r["cb_mae"])

# ---------------- dataset viz slices (all appliances) ----------------
datasets = {}
for s in SPECS:
    df = nl.resample_to(nl.Dataset(s["path"]).load(s["test"]), P)
    apps = [a for a in s["viz"] if a in df.columns]
    st, L = active_slice(df, "fridge" if "fridge" in df.columns else apps[0])
    datasets[s["name"]] = {
        "mains": np.round(df["mains"].to_numpy()[st:st + L], 1).tolist(),
        "appliances": {a: np.round(df[a].to_numpy()[st:st + L], 1).tolist() for a in apps},
    }

payload = {"window": W, "on_threshold": 15.0,
           "appliances": list(models_by_app), "home": HOME,
           "datasets": datasets, "models": models_by_app, "matrices": matrices,
           "zoo": {"redd_name": "REDD · US", "rows": zoo_rows}}
(DOCS / "demo_data.json").write_text(json.dumps(payload))
print(f"\nwrote docs/demo_data.json ({(DOCS/'demo_data.json').stat().st_size/1e3:.0f} KB)")
print("appliances:", list(models_by_app))
for app in models_by_app:
    print(f"  {app}: models={list(models_by_app[app])}")
print("zoo:", [(r["name"], r["cb_mae"]) for r in zoo_rows])
