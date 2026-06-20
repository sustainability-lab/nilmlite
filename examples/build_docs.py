"""Build the GitHub Pages explorer payload (docs/demo_data.json) + copy the ONNX.

Produces, with NO Seq2Point retraining (reuses data/iawe_out/seq2point_fridge_redd.onnx):
  * dataset slices (mains + appliances) for in-browser viz
  * runnable fridge models: Seq2Point (ONNX), Linear (weights), Mean (constant)
  * an honest leaderboard (cross-building vs cross-dataset)
All models are trained on REDD; the page lets you test them on REDD or iAWE.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

import nilmlite as nl
from nilmlite.baselines import Linear, Mean
from nilmlite.evaluate import build_xy

W = 99
DOCS = Path("docs")
ONNX = Path("data/iawe_out/seq2point_fridge_redd.onnx")

t2 = nl.Task(name="cross-building (REDD)", kind="cross_building", appliances=["fridge"],
             train=[nl.Split("data/redd", 1), nl.Split("data/redd", 2)],
             test=[nl.Split("data/redd", 3)], window=W)
t3 = nl.Task(name="cross-dataset (REDD → iAWE)", kind="cross_dataset", appliances=["fridge"],
             train=[nl.Split("data/redd", i) for i in (1, 2, 3)],
             test=[nl.Split("data/iawe", 1)], window=W)


def fmt(p):
    return f"{p/1e6:.1f}M" if p >= 1e6 else (f"{p/1e3:.0f}K" if p >= 1e3 else str(int(p)))


def n_params_onnx(p):
    m = onnx.load(str(p))
    return int(sum(int(np.prod(i.dims)) for i in m.graph.initializer))


def active_slice(df, key, L=1000):
    """Contiguous slice (length L) richest in `key` activity."""
    a = df[key].to_numpy().astype(float)
    best_s, best = 0, -1
    for s in range(0, max(1, len(a) - L), 120):
        sc = int((a[s + W // 2: s + W // 2 + (L - W + 1)] > 15).sum())
        if sc > best:
            best, best_s = sc, s
    return best_s, min(L, len(a) - best_s)


def pack_dataset(path, building, appliances, label, L=1000):
    df = nl.resample_to(nl.Dataset(path).load(building), 60)
    appliances = [a for a in appliances if a in df.columns]
    s, L = active_slice(df, "fridge", L)
    out = {"mains": np.round(df["mains"].to_numpy()[s:s + L], 1).tolist(),
           "appliances": {a: np.round(df[a].to_numpy()[s:s + L], 1).tolist() for a in appliances}}
    return label, out


# ---- models (all trained on REDD fridge) ----
Xtr, ytr = build_xy(t3, t3.train, "fridge")
mean_model = Mean().fit(Xtr, ytr)
lin = Linear(l2=5.0).fit(Xtr, ytr)
PARAMS = n_params_onnx(ONNX)
models = {
    "Seq2Point  ·  deep (ONNX)": {"type": "onnx", "file": ONNX.name,
                                  "appliance": "fridge", "params": fmt(PARAMS), "trained_on": "REDD"},
    "Linear  ·  ridge": {"type": "linear", "w": np.round(lin.w_, 5).tolist(),
                         "appliance": "fridge", "params": fmt(lin.n_params), "trained_on": "REDD"},
    "Mean  ·  constant": {"type": "const", "value": round(float(mean_model.value_), 2),
                          "appliance": "fridge", "params": "1", "trained_on": "REDD"},
}

# ---- leaderboard (honest) ----
sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
s2p = lambda X: sess.run(["power"], {"mains_window": X.astype(np.float32)})[0].ravel()
boards, s2p_mae = [], {}
for t in (t2, t3):
    Xa, ya = build_xy(t, t.train, "fridge")
    Xe, ye = build_xy(t, t.test, "fridge")
    rows = []
    for nm, mk in [("Mean", Mean), ("Linear", lambda: Linear(l2=5.0))]:
        m = mk().fit(Xa, ya); p = m.predict(Xe)
        rows.append({"name": nm, "mae": round(nl.metrics.mae(ye, p), 2),
                     "f1": round(float(nl.metrics.f1(ye, p)), 3), "params": fmt(m.n_params)})
    p = s2p(Xe); s2p_mae[t.kind] = nl.metrics.mae(ye, p)
    rows.append({"name": "Seq2Point", "mae": round(s2p_mae[t.kind], 2),
                 "f1": round(float(nl.metrics.f1(ye, p)), 3), "params": fmt(PARAMS)})
    rows.sort(key=lambda r: r["mae"])
    boards.append({"title": t.name, "kind": t.kind, "models": rows})

a, b = s2p_mae["cross_building"], s2p_mae["cross_dataset"]
gap = (f"Seq2Point wins <i>within</i> REDD (<b>{a:.1f} W</b>) but cross-dataset on iAWE it "
       f"degrades to <b>{b:.1f} W</b> — worse than the Mean baseline. "
       f"Cross-dataset generalization (US → India) is unsolved.")

datasets = dict([
    pack_dataset("data/redd", 3, ["fridge", "microwave", "dish_washer"], "REDD · home 3 (US)"),
    pack_dataset("data/redd", 1, ["fridge", "microwave", "dish_washer"], "REDD · home 1 (US)"),
    pack_dataset("data/iawe", 1, ["fridge", "air_conditioner", "washing_machine"], "iAWE · Delhi home (India)"),
])

DOCS.mkdir(exist_ok=True)
shutil.copy(ONNX, DOCS / ONNX.name)
payload = {"window": W, "on_threshold": 15.0, "datasets": datasets,
           "models": models, "boards": boards, "gap": gap}
(DOCS / "demo_data.json").write_text(json.dumps(payload))
print(f"wrote docs/demo_data.json ({(DOCS/'demo_data.json').stat().st_size/1e3:.0f} KB)")
print("datasets:", list(datasets))
print("models:", list(models))
for bd in boards:
    print(" ", bd["kind"], [(m["name"], m["mae"]) for m in bd["models"]])
