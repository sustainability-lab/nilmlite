"""Build the GitHub Pages demo payload from the cross-dataset run's ONNX.

Reuses data/iawe_out/seq2point_fridge_redd.onnx (no retraining): computes the
leaderboard (Mean/Linear via fit, Seq2Point via ONNX inference) and extracts a
contiguous iAWE mains slice for in-browser inference. Writes docs/demo_data.json
and copies the ONNX into docs/.
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
OUT = Path("data/iawe_out")
DOCS = Path("docs")
ONNX = OUT / "seq2point_fridge_redd.onnx"

t2 = nl.Task(name="cross-building (REDD)", kind="cross_building", appliances=["fridge"],
             train=[nl.Split("data/redd", 1), nl.Split("data/redd", 2)],
             test=[nl.Split("data/redd", 3)], window=W)
t3 = nl.Task(name="cross-dataset (REDD → iAWE)", kind="cross_dataset", appliances=["fridge"],
             train=[nl.Split("data/redd", i) for i in (1, 2, 3)],
             test=[nl.Split("data/iawe", 1)], window=W)


def n_params(p):
    m = onnx.load(str(p))
    return int(sum(int(np.prod(i.dims)) for i in m.graph.initializer))


def fmt(p):
    return f"{p/1e6:.1f}M" if p >= 1e6 else (f"{p/1e3:.0f}K" if p >= 1e3 else str(p))


sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
s2p = lambda X: sess.run(["power"], {"mains_window": X.astype(np.float32)})[0].ravel()
PARAMS = n_params(ONNX)

boards, s2p_mae = [], {}
for t in (t2, t3):
    Xtr, ytr = build_xy(t, t.train, "fridge")
    Xte, yte = build_xy(t, t.test, "fridge")
    rows = []
    for name, mk in [("Mean", Mean), ("Linear", lambda: Linear(l2=5.0))]:
        m = mk().fit(Xtr, ytr); p = m.predict(Xte)
        rows.append({"name": name, "mae": nl.metrics.mae(yte, p),
                     "f1": float(nl.metrics.f1(yte, p)), "params": m.n_params})
    p = s2p(Xte)
    s2p_mae[t.kind] = nl.metrics.mae(yte, p)
    rows.append({"name": "Seq2Point", "mae": nl.metrics.mae(yte, p),
                 "f1": float(nl.metrics.f1(yte, p)), "params": PARAMS})
    rows.sort(key=lambda r: r["mae"])
    for r in rows:
        r["params"] = fmt(r["params"]); r["mae"] = round(r["mae"], 2); r["f1"] = round(r["f1"], 3)
    boards.append({"title": t.name, "kind": t.kind, "models": rows})

a, b = s2p_mae["cross_building"], s2p_mae["cross_dataset"]
gap = (f"<b>Seq2Point</b> degrades <b>{a:.1f} W → {b:.1f} W</b> "
       f"({(b-a)/a*100:+.0f}%) going from a new <i>home</i> to a new <i>dataset</i> "
       f"(US → India). Cross-dataset generalization is the open problem.")

# --- contiguous iAWE slice with fridge activity for the live demo ---
iawe = nl.resample_to(nl.Dataset("data/iawe").load(1), 60)
mains = iawe["mains"].to_numpy().astype(float)
fridge = iawe["fridge"].to_numpy().astype(float)
L = 1200
best_s, best = 0, -1
for s in range(0, len(mains) - L, 150):
    score = int((fridge[s + W // 2: s + W // 2 + (L - W + 1)] > 15).sum())
    if score > best:
        best, best_s = score, s
s = best_s
mains_1d = np.round(mains[s:s + L], 1)
true_mid = np.round(fridge[s + W // 2: s + W // 2 + (L - W + 1)], 1)

payload = {
    "window": W, "on_threshold": 15.0,
    "trained_on": "REDD (US homes)", "tested_on": "iAWE (Delhi home)",
    "onnx": ONNX.name, "params_str": fmt(PARAMS),
    "mains_1d": mains_1d.tolist(), "true_mid": true_mid.tolist(),
    "boards": boards, "gap": gap,
}
DOCS.mkdir(exist_ok=True)
shutil.copy(ONNX, DOCS / ONNX.name)
(DOCS / "demo_data.json").write_text(json.dumps(payload))
kb = (DOCS / "demo_data.json").stat().st_size / 1e3
print(f"wrote docs/demo_data.json ({kb:.0f} KB), copied {ONNX.name} ({(DOCS/ONNX.name).stat().st_size/1e3:.0f} KB)")
print("gap:", gap.replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", ""))
for bd in boards:
    print(" ", bd["kind"], "->", [(m["name"], m["mae"]) for m in bd["models"]])
