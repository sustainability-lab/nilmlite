"""Real REDD: leaderboard (Mean/Linear/Seq2Point) + viz + portable ONNX export.

Needs data/redd (migrate once via tools/nilmtk_to_parquet.py) and the dl/viz/onnx
extras. Writes figures + leaderboard.html + seq2point_fridge.onnx to data/redd_out.
"""
from pathlib import Path

import numpy as np

import nilmlite as nl
from nilmlite.baselines import Linear, Mean
from nilmlite.evaluate import build_xy
from nilmlite.models_torch import Seq2Point

OUT = Path("data/redd_out")
OUT.mkdir(parents=True, exist_ok=True)
APPS = ["fridge", "microwave", "dish_washer"]
WINDOW = 99
ds = nl.Dataset("data/redd")
print("REAL REDD:", ds)

# ---------- viz on a readable one-day slice of building 1 ----------
b1 = nl.resample_to(ds.load(1), 60)
day = b1.slice(9000, 1440)
nl.viz.plot_power(day, title="REDD building 1 — one day (real)").savefig(OUT / "redd_power.png", dpi=130)
nl.viz.plot_signature(b1, "fridge").savefig(OUT / "redd_fridge_signature.png", dpi=130)
nl.viz.plot_day(b1, "fridge").savefig(OUT / "redd_fridge_day.png", dpi=130)
print("wrote power/signature/day figures")

# ---------- leaderboard: cross-building (train b1,b2 -> test b3) ----------
MODELS = {"Mean": Mean, "Linear": lambda: Linear(l2=5.0),
          "Seq2Point": lambda: Seq2Point(window=WINDOW, epochs=6)}
t2 = nl.Task(name="REDD T2 · cross-building", kind="cross_building", appliances=APPS,
             train=[nl.Split("data/redd", 1), nl.Split("data/redd", 2)],
             test=[nl.Split("data/redd", 3)], window=WINDOW,
             note="Real REDD. Train homes 1–2, test unseen home 3.")
print("\ntraining on REAL REDD (CPU)…")
res = nl.evaluate(MODELS, t2)
print(nl.leaderboard_text(res))
nl.leaderboard_html(res, OUT / "leaderboard.html", title="NILMBench · REDD")

# ---------- portable fridge model: predict, plot, export ONNX, verify ----------
Xtr, ytr = build_xy(t2, t2.train, "fridge")
Xte, yte = build_xy(t2, t2.test, "fridge")
model = Seq2Point(window=WINDOW, epochs=8).fit(Xtr, ytr)
pred = model.predict(Xte)
print(f"\nfridge Seq2Point on unseen home 3: MAE {nl.metrics.mae(yte, pred):.2f} W, "
      f"F1 {nl.metrics.f1(yte, pred):.3f}")

sl = slice(1200, 2200)
nl.viz.plot_disaggregation(yte[sl], pred[sl], mains=Xte[sl, WINDOW // 2],
                           title="REDD fridge — Seq2Point vs truth (unseen home)"
                           ).savefig(OUT / "redd_disagg.png", dpi=130)

onnx_path = OUT / "seq2point_fridge.onnx"
model.export_onnx(onnx_path)
import onnxruntime as ort  # noqa: E402
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
s = Xte[:512].astype(np.float32)
diff = float(np.max(np.abs(sess.run(["power"], {"mains_window": s})[0].ravel() - model.predict(s))))
print(f"ONNX {onnx_path} ({onnx_path.stat().st_size/1e3:.0f} KB, {model.n_params:,} params); "
      f"ONNX vs torch max diff {diff:.4f} W -> {'PARITY OK' if diff < 1 else 'MISMATCH'}")
print(f"\nall outputs in {OUT}/")
