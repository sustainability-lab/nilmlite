"""Train Seq2Point, put it on the leaderboard, export a portable ONNX, and verify
the ONNX matches PyTorch (the same file the browser will run).

    pip install -e ".[dl,onnx]"
    python examples/train_seq2point.py
"""
from pathlib import Path

import numpy as np

import nilmlite as nl
from nilmlite.baselines import Linear, Mean
from nilmlite.convert import write_synthetic_dataset
from nilmlite.evaluate import build_xy
from nilmlite.models_torch import Seq2Point

ROOT = Path("data/leaderboard")
REDD, UKD = ROOT / "redd_like", ROOT / "ukdale_like"
WINDOW = 99

write_synthetic_dataset(REDD, buildings=3, days=10, period_s=6, name="redd_like")
write_synthetic_dataset(UKD, buildings=3, days=10, period_s=6, name="ukdale_like",
                        appliance_scale=1.4, base_w=90.0, fridge_cycle_s=3300)

APPS = ["fridge", "microwave", "dish_washer"]
MODELS = {
    "Mean": Mean,
    "Linear": lambda: Linear(l2=5.0),
    "Seq2Point": lambda: Seq2Point(window=WINDOW, epochs=5),
}
t2 = nl.Task(name="T2 · cross-building", kind="cross_building", appliances=APPS,
             train=[nl.Split(str(REDD), 1), nl.Split(str(REDD), 2)],
             test=[nl.Split(str(REDD), 3)], window=WINDOW)
t3 = nl.Task(name="T3 · cross-dataset", kind="cross_dataset", appliances=APPS,
             train=[nl.Split(str(REDD), i) for i in (1, 2, 3)],
             test=[nl.Split(str(UKD), 1)], window=WINDOW)

print("training + scoring (CPU, this takes a minute)…")
results = [nl.evaluate(MODELS, t) for t in (t2, t3)]
for r in results:
    print(nl.leaderboard_text(r))
nl.leaderboard_html(results, ROOT / "leaderboard.html")
print(f"\nwrote {ROOT/'leaderboard.html'}")

# --- export a portable fridge model + verify ONNX == PyTorch ---
Xtr, ytr = build_xy(t2, t2.train, "fridge")
Xte, yte = build_xy(t2, t2.test, "fridge")
model = Seq2Point(window=WINDOW, epochs=6).fit(Xtr, ytr)
onnx_path = ROOT / "seq2point_fridge.onnx"
model.export_onnx(onnx_path)

import onnxruntime as ort  # noqa: E402
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
sample = Xte[:512].astype(np.float32)
ort_out = sess.run(["power"], {"mains_window": sample})[0].ravel()
torch_out = model.predict(sample)
max_diff = float(np.max(np.abs(ort_out - torch_out)))
print(f"\nONNX export: {onnx_path}  ({onnx_path.stat().st_size/1e3:.0f} KB, "
      f"{model.n_params:,} params)")
print(f"ONNX vs PyTorch max abs diff: {max_diff:.4f} W  "
      f"-> {'PARITY OK' if max_diff < 1.0 else 'MISMATCH'}")
print(f"fridge MAE on held-out home: {nl.metrics.mae(yte, torch_out):.2f} W")
