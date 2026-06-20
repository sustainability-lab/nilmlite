"""The headline: REAL cross-dataset generalization, US -> India.

Train fridge disaggregation on REDD (US homes), test on:
  * REDD held-out home   (cross-building, same distribution)
  * iAWE  Delhi home     (cross-dataset, US -> India distribution shift)

Needs data/redd and data/iawe (migrated via tools/nilmtk_to_parquet.py) and the
[dl,onnx,viz] extras. Writes leaderboard.html + figures + ONNX to data/iawe_out.
"""
from pathlib import Path

import numpy as np

import nilmlite as nl
from nilmlite.baselines import Linear, Mean
from nilmlite.evaluate import build_xy
from nilmlite.models_torch import Seq2Point

OUT = Path("data/iawe_out"); OUT.mkdir(parents=True, exist_ok=True)
WINDOW = 99
MODELS = {"Mean": Mean, "Linear": lambda: Linear(l2=5.0),
          "Seq2Point": lambda: Seq2Point(window=WINDOW, epochs=6)}

t2 = nl.Task(name="T2 · cross-building (REDD)", kind="cross_building", appliances=["fridge"],
             train=[nl.Split("data/redd", 1), nl.Split("data/redd", 2)],
             test=[nl.Split("data/redd", 3)], window=WINDOW,
             note="Train REDD homes 1-2, test unseen REDD home 3 (same distribution).")
t3 = nl.Task(name="T3 · cross-dataset (REDD → iAWE)", kind="cross_dataset", appliances=["fridge"],
             train=[nl.Split("data/redd", i) for i in (1, 2, 3)],
             test=[nl.Split("data/iawe", 1)], window=WINDOW,
             note="Train REDD (US homes), test iAWE (Delhi home) — US→India shift.")

print("training fridge on REAL REDD, evaluating on REDD + iAWE (CPU)…")
results = [nl.evaluate(MODELS, t) for t in (t2, t3)]
for r in results:
    print(nl.leaderboard_text(r))
nl.leaderboard_html(results, OUT / "leaderboard.html",
                    title="NILMBench · fridge · US → India")
print(f"wrote {OUT/'leaderboard.html'}")

# a Seq2Point trained on REDD, shown predicting on the unseen iAWE home + ONNX
Xtr, ytr = build_xy(t3, t3.train, "fridge")
Xte, yte = build_xy(t3, t3.test, "fridge")
model = Seq2Point(window=WINDOW, epochs=8).fit(Xtr, ytr)
pred = model.predict(Xte)
print(f"\nREDD-trained fridge on iAWE: MAE {nl.metrics.mae(yte, pred):.2f} W, "
      f"F1 {nl.metrics.f1(yte, pred):.3f}")

iawe = nl.resample_to(nl.Dataset("data/iawe").load(1), 60)
nl.viz.plot_power(iawe.slice(2000, 1440), appliances=["fridge", "air_conditioner"],
                  title="iAWE (Delhi) — one day: fridge + AC").savefig(OUT / "iawe_day.png", dpi=130)
sl = slice(3000, 4000)
nl.viz.plot_disaggregation(yte[sl], pred[sl], mains=Xte[sl, WINDOW // 2],
                           title="iAWE fridge — Seq2Point trained on REDD (cross-dataset)"
                           ).savefig(OUT / "iawe_disagg.png", dpi=130)
model.export_onnx(OUT / "seq2point_fridge_redd.onnx")
print(f"all outputs in {OUT}/")
