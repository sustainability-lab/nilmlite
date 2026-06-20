"""Build two synthetic 'datasets', define cross-building and cross-dataset tasks,
evaluate the built-in baselines, and emit a leaderboard (text + HTML).

    python examples/run_leaderboard.py

This is the benchmark spine: same harness, sealed test buildings, every metric.
Deep models (see examples/train_seq2point.py) drop straight into `MODELS`.
"""
from pathlib import Path

import nilmlite as nl
from nilmlite.baselines import Linear, Mean
from nilmlite.convert import write_synthetic_dataset

ROOT = Path("data/leaderboard")
REDD = ROOT / "redd_like"      # dataset A
UKD = ROOT / "ukdale_like"     # dataset B: shifted distribution (cross-dataset)

# two distinct datasets, 3 buildings each
write_synthetic_dataset(REDD, buildings=3, days=20, period_s=6,
                        name="redd_like", source="synthetic-A")
write_synthetic_dataset(UKD, buildings=3, days=20, period_s=6,
                        name="ukdale_like", source="synthetic-B",
                        appliance_scale=1.4, base_w=90.0, fridge_cycle_s=3300)

APPLIANCES = ["fridge", "microwave", "dish_washer"]
MODELS = {"Mean": Mean, "Linear": lambda: Linear(l2=5.0)}

# T2: train on REDD homes 1-2, test on unseen REDD home 3 (same dataset)
t2 = nl.Task(
    name="T2 · cross-building", kind="cross_building", appliances=APPLIANCES,
    train=[nl.Split(str(REDD), 1), nl.Split(str(REDD), 2)],
    test=[nl.Split(str(REDD), 3)],
    note="Train and test homes share a dataset/distribution.")

# T3: train on REDD, test on UK-DALE-like (unseen dataset)
t3 = nl.Task(
    name="T3 · cross-dataset", kind="cross_dataset", appliances=APPLIANCES,
    train=[nl.Split(str(REDD), 1), nl.Split(str(REDD), 2), nl.Split(str(REDD), 3)],
    test=[nl.Split(str(UKD), 1)],
    note="Test dataset has a different appliance/base-load distribution.")

results = [nl.evaluate(MODELS, t) for t in (t2, t3)]
for r in results:
    print(nl.leaderboard_text(r))

out = nl.leaderboard_html(results, ROOT / "leaderboard.html")
print(f"\nwrote {out}")
