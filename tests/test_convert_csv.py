import polars as pl

import nilmlite as nl
from nilmlite.convert import from_power_csvs


def test_from_power_csvs(tmp_path):
    base = 1370000000
    ts = list(range(base, base + 600))          # 600 s @ 1 Hz
    for nm, w in [("m.csv", 300.0), ("f.csv", 120.0), ("a.csv", 1500.0)]:
        pl.DataFrame({"timestamp": ts, "W": [w] * 600,
                      "V": [230.0] * 600}).write_csv(tmp_path / nm)

    out = from_power_csvs(
        {"mains": tmp_path / "m.csv", "fridge": tmp_path / "f.csv",
         "air_conditioner": tmp_path / "a.csv"},
        tmp_path / "ds", period_s=6, name="csv_ds")

    ds = nl.Dataset(out)
    assert ds.manifest.appliances == ["fridge", "air_conditioner"]
    b = ds.load(1)
    assert b.columns[:2] == [nl.TIME_COL, nl.MAINS_COL]
    assert len(b) == 600 // 6 + 1
    assert abs(float(b["fridge"].mean()) - 120.0) < 1e-3
    assert abs(float(b["air_conditioner"].mean()) - 1500.0) < 1e-3
