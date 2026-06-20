import numpy as np

import nilmlite as nl
from nilmlite.convert import make_synthetic


def test_roundtrip_and_resample(tmp_path):
    df, man = make_synthetic(days=2, period_s=6, seed=1)
    nl.save_manifest(tmp_path, man)
    nl.save_building(df, tmp_path / "building1.parquet")

    ds = nl.Dataset(tmp_path)
    assert ds.buildings == [1]
    loaded = ds.load(1)
    assert loaded.columns == [nl.TIME_COL, nl.MAINS_COL, "fridge", "microwave", "dish_washer"]
    assert len(loaded) == len(df)

    rs = nl.resample_to(loaded, 60)
    # 6s -> 60s is a 10x reduction (allow off-by-one at the final bucket)
    assert abs(len(rs) - len(df) // 10) <= 1


def test_windows_alignment():
    mains = np.arange(1000.0)
    target = np.arange(1000.0) * 2
    X, y = nl.seq2point_xy(mains, target, width=99)
    assert X.shape == (1000 - 99 + 1, 99)
    assert y.shape[0] == X.shape[0]
    # midpoint of the first window is index 49 -> target 98
    assert y[0] == target[49]
