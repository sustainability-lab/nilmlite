import nilmlite as nl
from nilmlite.baselines import Mean
from nilmlite.convert import write_synthetic_dataset


def test_cross_dataset_matrix(tmp_path):
    a = tmp_path / "a"; b = tmp_path / "b"
    write_synthetic_dataset(a, buildings=2, days=2, period_s=30, name="a")
    write_synthetic_dataset(b, buildings=2, days=2, period_s=30, name="b",
                            appliance_scale=1.5, base_w=90)

    specs = [
        {"name": "A", "path": str(a), "train": [1], "test": 2},
        {"name": "B", "path": str(b), "train": [1], "test": 2},
    ]
    res = nl.cross_dataset_matrix(specs, "fridge", Mean, period_s=60, window=49)
    assert res["names"] == ["A", "B"]
    assert len(res["matrix"]) == 2 and len(res["matrix"][0]) == 2
    assert all(v is not None and v >= 0 for row in res["matrix"] for v in row)
    txt = nl.matrix_text(res)
    assert "fridge" in txt and "A" in txt and "B" in txt
