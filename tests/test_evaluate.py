import nilmlite as nl
from nilmlite.baselines import Linear, Mean
from nilmlite.convert import write_synthetic_dataset


def test_evaluate_and_leaderboard(tmp_path):
    a = tmp_path / "a"
    write_synthetic_dataset(a, buildings=2, days=3, period_s=30, name="a")

    task = nl.Task(
        name="cb", kind="cross_building", appliances=["fridge", "microwave"],
        train=[nl.Split(str(a), 1)], test=[nl.Split(str(a), 2)],
        sample_period_s=60, window=49)

    result = nl.evaluate({"Mean": Mean, "Linear": lambda: Linear(l2=5.0)}, task)
    assert set(result["models"]) == {"Mean", "Linear"}
    for m in result["models"].values():
        assert m["mae"] >= 0
        assert m["params"] is not None
        assert m["infer_ms"] >= 0

    # Linear should learn something: not worse than the constant-mean floor
    assert result["models"]["Linear"]["mae"] <= result["models"]["Mean"]["mae"] + 1e-6

    text = nl.leaderboard_text(result)
    assert "cb" in text and "Mean" in text and "Linear" in text

    html = nl.leaderboard_html(result, tmp_path / "lb.html")
    assert html.exists() and "leaderboard" in html.read_text().lower()
