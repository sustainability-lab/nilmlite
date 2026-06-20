import numpy as np
import pytest

from nilmlite import metrics


def test_perfect_prediction():
    y = np.array([0.0, 100.0, 0.0, 2000.0, 50.0])
    r = metrics.report(y, y, threshold=15)
    assert r["mae"] == 0.0
    assert r["rmse"] == 0.0
    assert r["sae"] == 0.0
    assert r["nde"] == 0.0
    assert r["f1"] == 1.0
    assert r["accuracy"] == 1.0


def test_mae_rmse_values():
    yt = np.array([0.0, 10.0, 20.0])
    yp = np.array([0.0, 12.0, 17.0])
    assert metrics.mae(yt, yp) == pytest.approx((0 + 2 + 3) / 3)
    assert metrics.rmse(yt, yp) == pytest.approx(np.sqrt((0 + 4 + 9) / 3))


def test_f1_threshold():
    yt = np.array([0.0, 0.0, 100.0, 100.0])
    yp = np.array([0.0, 100.0, 100.0, 0.0])   # 1 TP, 1 FP, 1 FN
    assert metrics.precision(yt, yp, 15) == pytest.approx(0.5)
    assert metrics.recall(yt, yp, 15) == pytest.approx(0.5)
    assert metrics.f1(yt, yp, 15) == pytest.approx(0.5)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.mae([1, 2, 3], [1, 2])
