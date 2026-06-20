import numpy as np
import pytest

torch = pytest.importorskip("torch")
from nilmlite.models_torch import ZOO  # noqa: E402


@pytest.mark.parametrize("name", list(ZOO))
def test_zoo_fit_predict(name):
    rng = np.random.default_rng(0)
    W, N = 99, 300
    X = (rng.random((N, W)) * 400).astype("float32")
    y = (X[:, W // 2] * 0.3).astype("float32")
    m = ZOO[name](window=W, epochs=1, max_train=N).fit(X, y)
    p = m.predict(X[:32])
    assert p.shape == (32,)
    assert m.n_params and m.n_params > 0
    assert np.all(p >= 0)            # clamped non-negative
