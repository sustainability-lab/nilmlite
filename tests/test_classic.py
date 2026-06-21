import numpy as np

import nilmlite as nl
from nilmlite.classic import CombinatorialOptimization, evaluate_joint


def test_co_recovers_two_appliances():
    rng = np.random.default_rng(0)
    n = 4000
    # two appliances with clear on/off levels
    fridge = np.where(rng.random(n) < 0.4, 150.0, 0.0)
    kettle = np.where(rng.random(n) < 0.1, 2000.0, 0.0)
    mains = fridge + kettle + rng.normal(0, 3, n)

    co = CombinatorialOptimization(n_states=2).fit({"fridge": fridge, "kettle": kettle})
    pred = co.disaggregate(mains)
    assert set(pred) == {"fridge", "kettle"}
    assert pred["fridge"].shape == (n,)

    res = evaluate_joint(co, mains, {"fridge": fridge, "kettle": kettle})
    # CO should disaggregate these well-separated loads accurately
    assert res["kettle"]["f1"] > 0.9
    assert res["fridge"]["mae"] < 30
    assert co.n_params > 0
