import numpy as np
import pandas as pd

from dd_coherence import DDParams, run_dd_coherence


def test_smoke_runs_and_shapes_are_consistent():
    np.random.seed(0)
    n1, n2 = 200, 200
    x = np.r_[np.random.normal(0, 1, n1), np.random.normal(3, 1.5, n2)]
    y = np.r_[np.random.normal(0, 1, n1), np.random.normal(0, 1, n2)]
    df = pd.DataFrame({"x": x, "y": y})

    params = DDParams(m=20, k=10, r=3, cooldown=30, N0=120, Lref=120, theta=0.70, kappa=0.05, kappa_C=0.05)
    res = run_dd_coherence(df, cols=["x", "y"], params=params)

    assert isinstance(res, dict)
    assert "outputs" in res

    outs = res["outputs"]
    ts = outs["transitions_t_index"]
    assert isinstance(ts, list)
    assert all(isinstance(t, int) for t in ts)
    assert all(0 <= t < len(df) for t in ts)

    series = outs["series"]
    assert len(series["t"]) == len(df)
    assert len(series["Phi"]) == len(df)
    assert len(series["F"]) == len(df)
    assert len(series["Cand"]) == len(df)

    assert set(int(v) for v in series["Cand"] if v is not None).issubset({0, 1})
