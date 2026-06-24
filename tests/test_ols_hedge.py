import numpy as np
import pandas as pd
import pytest
from pairs_trading.models.ols_hedge import OLSHedge


def get_synthetic_df_coint(n=500):
    rng = np.random.default_rng(42)
    drift = np.cumsum(rng.normal(size=n))
    y1 = drift + rng.normal(scale=0.1, size=n)
    y2 = 2 * drift + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"y1": y1, "y2": y2})
    return df


def test_beta():
    df = get_synthetic_df_coint()
    hedge = OLSHedge()
    hedge.fit(df["y2"], df["y1"])
    assert hedge.beta == pytest.approx(2, rel=1e-2)


def test_spread():
    df = get_synthetic_df_coint()
    hedge = OLSHedge()
    hedge.fit(df["y2"], df["y1"])
    resid = hedge.spread(df["y2"], df["y1"])
    assert np.allclose(hedge.residuals, resid)
