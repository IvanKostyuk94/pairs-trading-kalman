import pandas as pd
import numpy as np
import pytest
from pairs_trading.models.kalman_hedge import KalmanHedge


def get_synthetic_df_coint(n=500):
    rng = np.random.default_rng(42)
    drift = np.cumsum(rng.normal(size=n))
    y1 = drift + rng.normal(scale=0.1, size=n)
    y2 = 2 * drift + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"y1": y1, "y2": y2})
    return df


def test_beta():
    df = get_synthetic_df_coint()
    hedge1 = KalmanHedge()
    hedge1.fit(df["y2"].iloc[:400], df["y1"].iloc[:400])
    beta1, alpha1, residuals1 = hedge1.transform(
        df["y2"].iloc[400:], df["y1"].iloc[400:]
    )
    hedge2 = KalmanHedge()
    hedge2.fit(df["y2"], df["y1"])
    assert hedge2.beta.iloc[400:].values == pytest.approx(
        beta1.values, rel=1e-2
    )
