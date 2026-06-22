import numpy as np
import pandas as pd
from pairs_trading.stats.cointegration import EGResult, JohansenResult


def get_synthetic_df_coint(n=500):
    rng = np.random.default_rng(42)
    drift = np.cumsum(rng.normal(size=n))
    y1 = drift + rng.normal(scale=0.1, size=n)
    y2 = 2 * drift + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"y1": y1, "y2": y2})
    return df


def get_synthetic_df_not_coint(n=500):
    rng = np.random.default_rng(42)
    drift1 = np.cumsum(rng.normal(size=n))
    drift2 = np.cumsum(rng.normal(size=n))

    y1 = drift1 + rng.normal(scale=0.1, size=n)
    y2 = 2 * drift2 + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"y1": y1, "y2": y2})
    return df


def test_eg_coint():
    df_coint = get_synthetic_df_coint()
    eg_coint = EGResult(df_coint, ["y1", "y2"])
    assert eg_coint.is_cointegrated


def test_eg_not_coint():
    df_not_coint = get_synthetic_df_not_coint()
    eg_not_coint = EGResult(df_not_coint, ["y1", "y2"])
    assert not eg_not_coint.is_cointegrated


def test_johansen_coint():
    df_coint = get_synthetic_df_coint()
    johansen_coint = JohansenResult(df_coint, ["y1", "y2"])
    assert johansen_coint.is_cointegrated


def test_johansen_not_coint():
    df_not_coint = get_synthetic_df_not_coint()
    johansen_not_coint = JohansenResult(df_not_coint, ["y1", "y2"])
    assert not johansen_not_coint.is_cointegrated
