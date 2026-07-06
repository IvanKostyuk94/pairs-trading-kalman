import pandas as pd
import numpy as np
from pairs_trading.stats.garch import GARCHVolatility, compute_position_sizes
from pairs_trading.models.ols_hedge import OLSHedge


def get_synthetic_df_coint(n=500):
    rng = np.random.default_rng(42)
    drift = np.cumsum(rng.normal(size=n))
    y1 = drift + rng.normal(scale=1.0, size=n)
    y2 = 2 * drift + rng.normal(scale=1.0, size=n)
    df = pd.DataFrame({"y1": y1, "y2": y2})
    return df


def test_sigma_positive():
    df = get_synthetic_df_coint()
    hedge = OLSHedge()
    hedge.fit(df["y2"], df["y1"])
    resid = hedge.spread(df["y2"], df["y1"])

    garch = GARCHVolatility()
    garch.fit(resid)
    assert (garch.sigma_t > 0).all()


def test_sigma_index():
    df = get_synthetic_df_coint()
    hedge = OLSHedge()
    hedge.fit(df["y2"], df["y1"])
    resid = hedge.spread(df["y2"], df["y1"])

    garch = GARCHVolatility()
    garch.fit(resid)
    assert (garch.sigma_t.index == resid.index).all()


def test_transform_length():
    df = get_synthetic_df_coint()
    hedge = OLSHedge()
    hedge.fit(df["y2"], df["y1"])
    resid = hedge.spread(df["y2"], df["y1"])

    resid_train = resid.iloc[:400]
    resid_test = resid.iloc[400:]
    garch = GARCHVolatility()
    garch.fit(resid_train)
    sigmat = garch.transform(resid_test)
    assert len(sigmat) == len(resid_test)


def test_transform_index():
    df = get_synthetic_df_coint()
    hedge = OLSHedge()
    hedge.fit(df["y2"], df["y1"])
    resid = hedge.spread(df["y2"], df["y1"])

    resid_train = resid.iloc[:400]
    resid_test = resid.iloc[400:]
    garch = GARCHVolatility()
    garch.fit(resid_train)
    sigmat = garch.transform(resid_test)
    assert (sigmat.index == resid_test.index).all()


def test_transform_sigma_positive():
    df = get_synthetic_df_coint()
    hedge = OLSHedge()
    hedge.fit(df["y2"], df["y1"])
    resid = hedge.spread(df["y2"], df["y1"])

    resid_train = resid.iloc[:400]
    resid_test = resid.iloc[400:]
    garch = GARCHVolatility()
    garch.fit(resid_train)
    sigmat = garch.transform(resid_test)
    assert (sigmat > 0).all()


def test_compute_position_sizes_inverse():
    df = get_synthetic_df_coint()
    hedge = OLSHedge()
    hedge.fit(df["y2"], df["y1"])
    resid = hedge.spread(df["y2"], df["y1"])

    garch = GARCHVolatility()
    garch.fit(resid)
    sigmat = garch.sigma_t
    sigmat2 = 2 * sigmat
    pos1 = compute_position_sizes(sigmat)
    pos2 = compute_position_sizes(sigmat2)

    assert np.allclose(2 * pos2.values, pos1.values)
