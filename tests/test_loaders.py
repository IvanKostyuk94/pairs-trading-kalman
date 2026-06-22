import pandas as pd
import numpy as np
import pytest

from pairs_trading.data.loaders import validate_prices


def make_prices(tickers, n_days=100):
    index = pd.date_range("2020-01-01", periods=n_days, freq="B")
    data = {ticker: np.ones(n_days) * 100 for ticker in tickers}
    return pd.DataFrame(data, index=index)


def test_valid_passes():
    tickers = ["AA", "BB"]
    df = make_prices(tickers)
    validate_prices(df, tickers)


def test_empty():
    df = pd.DataFrame()
    tickers = ["AA", "BB"]
    with pytest.raises(ValueError, match="empty"):
        validate_prices(df, tickers)


def test_nan():
    tickers = ["AA", "BB"]
    df = make_prices(tickers)
    df.iloc[5, 0] = np.nan
    with pytest.raises(ValueError, match="Nan"):
        validate_prices(df, tickers)


def test_ticker():
    tickers = ["AA", "BB", "CC"]
    df = make_prices(tickers[:2])
    with pytest.raises(KeyError, match="CC"):
        validate_prices(df, tickers)
