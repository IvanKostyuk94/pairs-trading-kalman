import yfinance as yf
import pandas as pd

from pathlib import Path


def validate_prices(df: pd.DataFrame, tickers: list[str]) -> None:
    if len(df) == 0:
        raise ValueError("DataFrame is empty")

    if df.isna().sum().sum() != 0:
        raise ValueError("Some tickers have Nan values")

    for ticker in tickers:
        if ticker not in df.columns:
            raise KeyError(f"{ticker} is not present in the DataFrame")

    return None


def load_prices(
    tickers: list[str],
    start: str | None = None,
    end: str | None = None,
    refresh=False,
) -> pd.DataFrame:
    cached_path = Path("data/raw") / f"{'_'.join(tickers)}.parquet"
    if refresh:
        if start is None or end is None:
            raise ValueError("start and end are required when refresh=True")
        df = yf.download(tickers, start, end)
        df = df["Close"]
        if df is None:
            raise ValueError("yfinance returned no data")
        validate_prices(df, tickers)
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cached_path)
        return df
    else:
        return pd.read_parquet(cached_path)
