import pandas as pd
import numpy as np
from pairs_trading.backtest.engine import BacktestEngine


def get_synthetic_df_coint(n=500):
    rng = np.random.default_rng(42)
    drift = np.cumsum(rng.normal(size=n))
    y1 = drift + rng.normal(scale=0.1, size=n)
    y2 = 2 * drift + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"y1": y1, "y2": y2})
    return df


def test_flat_signals():
    df = get_synthetic_df_coint()
    signals = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    hedge_ratios = pd.Series(np.ones(len(df)), index=df.index)
    engine = BacktestEngine()
    result = engine.run(
        df["y2"], df["y1"], signals=signals, hedge_ratios=hedge_ratios
    )
    assert result.gross_pnl.sum() == 0


def test_net_gross():
    df = get_synthetic_df_coint()
    signals = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    signals.iloc[100:105] = 1
    signals.iloc[200:220] = -1
    hedge_ratios = pd.Series(np.ones(len(df)), index=df.index)
    engine = BacktestEngine()
    result = engine.run(
        df["y2"], df["y1"], signals=signals, hedge_ratios=hedge_ratios
    )
    assert result.gross_pnl.sum() >= result.pnl.sum()


def test_net_gross_same():
    df = get_synthetic_df_coint()
    signals = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    signals.iloc[100:105] = 1
    signals.iloc[200:220] = -1
    hedge_ratios = pd.Series(np.ones(len(df)), index=df.index)
    engine = BacktestEngine()
    result = engine.run(
        df["y2"],
        df["y1"],
        signals=signals,
        hedge_ratios=hedge_ratios,
        cost=0.0,
    )
    assert result.gross_pnl.sum() == result.pnl.sum()
