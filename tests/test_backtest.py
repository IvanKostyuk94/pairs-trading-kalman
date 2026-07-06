import pandas as pd
import numpy as np
from pairs_trading.backtest.engine import BacktestEngine
from pairs_trading.config import SIGNAL


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


def test_max_holding_no_position_exceeds_limit():
    df = get_synthetic_df_coint()
    n = 500
    test_signals = pd.Series(np.ones(n))
    test_hedge_ratios = pd.Series(np.ones(n))

    engine = BacktestEngine()
    result = engine.run(
        df["y1"],
        df["y2"],
        signals=test_signals,
        hedge_ratios=test_hedge_ratios,
    )
    assert np.sum(result.positions > 0) == SIGNAL.max_holding


def test_position_sizes_scale_pnl():
    df = get_synthetic_df_coint()
    n = 500
    test_signals = pd.Series(np.ones(n))
    test_hedge_ratios = pd.Series(np.ones(n))
    pos1 = pd.Series(np.ones(n))
    pos2 = 2 * pos1

    engine1 = BacktestEngine()
    result1 = engine1.run(
        df["y1"],
        df["y2"],
        signals=test_signals,
        hedge_ratios=test_hedge_ratios,
        position_sizes=pos1,
    )

    engine2 = BacktestEngine()
    result2 = engine2.run(
        df["y1"],
        df["y2"],
        signals=test_signals,
        hedge_ratios=test_hedge_ratios,
        position_sizes=pos2,
    )

    assert np.allclose(result2.gross_pnl, 2 * result1.gross_pnl)
