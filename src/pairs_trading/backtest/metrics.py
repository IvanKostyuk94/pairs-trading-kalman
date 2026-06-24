import pandas as pd
import numpy as np


def sharpe_ratio(pnl: pd.Series) -> float:
    mu = pnl.mean()
    std = pnl.std()
    annual_trading_days = 252
    return mu / std * np.sqrt(annual_trading_days)


def max_drawdown(pnl: pd.Series) -> float:
    equity = pnl.cumsum()
    run_max = equity.cummax()
    return ((run_max - equity)).max()


def hit_rate(pnl: pd.Series) -> float:
    num_reg = 1e-19
    return (pnl > 0).sum() / (np.abs(pnl) > num_reg).sum()


def average_holding_period(positions: pd.Series) -> float:
    lens = []
    position = positions.iloc[0]
    l = 1
    for i, p in enumerate(positions.iloc[1:], 1):
        if p != position:
            if position != 0:
                lens.append(l)
                l = 1
            position = p
        elif position != 0:
            l += 1
        else:
            continue
    return float(np.mean(lens))


def n_trades(positions: pd.Series) -> int:
    return ((positions.shift(1) == 0) & (positions != 0)).sum()


def summarize(pnl: pd.Series, positions: pd.Series) -> dict:
    return {
        "sharpe": sharpe_ratio(pnl),
        "max_drawdown": max_drawdown(pnl),
        "hit_rate": hit_rate(pnl),
        "average_holding_period": average_holding_period(positions),
        "n_trades": n_trades(positions),
    }
