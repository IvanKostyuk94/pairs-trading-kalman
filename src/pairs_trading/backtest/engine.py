import pandas as pd
import numpy as np
from pairs_trading.config import COST
from pairs_trading.backtest.metrics import summarize


class BacktestResult:
    def __init__(self, pnl, positions, gross_pnl, cost_series, metrics):
        self.pnl = pnl
        self.positions = positions
        self.gross_pnl = gross_pnl
        self.cost_series = cost_series
        self.metrics = metrics


class BacktestEngine:
    def __init__(self):
        pass

    def _get_pnl(
        self,
        y: pd.Series,
        x: pd.Series,
        signals: pd.Series,
        hedge_ratios: pd.Series,
        notional: float,
    ) -> pd.Series:
        x_returns = (x - x.shift(1)) / x.shift(1)
        y_returns = (y - y.shift(1)) / y.shift(1)

        pnl = pd.Series(0.0, index=y.index)
        for i in range(1, len(y_returns)):
            if signals.iloc[i - 1] == 1:
                pnl.iloc[i] = notional * (
                    y_returns.iloc[i]
                    - hedge_ratios.iloc[i - 1] * x_returns.iloc[i]
                )
            if signals.iloc[i - 1] == -1:
                pnl.iloc[i] = notional * (
                    -y_returns.iloc[i]
                    + hedge_ratios.iloc[i - 1] * x_returns.iloc[i]
                )
        return pnl

    def _get_costs(
        self,
        signals: pd.Series,
        hedge_ratios: pd.Series,
        notional: float,
        cost: float,
    ) -> pd.Series:
        trades = (signals != signals.shift(1)).fillna(False).astype(float)
        mult = cost / 10000
        return notional * trades * mult * (1 + hedge_ratios)

    def run(
        self,
        y: pd.Series,
        x: pd.Series,
        signals: pd.Series,
        hedge_ratios: pd.Series,
        notional=1.0,
        cost=COST.cost_bps,
    ) -> BacktestResult:
        gross_pnl = self._get_pnl(y, x, signals, hedge_ratios, notional)
        costs = self._get_costs(signals, hedge_ratios, notional, cost)
        pnl = gross_pnl - costs
        metrics = summarize(pnl, signals)
        result = BacktestResult(pnl, signals, gross_pnl, costs, metrics)
        return result
