import pandas as pd
from pairs_trading.config import COST, SIGNAL
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
        position_sizes: pd.Series | None = None,
    ) -> pd.Series:
        x_returns = (x - x.shift(1)) / x.shift(1)
        y_returns = (y - y.shift(1)) / y.shift(1)

        pnl = pd.Series(0.0, index=y.index)
        for i in range(1, len(y_returns)):
            if position_sizes is not None:
                invest = position_sizes.iloc[i]
            else:
                invest = 1.0
            if signals.iloc[i - 1] == 1:
                pnl.iloc[i] = invest * (
                    y_returns.iloc[i]
                    - hedge_ratios.iloc[i - 1] * x_returns.iloc[i]
                )
            if signals.iloc[i - 1] == -1:
                pnl.iloc[i] = invest * (
                    -y_returns.iloc[i]
                    + hedge_ratios.iloc[i - 1] * x_returns.iloc[i]
                )
        return pnl

    def _get_costs(
        self,
        signals: pd.Series,
        hedge_ratios: pd.Series,
        cost: float,
        position_sizes: pd.Series | None = None,
    ) -> pd.Series:
        trades = (signals != signals.shift(1)).fillna(False).astype(float)
        mult = cost / 10000
        sizes = (
            position_sizes
            if position_sizes is not None
            else pd.Series(1.0, index=signals.index)
        )
        return sizes * trades * mult * (1 + hedge_ratios)

    def _apply_max_holding(self, signals: pd.Series, max_holding: int):
        effective_signals = pd.Series(0, index=signals.index)
        counter = 0
        state = 0
        for i, signal in enumerate(signals.values):
            if signal == 1:
                if counter >= max_holding and state == 1:
                    effective_signals.iloc[i] = 0
                else:
                    if state != 1:
                        state = 1
                        counter = 0
                    counter += 1
                    effective_signals.iloc[i] = 1
            elif signal == -1:
                if counter >= max_holding and state == -1:
                    effective_signals.iloc[i] = 0
                else:
                    if state != -1:
                        state = -1
                        counter = 0
                    counter += 1
                    effective_signals.iloc[i] = -1
            else:
                effective_signals.iloc[i] = 0
                counter = 0
        return effective_signals

    # Function to lock the ratio and size at entry into a position
    def _lock_at_entry(
        self,
        signals: pd.Series,
        hedge_ratios: pd.Series,
        position_sizes: pd.Series | None = None,
    ) -> tuple[pd.Series, pd.Series]:
        locked_ratios = pd.Series(0.0, index=signals.index)
        locked_sizes = pd.Series(1.0, index=signals.index)
        current_ratio = 0.0
        current_size = 1.0
        for i in range(len(signals)):
            if signals.iloc[i] != 0 and (i == 0 or signals.iloc[i - 1] == 0):
                current_ratio = hedge_ratios.iloc[i]
                if position_sizes is not None:
                    current_size = position_sizes.iloc[i]
                else:
                    current_size = 1.0
            if signals.iloc[i] != 0:
                locked_ratios.iloc[i] = current_ratio
                locked_sizes.iloc[i] = current_size
            elif i > 0 and signals.iloc[i - 1] != 0:
                locked_ratios.iloc[i] = current_ratio
                locked_sizes.iloc[i] = current_size
        return locked_ratios, locked_sizes

    def run(
        self,
        y: pd.Series,
        x: pd.Series,
        signals: pd.Series,
        hedge_ratios: pd.Series,
        cost: float = COST.cost_bps,
        max_holding: int = SIGNAL.max_holding,
        position_sizes: pd.Series | None = None,
    ) -> BacktestResult:
        effective_signals = self._apply_max_holding(signals, max_holding)
        locked_ratios, locked_sizes = self._lock_at_entry(
            effective_signals, hedge_ratios, position_sizes
        )
        gross_pnl = self._get_pnl(
            y, x, effective_signals, locked_ratios, locked_sizes
        )
        costs = self._get_costs(
            effective_signals, locked_ratios, cost, locked_sizes
        )
        pnl = gross_pnl - costs
        metrics = summarize(pnl, effective_signals)
        result = BacktestResult(
            pnl, effective_signals, gross_pnl, costs, metrics
        )
        return result
