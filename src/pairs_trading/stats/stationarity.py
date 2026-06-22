import pandas as pd

from statsmodels.tsa.stattools import adfuller, kpss

from pairs_trading.config import COINT


class AdfResult:
    def __init__(self, time_series: pd.Series) -> None:
        adf_results = adfuller(time_series)
        self.statistic = adf_results[0]
        self.p_value = adf_results[1]
        self.is_stationary = self.p_value < COINT.p_value_threshold

    def print_results(self) -> None:
        print(
            f"AdfResult(statistic={self.statistic}, p-value={self.p_value}, stationary: {self.is_stationary})"
        )
        return


class KpssResult:
    def __init__(self, time_series: pd.Series) -> None:
        kpss_results = kpss(time_series)
        self.statistic = kpss_results[0]
        self.p_value = kpss_results[1]
        self.is_stationary = self.p_value > COINT.p_value_threshold

    def print_results(self) -> None:
        print(
            f"KpssResult(statistic={self.statistic}, p-value={self.p_value}, stationary: {self.is_stationary})"
        )
        return
