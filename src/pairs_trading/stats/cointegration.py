from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import pandas as pd
from pairs_trading.config import COINT
from collections.abc import Sequence


class EGResult:
    def __init__(
        self, time_series_df: pd.DataFrame, tickers: Sequence[str]
    ) -> None:
        if len(tickers) != 2:
            raise ValueError(
                f"EGResult expects exactly 2 tickers, got {len(tickers)}"
            )
        eg_results_1 = coint(
            time_series_df[tickers[0]], time_series_df[tickers[1]]
        )
        eg_results_2 = coint(
            time_series_df[tickers[1]], time_series_df[tickers[0]]
        )
        if eg_results_1[1] < eg_results_2[1]:
            eg_results = eg_results_1
            self.ticker1 = tickers[0]
            self.ticker2 = tickers[1]
        else:
            eg_results = eg_results_2
            self.ticker1 = tickers[1]
            self.ticker2 = tickers[0]

        self.statistic = eg_results[0]
        self.p_value = eg_results[1]
        self.is_cointegrated = self.p_value < COINT.p_value_threshold

    def print_results(self) -> None:
        print(
            f"Engle-Granger Result: y = {self.ticker1}, x={self.ticker2} (statistic={self.statistic}, p-value={self.p_value}, cointegrated: {self.is_cointegrated})"
        )
        return


class JohansenResult:
    def __init__(
        self, time_series_df: pd.DataFrame, tickers: Sequence[str]
    ) -> None:
        if len(tickers) != 2:
            raise ValueError(
                f"JohansenResult expects exactly 2 tickers, got {len(tickers)}"
            )
        j_results = coint_johansen(
            time_series_df[list(tickers)], det_order=0, k_ar_diff=1
        )

        self.statistic = j_results.lr1[0]
        self.critical_value = j_results.cvt[0, COINT.johansen_cv_idx]

        self.is_cointegrated = self.statistic > self.critical_value

    def print_results(self) -> None:
        print(
            f"Johansen Result: (statistic: {self.statistic}, critical values:{self.critical_value}, cointegrated: {self.is_cointegrated})"
        )
        return
