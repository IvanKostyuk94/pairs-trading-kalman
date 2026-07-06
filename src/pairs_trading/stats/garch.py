import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pairs_trading.config import GARCHConfig, GARCH


class GARCHVolatility:
    def __init__(self, config: GARCHConfig = GARCH):
        self.p = config.p
        self.q = config.q

    def _fit_garch_constrained(self, residuals, beta_min=0.4):
        eps = residuals.values
        n = len(eps)
        var0 = np.var(eps)

        def neg_loglik(params):
            alpha, beta = params
            omega = var0 * (1 - alpha - beta)  # stationarity constraint
            sigma2 = np.empty(n)
            sigma2[0] = var0
            for t in range(1, n):
                sigma2[t] = (
                    omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
                )
            if np.any(sigma2 <= 0):
                return 1e10
            return 0.5 * np.sum(np.log(sigma2) + eps**2 / sigma2)

        bounds = [(0.01, 0.4), (beta_min, 0.98)]
        res = minimize(
            neg_loglik, [0.05, 0.90], method="L-BFGS-B", bounds=bounds
        )
        alpha, beta = res.x
        omega = var0 * (1 - alpha - beta)
        return omega, alpha, beta

    def fit(self, residuals: pd.Series, window: int = 60) -> None:
        self.residuals = residuals
        mu = residuals.rolling(window, min_periods=1).mean()
        self._demeaned = residuals - mu
        eps = self._demeaned.values
        n = len(eps)
        var0 = np.var(eps)

        self.omega, self.alpha, self.beta = self._fit_garch_constrained(
            self._demeaned
        )

        sigma2 = np.empty(n)
        sigma2[0] = var0
        for t in range(1, n):
            sigma2[t] = (
                self.omega
                + self.alpha * eps[t - 1] ** 2
                + self.beta * sigma2[t - 1]
            )
        self.sigma_t = pd.Series(np.sqrt(sigma2), index=residuals.index)
        return

    def transform(
        self, residuals_test: pd.Series, window: int = 60
    ) -> pd.Series:
        mu_test = residuals_test.rolling(window, min_periods=1).mean()
        demeaned = (residuals_test - mu_test).values
        last_resid2 = float(self._demeaned.iloc[-1] ** 2)
        last_sigma2 = float(self.sigma_t.iloc[-1] ** 2)

        # initialize the inference using the last elements of the train set
        sigma2 = np.zeros(len(demeaned))
        sigma2[0] = (
            self.omega + self.alpha * last_resid2 + self.beta * last_sigma2
        )

        for i in range(1, len(demeaned)):
            sigma2[i] = (
                self.omega
                + self.alpha * demeaned[i - 1] ** 2
                + self.beta * sigma2[i - 1]
            )

        return pd.Series(np.sqrt(sigma2), index=residuals_test.index)


def compute_position_sizes(conditional_vol: pd.Series):
    return conditional_vol.mean() / conditional_vol
