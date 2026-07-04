import pandas as pd
import numpy as np
from pairs_trading.config import KalmanConfig, KALMAN
from pairs_trading.models.ols_hedge import OLSHedge


class KalmanHedge:
    def __init__(self, config: KalmanConfig = KALMAN):
        self._Q = config.delta * np.eye(2)

    def _get_Ht(self, x_t: float) -> np.ndarray:
        return np.array([x_t, 1.0])

    def _predict_Pt1(self) -> None:
        self._P = self._P + self._Q
        return

    def _innovation(self, y_t: float, H_t: np.ndarray) -> float:
        v_t = y_t - H_t @ self._theta
        return float(v_t)

    def _innovation_variance(self, H_t: np.ndarray) -> np.ndarray:
        S_t = H_t @ self._P @ H_t.T + self._R
        return S_t

    def _Kalman_gain(self, H_t: np.ndarray, S_t: np.ndarray) -> np.ndarray:
        K_t = self._P @ H_t.T / S_t
        return K_t

    def _update_theta(self, K_t: np.ndarray, v_t: float) -> None:
        self._theta = self._theta + K_t * v_t
        return

    def _update_Pt(self, K_t: np.ndarray, H_t: np.ndarray):
        self._P = (np.eye(2) - np.outer(K_t, H_t)) @ self._P

    def _update(self, yt: float, xt: float) -> tuple[float, float, float]:
        Ht = self._get_Ht(xt)
        self._predict_Pt1()
        vt = self._innovation(yt, Ht)
        St = self._innovation_variance(Ht)
        Kt = self._Kalman_gain(Ht, St)
        self._update_theta(Kt, vt)
        self._update_Pt(Kt, Ht)
        return (float(self._theta[0]), float(self._theta[1]), vt)

    def _run_filter(
        self, target_asset: pd.Series, feature_asset: pd.Series
    ) -> None:
        self.alpha = pd.Series(0.0, index=target_asset.index)
        self.beta = pd.Series(0.0, index=target_asset.index)
        self.residuals = pd.Series(0.0, index=target_asset.index)

        for i, (yt, xt) in enumerate(
            zip(target_asset.values, feature_asset.values)
        ):
            beta, alpha, resid = self._update(yt, xt)
            self.beta.iloc[i] = beta
            self.alpha.iloc[i] = alpha
            self.residuals.iloc[i] = resid
        return

    def _get_ols_init(
        self, target_asset: pd.Series, feature_asset: pd.Series
    ) -> None:
        ols = OLSHedge()
        ols.fit(target_asset, feature_asset)
        self._theta = np.array([ols.beta, ols.alpha])
        self._P = np.diag(
            [ols.model.bse.iloc[1] ** 2, ols.model.bse.iloc[0] ** 2]
        )
        self._R = ols.residual_std**2
        return

    def fit(
        self,
        target_asset: pd.Series,
        feature_asset: pd.Series,
    ) -> None:
        self._get_ols_init(target_asset, feature_asset)
        self._run_filter(target_asset, feature_asset)
        return

    def transform(
        self,
        target_asset: pd.Series,
        feature_asset: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        self._run_filter(target_asset, feature_asset)
        return (self.beta, self.alpha, self.residuals)

    def spread(self):
        if not hasattr(self, "residuals"):
            raise RuntimeError("Call fit() before spread()")
        return self.residuals
