from pairs_trading.config import RollingOLSConfig, ROLLING_OLS
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import pandas as pd


class RollingOLSHedge:
    def __init__(self, config: RollingOLSConfig = ROLLING_OLS):
        self.window = config.window

    def fit(self, target_asset: pd.Series, feature_asset: pd.Series):
        self._train_y = target_asset
        self._train_x = feature_asset
        result = RollingOLS(
            target_asset, sm.add_constant(feature_asset), window=self.window
        ).fit()
        self.beta = result.params.iloc[:, 1].set_axis(target_asset.index)
        self.alpha = result.params.iloc[:, 0].set_axis(target_asset.index)
        self.residuals = (
            target_asset - self.beta.values * feature_asset - self.alpha.values
        )

    def transform(
        self, target_asset: pd.Series, feature_asset: pd.Series
    ) -> tuple:
        y = pd.concat([self._train_y, target_asset])
        x = pd.concat([self._train_x, feature_asset])

        result = RollingOLS(y, sm.add_constant(x), window=self.window).fit()
        n_train = len(self._train_y)
        beta = result.params.iloc[n_train:, 1].set_axis(target_asset.index)
        alpha = result.params.iloc[n_train:, 0].set_axis(target_asset.index)

        residuals = target_asset - beta.values * feature_asset - alpha.values
        return beta, alpha, residuals
