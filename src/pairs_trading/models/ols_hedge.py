import pandas as pd
import numpy as np

import statsmodels.api as sm


class OLSHedge:
    def __init__(self):
        pass

    def fit(self, target_asset: pd.Series, feature_asset: pd.Series) -> None:
        X = sm.add_constant(feature_asset)
        self.model = sm.OLS(target_asset, X).fit()
        self.alpha = self.model.params.iloc[0]
        self.beta = self.model.params.iloc[1]
        self.residuals = self.model.resid  # In sample residuals
        self.residual_std = np.std(self.residuals)
        self.rsquared = self.model.rsquared
        return

    def spread(
        self, target_asset: pd.Series, feature_asset: pd.Series
    ) -> pd.Series:
        if not hasattr(self, "model"):
            raise RuntimeError("Call fit() before spread()")
        X = sm.add_constant(feature_asset)
        resid = target_asset - self.model.predict(X)
        return resid
