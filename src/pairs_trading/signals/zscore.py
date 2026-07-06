import pandas as pd
import numpy as np
from pairs_trading.config import SIGNAL, SignalConfig


def rolling_zscore(spread: pd.Series, window=60) -> pd.Series:
    mu_t = spread.rolling(window).mean()
    std_t = spread.rolling(window).std()
    zscore = (spread - mu_t) / std_t
    return zscore


def garch_zscore(
    spread: pd.Series, cond_volatility: pd.Series, window=60
) -> pd.Series:
    mu_t = spread.rolling(window).mean()
    zscore = (spread - mu_t) / cond_volatility
    return zscore


def generate_signal(
    zscore: pd.Series, config: SignalConfig = SIGNAL
) -> pd.Series:
    signal = np.zeros(len(zscore))
    state = 0
    for i in range(len(signal)):
        z = zscore.iloc[i]
        if z > config.entry_z:
            if z < config.stop_z:
                state = -1
            else:
                state = 0
        if z < -config.entry_z:
            if z > -config.stop_z:
                state = 1
            else:
                state = 0
        if np.abs(z) < config.exit_z:
            state = 0
        signal[i] = state
    return pd.Series(signal, index=zscore.index)
