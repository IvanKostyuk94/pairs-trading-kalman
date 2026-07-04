from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DataConfig:
    tickers: tuple[str, ...] = ("GLD", "SLV")
    start: str = "2017-01-01"
    end: str = "2025-12-31"


@dataclass(frozen=True)
class SplitConfig:
    coint_start: str = "2019-09-01"
    coint_end: str = "2019-12-31"
    train_start: str = "2017-01-01"
    train_end: str = "2019-12-31"
    val_start: str = "2020-01-01"
    val_end: str = "2022-12-31"
    test_start: str = "2022-01-01"
    test_end: str = "2025-12-31"


@dataclass(frozen=True)
class CointConfig:
    p_value_threshold: float = 0.05
    johansen_cv_idx: int = 1  # 0=90%, 1=95%, 2=99%


@dataclass(frozen=True)
class SignalConfig:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5
    max_holding: int = 30


@dataclass(frozen=True)
class CostConfig:
    cost_bps: float = 5.0


@dataclass(frozen=True)
class KalmanConfig:
    delta: float = 7e-7


@dataclass(frozen=True)
class GARCHConfig:
    p: int = 1
    q: int = 1
    target_vol: float = 0.01


DATA = DataConfig()
SPLIT = SplitConfig()
COINT = CointConfig()
SIGNAL = SignalConfig()
COST = CostConfig()
KALMAN = KalmanConfig()
GARCH = GARCHConfig()
