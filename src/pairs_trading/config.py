from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    tickers: tuple[str, ...] = ("GLD", "SLV")
    start: str = "2020-01-01"
    end: str = "2024-12-31"


@dataclass(frozen=True)
class SplitConfig:
    train_start: str = "2020-01-01"
    train_end: str = "2023-01-01"
    test_start: str = "2023-01-02"
    test_end: str = "2024-12-31"


@dataclass(frozen=True)
class CointConfig:
    p_value_threshold: float = 0.05
    johansen_cv_idx: int = 1  # 0=90%, 1=95%, 2=99%


@dataclass(frozen=True)
class SignalConfig:
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5


@dataclass(frozen=True)
class CostConfig:
    cost_bps: float = 5.0


@dataclass(frozen=True)
class KalmanConfig:
    delta: float = 1e-4
    obs_noise: float = 1e-3


DATA = DataConfig()
SPLIT = SplitConfig()
COINT = CointConfig()
SIGNAL = SignalConfig()
COST = CostConfig()
KALMAN = KalmanConfig()
