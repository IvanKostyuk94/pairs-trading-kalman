# Pairs Trading with Kalman and Rolling OLS Dynamic Hedge Ratios

## Overview
The primary goal of this project is to investigate whether a dynamic hedge ratio improves pairs trading performance over a static OLS baseline. Additionally, we compare GARCH based volatility estimation for signal generation to a running average based one. We test four ETF pairs (IAU/GDX, GLD/SLV, XLF/KBE, SPY/IVV) on daily price data from 2017–2025. 

The main finding is that dynamic hedge ratios provide pair-specific benefits: rolling OLS outperforms on GDX/IAU while Kalman hedging generated larger PnL for SLV/GLD albeit at a slightly lower Sharpe-ratio. While a simple OLS dominates for KBE/XLF. I additionally find that GARCH-based signal generation underperforms a simple rolling z-score.

## Research Questions
- Does a Kalman-filtered dynamic hedge ratio outperform a static OLS hedge on a risk-adjusted basis?
- Does Kalman outperform more on pairs where $\beta$ is empirically unstable?
- Does GARCH conditional volatility improve signal quality over a 60-day rolling standard deviation?
- What is the optimal rolling OLS window for hedge ratio estimation?

## Methodology
1. Candidate pairs are screened for cointegration using the Engle-Granger and Johansen tests on a short (6 month) cointegration window.

2. A static OLS hedge is fitted on the training set (2017–2019) and used as a baseline. 

3. A Kalman filter with OLS-initialised priors tracks the hedge ratio dynamically out-of-sample.

4. Signal generation uses a rolling z-score of the OLS spread with entry/exit/stop-loss thresholds tuned via grid search on the validation set (2020–2022).

5. A GARCH(1,1) model is fitted on training residuals to produce conditional volatility estimates for alternative signal generation and position sizing and compared the rolling average estimator.

6. A rolling OLS hedge with window hyperparameter search is compared against both static OLS and Kalman.

7. All final results are evaluated on the test set (2023–2025) using 5 bps round-trip transaction costs.

## Data
| | |
|---|---|
| **Tickers** | IAU, GDX, GLD, SLV, XLF, KBE, SPY, IVV |
| **Source** | Yahoo Finance via `yfinance` |
| **Training** | 2017-01-01 – 2019-12-31 |
| **Validation** | 2020-01-01 – 2022-12-31 |
| **Test** | 2023-01-01 – 2025-12-31 |

## Results
| Pair    | Method      | Sharpe | Max Drawdown | Hit Rate | Avg Holding | N Trades |
|---------|-------------|-------:|-------------:|---------:|------------:|---------:|
| GDX/IAU | OLS         | -0.27  | 0.13         | 0.49     | 23.8        | 10       |
|         | Kalman      | -0.11  | 0.18         | 0.47     |             |          |
|         | Rolling OLS | -0.03  | 0.15         | 0.47     |             |          |
| SLV/GLD | OLS         |  0.23  | 1.70         | 0.50     | 16.6        | 18       |
|         | Kalman      |  0.21  | 2.26         | 0.50     |             |          |
|         | Rolling OLS |  0.13  | 1.86         | 0.50     |             |          |
| KBE/XLF | OLS         |  0.35  | 0.08         | 0.50     | 16.8        | 13       |
|         | Kalman      |  0.19  | 0.11         | 0.51     |             |          |
|         | Rolling OLS |  0.25  | 0.09         | 0.51     |             |          |
| IVV/SPY | OLS         | -2.77  | 0.03         | 0.44     | 5.3         | 21       |
|         | Kalman      | -2.76  | 0.03         | 0.44     |             |          |
|         | Rolling OLS | -2.75  | 0.03         | 0.44     |             |          |

Key findings:
- **SLV/GLD** and **KBE/XLF** are the only reliably tradable pairs across all hedge ratio methods
- **Kalman vs OLS vs Rolling OLS**: Kalman generates higher cumulative PnL on SLV/GLD but lower Sharpe due to amplified drawdowns during the 2025 silver squeeze; hypothesis that Kalman benefits most where $\beta$ is unstable is not supported.
- **Rolling OLS**: outperforms on GDX/IAU (Sharpe -0.03 vs OLS −0.27) which however is not tradable and underperforms on SLV/GLD and KBE/XLF.
- **GARCH**: correctly removes autocorrelation in squared residuals in-sample but underperforms rolling z-score out-of-sample. GARCH's long volatility memory delays normalisation after COVID, suppressing valid post-shock entry signals.

## Repository Structure
```
notebooks/
├── 01_pair_selection.ipynb       # Cointegration screening
├── 02_ols_baseline.ipynb         # Static OLS hedge and baseline backtest
├── 03_kalman_hedge.ipynb         # Kalman filter implementation and comparison
├── 04_trading_rules.ipynb        # Signal generation, GARCH, grid search
├── 05_rolling_ols_hedge.ipynb    # Rolling OLS window search and comparison
└── 06_results.ipynb              # Out-of-sample evaluation (2023–2025)

src/pairs_trading/
├── models/      # OLSHedge, KalmanHedge, RollingOLSHedge
├── signals/     # rolling_zscore, garch_zscore, generate_signal
├── stats/       # GARCHVolatility, cointegration tests
├── backtest/    # BacktestEngine, metrics
└── config.py    # All hyperparameters
```

## Setup
```bash
uv sync
jupyter lab
```