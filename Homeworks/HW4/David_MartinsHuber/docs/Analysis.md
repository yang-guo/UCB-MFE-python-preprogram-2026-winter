# Portfolio analysis conventions

`backtesting/analytics.py` provides `BacktestAnalytics(result)`. Analysis reads daily portfolio NAV, including fees, cash, and marked open lots; it does not use the average return of individual trades. Both portfolios use the same dates and the initial pre-trade capital row, so first-entry fees are included. The benchmark is invested at the first test close, as required by HW4. CAGR uses elapsed calendar time between the first and last NAV rows.

## Required homework outputs

| Output | Definition |
|---|---|
| Final value | Last NAV, including cash and open positions. |
| CAGR | `(final_NAV / initial_NAV) ** (1 / elapsed_years) - 1`, with 365.25 calendar days per year. |
| Maximum drawdown | Minimum of `NAV / running_peak_NAV - 1`; reported as a negative fraction. |
| Sharpe | Daily mean excess return divided by sample daily return standard deviation, multiplied by `sqrt(252)`. Risk-free rate defaults to 2% annually, converted to `(1.02 ** (1 / 252)) - 1` daily. |
| Cumulative-return comparison | Both normalized NAV curves on the same chart. |
| Benchmark outperformance | Final-value difference in dollars and total-return difference in percentage points, plus whether the strategy beat the benchmark. |

The benchmark is `^SP500TR` from the supplied data. Do not add dividends separately to adjusted-price returns.

## Additional metrics

| Metric | Definition and units |
|---|---|
| Total return | `final_NAV / initial_NAV - 1`. |
| Annualized volatility | Sample standard deviation of daily simple returns times `sqrt(252)`. |
| Beta | Slope of the regression of strategy daily excess returns on benchmark daily excess returns: covariance divided by benchmark variance. Unitless. |
| Annualized alpha | Intercept of that regression, multiplied by 252. Arithmetic annualization of Jensen's alpha; not CAGR or raw benchmark outperformance. |
| VaR at 95% | Negative of the 5th percentile of observed daily simple returns. |
| VaR at 99% | Negative of the 1st percentile of observed daily simple returns. |
| Daily observations | Number of returns used; excludes the initial capital row. |

Alpha and beta use aligned observations and a constant daily risk-free rate. The benchmark's regression against itself has beta 1 and alpha 0 when its variance is nonzero. With insufficient observations or effectively zero benchmark variance, alpha and beta are `NaN`. Sharpe is `NaN` for effectively zero portfolio volatility.

VaR uses historical empirical quantiles with linear interpolation, a **one-market-session horizon**, and decimal return units. For example, `0.03` means a 3% loss threshold. A negative VaR is retained when the estimated tail quantile is a gain. It is not an annualized measure or a guaranteed loss ceiling. Small samples, especially at 99%, give weak tail estimates. The observation count is displayed to make sample size visible. No normality assumption, dollar conversion, or square-root-of-time scaling is applied.

All return and risk metrics are decimal fractions unless explicitly labeled percentage points. NAV gaps, nonpositive values, duplicated dates, and nonfinite values raise errors rather than being filled or silently dropped. Daily simple returns use the provided market sessions; the module cannot detect absent sessions without a separate reference calendar.

Finite input NAVs are not sufficient: derived daily returns, cumulative returns, benchmark comparisons, and required performance metrics must also remain finite. Unsupported overflow raises `ValueError`. `NaN` remains intentional only for undefined ratios or sample statistics such as Sharpe/beta/alpha with insufficient variation or observations.

## Usage

```python
from backtesting import BacktestAnalytics

analysis = BacktestAnalytics(result, annual_rf=0.02)
display(analysis.summary())
display(analysis.comparison())
analysis.plot_cumulative_returns()
```

`result.performance()` delegates to the same summary implementation, retaining its existing interface. Analysis describes realized portfolio behavior; it does not validate feature availability, model quality, or the backtest's economic assumptions. The homework still requires the discussion of findings, risks, limitations, and evidence of accounting/timing checks.
