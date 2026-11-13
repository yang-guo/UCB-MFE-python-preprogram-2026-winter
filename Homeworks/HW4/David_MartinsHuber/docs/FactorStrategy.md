# FF5 regression strategy: model, sleeves, and execution

`strategies/factor.py` contains the tactical strategy and the market baseline.
Both use the point-in-time eligible cross-section from `FamaFrench5` and respect
the homework's 90-calendar-day holding rule.

## Model B: rolling FF5 regression

At each month-end signal `t`, for every stock with enough matured observations:

```text
r_i,t = alpha_i + b_i MKT_t + s_i SMB_t + h_i HML_t + r_i RMW_t + c_i CMA_t + e_i,t
```

- `r_i` is the **90-calendar-day** return: entry at the next session's close
  after the signal, exit at the first quoted close on or after 90 days.
- Factor returns are three compounded monthly FF5 returns aligned to the same
  formation date.
- Only observations whose exit is **strictly before** `t` enter the fit. A
  target that is still open cannot contribute.
- The rolling window is 60 monthly observations by default; at least 24 are
  required.
- The prediction is `alpha + beta @ expected_factor_return`, where the expected
  factor return is the trailing 12-month mean of the 90-day factor returns that
  are fully known at the signal. Because a 90-day factor window ending at `t`
  needs factor returns through `t + 2`, the mean stops three month-ends before
  the signal. This is a ranking signal, not a forecast.
- No risk-free rate is used in the model.

The fit is point-in-time: `DecisionContext.prices` stops at the signal close,
and the factor result is sliced to the same date. Future prices, future
filings, and open targets cannot change the scores.

## Signal and target weights

The eligible universe is the `FamaFrench5` cross-section: point-in-time
membership, positive book equity, finite BM/OP/INV, no Financial Services, and
no minimum market-cap filter.

At the signal date:

1. Rank the names with a score using count-based quintiles (exact 20% of the
   eligible cross-section; ties are broken stably).
2. Bottom 20%: target weight **0**.
3. Middle 60%: target weight equal to its **cap weight**.
4. Top 20%: cap weight **plus the entire bottom-quintile pool**, distributed in
   proportion to the top names' cap weights.
5. Renormalize to 1. If the cross-section is too small or has no score
   dispersion, fall back to cap weights rather than sitting in cash.

The strategy never sells a name because its score fell. Existing lots remain
until the engine's 90-day maturity exit.

## Three-sleeve buffer ramp

The strategy is started in the Q1 2016 buffer, not on the first test day:

| Date | Action |
|---|---|
| Jan 2016 | Invest 1/3 of NAV in the cap-weighted market |
| Feb 2016 | Invest another 1/3 in the market |
| Mar 2016 | Invest the remaining 1/3 in the market |
| Apr 2016 | First sleeve matures; reinvest at the next session into the top quintile |
| May 2016 | Second sleeve matures and tilts |
| Jun 2016 | Third sleeve matures and tilts |
| Jul 2016 onward | One sleeve matures each month and is reinvested into the current top quintile |

Each sleeve is a 90-day rolling bucket. The engine sells it at maturity; the
strategy reinvests the cash at the **next session**, using the most recent
fitted signal. This is why `propose_buys` can act mid-month: it is not a new
signal, only the delayed reinvestment of a matured sleeve.

During the ramp the strategy holds market weights. After the ramp the whole
portfolio is tactical, with roughly one third refreshed each month. The three
sleeves are not re-equalized; each compounds its own proceeds. Orders below
`min_order_notional` (default $1) are skipped so residual cash does not create
a full-basket dribble order.

## Baseline

`CapWeightedMarketStrategy` targets market-cap weights, fills only positive
gaps versus current holdings, and rolls every maturing cohort back into the
market. It also reinvests maturity proceeds at the next session. It uses the
same eligible universe as the tilt, so the comparison isolates the signal
rather than a universe change.

## Engine support

Two engine features make this possible:

- `Strategy.self_fitting = True`: the engine still owns the refit phase,
  model-version counter, and audit record, but ignores any supplied
  `training_panel` and does not require one. `FactorTiltStrategy.refit(data,
  context)` receives `data=None` and fits from `context.prices` plus the factor
  result.
- `Backtester(..., delisting="error")`: bounded stale marks are permitted for
  valuation only. Exceeding `max_stale_sessions` stops the run for investigation;
  no sale is fabricated, and no missing-price rule bypasses 90-day maturity.

## Usage

```python
from backtesting import Backtester, FamaFrench5, MarketData
from strategies import CapWeightedMarketStrategy, FactorTiltStrategy

data = MarketData.load("data/equities_top2000.duckdb", end="2026-09-30")
# Build enough factor history for the 24-60 month regression lookback before
# the first 2016 signal; a 2015 start would leave the first year without scores.
factors = FamaFrench5(lag_months=6).build(
    data, start="2005-01-01", end="2026-09-30")

strategy = FactorTiltStrategy(factors, window=60, min_obs=24,
                              top_q=0.2, bottom_q=0.2, ramp_months=3)
engine = Backtester(market_data=data, strategy=strategy,
                    initial_cash=1_000_000, fee_bps=5,
                    delisting="error")
result = engine.run("2016-01-04", "2026-09-30")  # buffer ramp included
```

Use `CapWeightedMarketStrategy(factors)` for the baseline. Report the test
period from Apr 1, 2016 while allowing the buffer trades to establish the
sleeves.

## Limitations

- The regression uses monthly 90-day overlapping observations; residuals are
  autocorrelated and the coefficient standard errors are not corrected.
- Historical observations are filtered to names eligible at the signal date,
  not each observation's own formation date. This is a selection choice, not
  look-ahead.
- A label whose exit session has no quote for that symbol is dropped rather
  than carried to the engine's later quoted exit; this is conservative sample
  attrition.
- The expected factor return is a trailing mean, not a forecast.
- The mid-month reinvestment uses the last month-end characteristics and
  market caps; the target is slightly stale within the month.
- The strategy never sells early, so a name that falls into the bottom
  quintile remains held until maturity.
- The model does not use a risk-free rate, transaction-cost-aware optimization,
  or covariance risk targeting.
- Delisting exits use the last observed mark; the true delisting return is not
  in the database.
