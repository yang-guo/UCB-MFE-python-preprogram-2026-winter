# Pairs_Tilt: long-only relative-value sleeves

`Pairs_Tilt` is the third research model. It uses the same point-in-time
financial characteristics as `FamaFrench5`, groups similar stocks, tests pairs
for cointegration, and tilts a long-only portfolio toward the cheap leg of a
cointegrated pair. It is deliberately **not** a market-neutral stat-arb book:
the expensive leg is not shorted or purchased.

## Data clock: 10x cap-weighted volume bars

The cap-weighted eligible universe's volume defines a common sampling clock:

1. Build the daily cap-weighted portfolio price and volume from `MarketData`
   and the point-in-time `FamaFrench5` market equity weights.
2. Take the trailing 90-session median of the portfolio's daily volume through the **prior** session.
3. Close a bar when cumulative portfolio volume reaches **10x** that median; the start session's own volume is not used to set the target.
4. Sample every stock's adjusted close at those same bar ends.

All pair observations therefore share one clock. The construction uses only
trailing volume and closes each bar at the session where the target is reached;
it does not use future bars or future prices. At a signal bar end, every pair
window is explicitly truncated to `log_prices.loc[:bar_end]`.

## Clusters

At each cluster refresh (every 13 bars by default; roughly semiannual at
the 10x bar size), the latest point-in-time eligible cross-section is
clustered on:

- `log ME` (size),
- `BM` (value),
- `OP` (profitability),
- `INV` (investment),
- no sector dummies by default: `universe.sector` is a current vendor snapshot, not a point-in-time field. Pair eligibility also ignores the static `sector_ok` flag.

Ratio features are signed-log compressed, bounded by median +/- 5 IQR, and
robust-standardized. Ward hierarchical clustering
is run over a range of cluster counts; the count with the best silhouette score
is used, subject to a minimum cluster size of five names. Ward splinters are
merged into the nearest large cluster; if no partition can produce at least
two surviving clusters, all names are placed in one cluster. Clusters are
refreshed only with characteristics whose `date` is on or before the refresh
bar; earlier clusters are reused between refreshes. At the 10x bar size the
13-bar refresh is roughly semiannual, not quarterly.

## Rebalance cadence

Cointegration is retested at each **rebalance period**, not at every completed
volume bar. The default is `rebalance_days=10`, matching the ten-calendar-day
sleeves. For each decision date, the latest completed bar at or before that
date is used; if no new bar exists, the previous pair screen is reused. The
selected pairs and hedge ratios are frozen until the next rebalance. Set
`rebalance_days=None` to restore the legacy every-bar screen, or pass an
explicit `signal_dates` list to `PairsResearch.build`.

Within each cluster:

1. Compute bar returns over the trailing 104 bars (about two years at the 10x bar size).
2. Pre-filter candidate pairs by absolute return correlation (20 per cluster by
   default).
3. Run an Engle-Granger regression `log P_A = alpha + beta log P_B` and the
   `statsmodels.coint` test (constant, AIC lag selection up to four lags) on
   the residual.
4. Keep pairs with FDR-adjusted ADF p-value below 0.05 and a half-life between
   2 and 26 bars.
5. Run a Kalman filter with a random-walk `beta`, an OLS prior, and a static
   `alpha` over the same window; use the final **one-step prediction error**
   divided by its prediction standard deviation as the z-score.
6. Require `|z| >= 2` at the bar end.
7. The long-only candidate is the cheap leg: `P_A` if `z < -2`, `P_B` if
   `z > +2`. The expensive leg is not purchased.

The strongest pairs per cluster are retained by adjusted p-value and absolute
z-score. The execution strategy applies the one-pair-per-stock greedy rule
across clusters.

## Walk-forward checks

- `PairsResearch.build` loops over rebalance decision dates in order and
  maps each to the latest completed bar end at or before it.
- A cluster refresh reads only characteristics with `date <= bar_end`.
- Pair correlation, OLS, ADF, half-life, and Kalman windows use only bars up to
  `bar_end`; `log_prices` is sliced before the window is taken.
- The volume-bar target uses the trailing median through the prior session, so
  the bar-start session's own volume is not used.
- `PairsTiltStrategy` uses the latest pair signal with `date <= signal` and the
  latest factor cross-section with `date <= signal`; future signals are
  explicitly filtered before selection.
- The market fallback uses point-in-time eligible cap weights, not a future
  universe.
- Delisted names are not forward-filled across bar ends; missing bar
  observations simply drop that pair observation.

## Sleeves and allocation

`sleeve_days` is the length of one sleeve and `ramp_sleeves` is how many
sleeves make up the ramp, so `sleeve_days * ramp_sleeves` is the holding
period. The development notebook uses `sleeve_days=30, ramp_sleeves=3`: three
overlapping 30-day sleeves with a 90-calendar-day holding period, matched to
`rebalance_days=30`. The class defaults remain `10` and `9`.

- Every scheduled decision adds one sleeve; the oldest sleeve matures 90
  calendar days later.
- The first `ramp_sleeves` scheduled decisions invest `NAV / ramp_sleeves` in
  the cap-weighted eligible universe.
- After the ramp, a maturing sleeve is reinvested at the next session.
- At most three pairs per sleeve; each pair receives `sleeve cash / 3`.
- One pair per stock per sleeve: both legs are marked used, so a stock cannot
  be the long leg in one pair and the expensive leg in another in the same
  sleeve.
- Any unused sleeve capital buys the cap-weighted eligible universe.
- If no pair is eligible, the whole sleeve buys the market.
- No early sales: each purchase exits at its 90-day maturity.

## Usage

The sleeve is researched and demonstrated in `Research.ipynb` section 10; it is
not part of the `Backtesting.ipynb` portfolio, which runs the FF5 factor tilt
and the HMM volume strategy.

```python
from backtesting import Backtester, FamaFrench5, MarketData
from research.pairs import PairsConfig, PairsResearch
from strategies import PairsTiltStrategy

data = MarketData.load("data/equities_top2000.duckdb", end="2026-09-30")
factors = FamaFrench5(lag_months=6).build(
    data, start="2005-01-01", end="2026-09-30")
pairs = PairsResearch(PairsConfig(rebalance_days=30, max_clusters=7)).build(
    data, factors, start="2005-01-01", end="2026-09-30")

strategy = PairsTiltStrategy(pairs, factors, max_pairs=3,
                             sleeve_days=30, ramp_sleeves=3)
engine = Backtester(market_data=data, strategy=strategy,
                    initial_cash=1_000_000, fee_bps=5,
                    delisting="error")
result = engine.run("2016-01-04", "2026-09-30")
```

## Limitations

- Long-only only; it cannot short the expensive leg or run a market-neutral
  spread. The hedge ratio is used to identify the cheap leg and to size the
  relative-value signal, not to build a short book.
- No convergence-based early exit. Every lot is held for 90 calendar days,
  even if the spread converges earlier.
- The 10x portfolio volume bars are much longer than daily bars, so the pair
  sample is smaller and slower to refresh.
- Clustering is unstable at cluster boundaries; pair selection is refreshed
  at each rebalance and clusters refresh only periodically. A pair can remain
  eligible for up to one rebalance interval.
- Multiple testing is controlled with FDR across the candidate pairs, but the
  research still involves selection and should be validated out of sample.
- `coint` size is still somewhat above the nominal 5% at short windows, and
the 104-bar window has limited power for half-lives near the top of the
allowed range. Treat the pair screen as a filter, not a guarantee.
- `adj_close` is split/dividend adjusted with later corporate-action data. This
  is standard total-return data, but it is not an original point-in-time price
  vintage; the raw `close` would contain split jumps and is not used for the
  pair levels.
- Delisting exits use the last observed mark; the true delisting return is not
  in the database.
- The model is research code; it is not automatically part of the final HW4
  submission strategy.
