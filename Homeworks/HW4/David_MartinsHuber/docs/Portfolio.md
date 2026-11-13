# Shadow portfolios and portfolio allocation

`ShadowPortfolioBacktester` coordinates independent shadow accounts and a main portfolio account. `PortfolioStrategy` implements the existing `Strategy` interface and composes a `PortfolioOptimizer`. Individual strategies keep their own logic and model state. The loader is unchanged from the lecture.

## Shared clock and information boundary

For each execution session, the coordinator follows this sequence:

1. Each shadow receives a separate context through the preceding signal close, applies any scheduled refit, and submits its purchase proposals. Each has its own strategy state, OMS, cash, and open lots.
2. A barrier verifies matching execution dates and signal cutoffs. Completed shadow NAV histories through that cutoff are copied into the portfolio adapter.
3. The optimizer receives only those completed daily shadow returns and the main portfolio snapshot. It cannot access shadow engines or full training/quote tables through its context. It returns long-only strategy weights summing to one.
4. The adapter combines the current component purchase proposals, and the main account reserves its purchase budgets. All decisions finish before any account reads execution-day prices.
5. Shadow and main accounts execute at the shared next close using the normal quote and 90-calendar-day lot rules. Their completed history becomes eligible for the next decision.

`Backtester.begin`, `prepare`, `advance`, and `finish` expose the incremental lifecycle; `Backtester.run` uses these same methods for ordinary standalone runs. The coordinator never repeatedly runs a historical backtest inside another backtest.

Coordinated engines share one validated, pre-pivoted `MarketPanel`, so each shadow does not re-pivot or re-copy the full price history. `ShadowPortfolioBacktester` defaults to `parallel=True` and prepares/advances independent shadows with a thread pool (`max_workers` defaults to the smaller of the shadow count and CPU count). Pass `parallel=False` or an explicit `max_workers` for single-threaded debugging. Contexts use copy-on-write prefix views and the OMS maintains incremental reservations, aggregate holdings, and a maturity queue.

The earliest history row is initial capital before the first shadow execution. Optional `shadow_start` can precede the main period, giving the optimizer a genuine walk-forward warm-up record. Those warm-up trades are shadow-only: the main portfolio still starts with its configured cash and no inherited positions.

## Refitting and provenance

Use a `ShadowSpec` for a component that needs its own training panel and feature names. All shadows use the same cutoff but can choose different historical training windows within their refit methods. The engine supplies only matured targets with `label_end < signal_date`; publication-time restrictions and separated features/targets apply as in the standalone engine. A successful refit activates a copied model, while failure stops the coordinated run.

Shadow history is append-only. Republishing or changing a previously published NAV prefix is rejected. Returned data is detached from account records, and the optimizer receives a copy. Orders, fills, and lots contain `strategy_id` and the entry `model_version`; fills also record signal and execution dates. Each shadow result includes its fit audit. A later refit never recalculates earlier trades or returns.

Different input windows are safe when they respect the common cutoff. Historical rows must still have point-in-time feature values. The system cannot detect a pre-trained model that already contains future knowledge or prevent arbitrary Python from loading future files. Start components in a fresh state and use the supplied refit API; fix strategy selection, fitting procedures, seeds, and allocation settings during development. Later source restatements remain a data limitation.

## What allocation means in this version

For component `i`, symbol `s`, and main NAV `V`, the candidate purchase is:

`V × strategy_weight[i] × shadow_purchase_budget[i, s] / shadow_NAV[i]`.

This preserves the shadow's proposed deployment fraction. If the combined amount exceeds main available cash, scale all purchases proportionally. Combine overlapping symbol budgets into a single main order. There is no forced investment of an idle component's allowance and no forced early sale to reach new weights.

Weights therefore control **new purchase flows**. They are not guaranteed current exposure weights, because existing 90-day lots remain invested. Each component's proposal is generated against its standalone shadow account, not a live component subaccount. Standalone shadow performance is not that component's realized contribution to the main portfolio.

Budget fractions are calculated before scaling by main NAV to avoid unnecessary intermediate overflow. Proposed amounts and totals must remain finite. Finite positive shadow NAVs must also produce finite daily returns before history publication; invalid derived returns are rejected without changing the previously published prefix or calling the optimizer.

The main account uses the adapter's `strategy_id`. `contributions` records each component's proposed share of main purchase budgets and model version. A missing quote may reject the resulting order; this table is not a fill ledger or separate PnL attribution. Use main orders/fills to determine execution status.

An optional main-account risk layer now reviews merged budgets before OMS submission. `approved_contributions` attributes the resulting approved budgets proportionally to component proposals; original `contributions` remain before risk review. See `Risk.md` for interfaces and limitations. Shadow risk controls are opt-in per `ShadowSpec`, with independent state.

## Optimizers

- `EqualWeightOptimizer` (EW): fixed equal weights; default baseline.
- `MinimumVarianceOptimizer` (MVP): long-only minimum variance using trailing shadow returns (default 63 observations), Ledoit-Wolf covariance shrinkage, and an optional maximum strategy weight. Passing an explicit `shrinkage` keeps the older fixed diagonal estimator. It uses equal weights until at least 20 completed returns exist, and when all series have effectively zero variance. Infeasible limits or solver failures stop the run.
- `MaximumSharpeOptimizer` (alias `MSPOptimizer`): long-only maximum Sharpe ratio. Expected returns are sample means; covariance is Ledoit-Wolf shrunk. `annual_rf` is converted to one period with `sessions_per_year` (default 252). A binding `max_weight` is a hard constraint; solver failure or an invalid solution raises `PortfolioOptimizationError`.
- `MaximumDiversificationOptimizer` (alias `MDPOptimizer`): long-only maximum diversification ratio `(w' sigma) / sqrt(w' Sigma w)`, with Ledoit-Wolf covariance. It needs no expected-return forecast. Exact zero-variance strategies receive zero weight.
- `EqualRiskContributionOptimizer` (alias `ERCOptimizer`): long-only equal-risk-contribution weights using trailing shadow returns. See the section below.
- Custom optimizers implement `PortfolioOptimizer.weights(AllocationContext)` and may expose reproducibility settings through `settings()`. The return history includes fee effects and any permitted stale marks; sample size and allocation choices require research validation. A low-risk or cash-heavy shadow can dominate minimum variance; this is an algorithmic property, not evidence of predictive quality.

### Covariance estimation

`ledoit_wolf_covariance(returns)` implements the Ledoit-Wolf well-conditioned
estimator,

```text
Sigma_LW = (1 - lambda) * Sigma_sample + lambda * mu * I,
mu = trace(Sigma_sample) / N,
```

with the analytical shrinkage intensity `lambda` from the paper. MVP, MSP, and
MDP use it by default. Passing `shrinkage=<float>` switches to the older fixed
diagonal estimator; `covariance="sample"` disables shrinkage. The chosen method
and intensity are recorded in `settings()` for each allocation.

MSP uses historical mean returns, so its weights are a sample estimate rather
than a forecast. MDP avoids expected returns entirely but still relies on the
sample covariance. Both are long-only, fully invested, and never use leverage.

```python
from backtesting import MaximumSharpeOptimizer, MaximumDiversificationOptimizer

msp = MaximumSharpeOptimizer(lookback=63, min_observations=20, annual_rf=0.02)
mdp = MaximumDiversificationOptimizer(lookback=63, min_observations=20)
runner = ShadowPortfolioBacktester(..., optimizer=mdp)
```

### Equal risk contribution (ERC)

For weights `w` and covariance `Sigma`, strategy `i` contributes
`w_i (Sigma w)_i / sigma_p` of portfolio risk, where
`sigma_p = sqrt(w' Sigma w)`. `EqualRiskContributionOptimizer` seeks equal
contributions subject to `w_i >= 0` and `sum(w_i) = 1`, so it is long-only and
never uses leverage.

- It uses trailing shadow returns, sample covariance, and optional shrinkage
  toward the diagonal (default 10%).
- Warm-up (`len(returns) < min_observations`) uses equal weights.
- A strategy with zero sample variance receives zero weight; the remaining
  strategies are solved on the reduced covariance.
- `max_weight` is a hard cap. If the cap prevents equal risk contributions, the
  optimizer raises instead of returning a non-ERC allocation.
- A Newton solver with backtracking on the Spinu convex objective solves the
  ERC first-order condition. It is deterministic and avoids the flat
  squared-dispersion objective that can stop far from the solution.
- `maxiter` (1 to 1,000,000) and `ftol` (0 to 1] bound the solver; a
  non-converged or non-ERC point raises rather than being published.
- After solving, risk contributions are recomputed and must satisfy
  `max_relative_dispersion <= rc_tolerance` (default `1e-8`). This catches a
  solver that reports success at a point that is not ERC.
- Input, configuration, cap, or non-convergence failures raise
  `PortfolioOptimizationError` (a `ValueError`) with `status`, `solver_message`,
  `iterations`, `objective`, candidate `weights`, `risk_contributions`, and
  `dispersion` where they are defined. The shadow coordinator marks the run
  `FAILED`; no main orders or fills are created.
- `optimizer.last_result` retains the latest diagnostics, including failures.
  The coordinator deep-copies the optimizer, so inspect
  `runner._main.strategy.optimizer.last_result` after a run or failure.

```python
from backtesting import ERCOptimizer, PortfolioOptimizationError

runner = ShadowPortfolioBacktester(
    ..., optimizer=ERCOptimizer(lookback=63, min_observations=20, shrinkage=.1),
)
try:
    bundle = runner.run(...)
except PortfolioOptimizationError as error:
    print(error.status, error.iterations, error.dispersion)
```

Each allocation record includes cutoff, available return count, component weight/model version, cash scaling, optimizer name, and settings. Settings and windows must not be tuned using later final-period performance.

## Example with the existing momentum demonstration

After loading prices, benchmark, membership, and fundamentals with the shared loader:

```python
from backtesting import ShadowPortfolioBacktester, MinimumVarianceOptimizer, BacktestAnalytics
from strategies.demo import MonthlyMomentum

runner = ShadowPortfolioBacktester(
    prices, benchmark, membership,
    shadows=[MonthlyMomentum("momentum_21", lookback=21),
             MonthlyMomentum("momentum_63", lookback=63)],
    optimizer=MinimumVarianceOptimizer(lookback=63, min_observations=20),
    fundamentals=fundamentals,
)
bundle = runner.run("2013-01-01", "2013-12-31", shadow_start="2012-01-03")
display(BacktestAnalytics(bundle.portfolio).summary())
display(bundle.allocations)
display(bundle.contributions)
shadow_nav = {name: result.history.nav for name, result in bundle.shadows.items()}
```

For fitted components, pass `ShadowSpec(strategy, training_panel, tuple(feature_columns))` instead of the bare strategy. Pass `market_data=MarketData.load(...)` instead of separate frames when component strategies declare `required_financials` or need volume; each shadow engine gets its own point-in-time view. Run a fresh coordinator for each experiment. The current implementation holds a separate engine/data view for each shadow; memory and computation grow with component count.

Requested ranges outside the supplied benchmark coverage are rejected by default. An explicit `allow_partial=True` on the coordinator warns and records requested/effective dates in each result's `coverage`. Use full required coverage for a final HW4 experiment; partial output must not be presented as the assigned full-period result.

## Verification

`tests/test_portfolio.py` checks a one-shadow equivalence to the standalone engine, synchronized warm-up and overlapping orders, valid minimum-variance weights, copied/append-only histories, invalid weights, failed components, and causal independence. `tests/test_erc.py` checks inverse-volatility two-asset weights, equal risk contributions with three assets, long-only/cap behavior, warm-up and zero-variance handling, non-convergence diagnostics, and the failed-run state with no main fills. It changes future prices, benchmark values, and unfinished labels, then compares earlier outputs. A full-run prefix is also compared with a truly truncated run. These checks validate the execution/data boundary, not future performance of any research model.
