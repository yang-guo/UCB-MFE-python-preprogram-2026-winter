# Risk management: signal-close purchase controls

`RiskManager` reviews combined stock purchase budgets before the OMS receives them. It composes `RiskPolicy` subclasses; it does not inherit the portfolio optimizer or strategy. The optimizer chooses capital allocation, risk policies can only reduce it, and the OMS books validated fills. The feature is optional: `risk_manager=None` preserves existing behavior.

## Interfaces

| Interface | Responsibility |
|---|---|
| `RiskContext` | Signal/execution dates, an engine-owned signal-close portfolio snapshot, completed main-account NAV history, fee rate, strategy ID and model version. No execution-day quote or unrestricted engine access. |
| `RiskPolicy.apply(context, budgets)` | Abstract method returning a `RiskAdjustment` with reduced budgets, reasons, and diagnostic metrics. |
| `RiskManager.review(context, proposed)` | Validate historical inputs, run policies, enforce available cash, and return a `RiskDecision`. |
| `RiskDecision` | Original and approved budgets plus a per-policy audit table. |

The engine builds `RiskContext.portfolio` from its own OMS snapshot at the signal close; it does not reuse the planning object handed to the strategy. Proposal symbols must be exact Python strings (NumPy string scalars are normalized), so a crafted `str` subclass cannot split one ticker into several cap budgets or mislabel orders and fills. Proposed amounts are converted to plain finite nonnegative floats at the trust boundary, so a numeric subclass cannot lie to cap arithmetic. The reviewed holdings must be a plain `dict`; a mapping subclass is rejected because it could report one exposure for reconciliation and another to a policy. The manager then deep-copies policy state, inputs, and audit data. Failed or invalid policy decisions do not commit gate state or audit records. A policy cannot add a symbol, return a nonfinite/negative budget, or increase an incoming amount. The manager rejects repeated signal dates and changes to previously supplied NAV history.

## Policies and order

1. **`DrawdownGate`** (priority 10): reduce new purchases once drawdown reaches or exceeds `threshold`, including exact equality. The comparison allows a few units in the last place so a 10% loss measured as `0.09999999999999998` still triggers a 10% gate. `scale=0` blocks entries; a value between zero and one reduces them. The high-water mark uses completed main-account NAV from initial capital, including fees and open positions. With `resume_drawdown=None`, activation latches for the rest of the run. An explicit lower recovery threshold enables hysteresis, and recovery releases at exact equality with the same tolerance. It does not pause market time, refits, or mandatory maturity exits.
2. **`VolatilityTarget`** (priority 20): estimate sample realized daily-return volatility over the trailing `lookback`, annualized by `sqrt(sessions_per_year)`. Once `min_observations` exist, apply `min(1, annual_target / realized_volatility)`. Zero volatility never introduces leverage. Insufficient history follows explicit `warmup="allow"` (default) or `"block"`, and is recorded. It uses the main account, not shadow strategy returns. Nonfinite estimates stop the run.
3. **`SingleNameCap`** (priority 30): aggregate all existing lots by symbol and calculate `room = max(0, cap * signal_NAV - current_stock_value)`. Convert room to a fee-inclusive purchase budget. Check the combined proposals, including overlapping component strategies. Existing breaches prevent further buys and are logged even if no purchase was proposed. Unused room/cash is not redistributed.
4. **Cash limit**: proportionally reduce any remaining proposals to signal-date available cash. Conservative representable rounding keeps the combined budget within cash. Rejected capital stays idle.

Policies are sorted by priority; input order breaks ties. Later policies cannot reverse an earlier reduction. Each stage records dates, strategy/model provenance, symbol, before/after budget, reason, and metrics. Drawdown records its peak, loss and activation state; volatility records observations, estimate and scaling; stock caps record current exposure and projected signal-close weight.

## Limit semantics

The cap denominator is **pre-trade signal-close NAV**. A 25% cap means no proposed purchase adds more exposure than that baseline permits. It is not a guarantee that the realized next-close weight is at most 25%: execution-day prices can move, entry fees reduce NAV, and later market moves can increase an existing weight.

HW4 requires holding each purchase for 90 calendar days. These controls never force early liquidation. A drawdown restriction cannot stop losses on existing lots. Volatility scaling reduces new deployment, not current holdings, and therefore cannot guarantee a whole-portfolio volatility target. Realized estimates may lag or be low while much of the account is in cash. A covariance-based projected-risk model would be a separate research/design extension.

## Standalone usage

```python
from backtesting import RiskManager, SingleNameCap, DrawdownGate, VolatilityTarget, Backtester

controls = RiskManager([
    DrawdownGate(threshold=0.15, scale=0.0, resume_drawdown=0.05),
    VolatilityTarget(annual_target=0.10, lookback=63, min_observations=20, warmup="allow"),
    SingleNameCap(cap=0.25),
])
engine = Backtester(prices, benchmark, membership, strategy, risk_manager=controls)
result = engine.run(start, end)
display(result.risk)
```

The engine owns its own copy of `controls`. Risk review happens after refitting/strategy proposals and before OMS reservations. The strategy does not receive execution-day prices to change its proposal after review. The engine retains `raw_proposals`; `pending_proposals` contains approved budgets. No fill is invented when an entry quote is missing.

## Portfolio/shadow usage

Pass the same `risk_manager=controls` argument to `ShadowPortfolioBacktester` to control the **main account** after portfolio allocation. Shadow accounts are unaffected by default, so their independently simulated performance is not contaminated by the main account's risk state. To apply controls deliberately to a component's standalone simulation, set `ShadowSpec(..., risk_manager=its_controls)`; each engine gets an independent copy.

- `bundle.portfolio.risk`: main risk decisions and metrics.
- `bundle.contributions`: component proposals before main risk review (existing meaning).
- `bundle.approved_contributions`: component `proposed_budget` and proportionally attributed `approved_budget` after review. These sums match approved main orders; they are not component positions or realized PnL.
- Missing execution quotes can still reject approved orders. Consult `orders` and `fills` for execution, rather than treating risk approval as a transaction.

Main risk history starts with main-account initial capital, not earlier shadow warm-up. Main and shadow decisions still complete behind the shared execution barrier.

## Validation

`tests/test_risk.py` checks hand-calculated fee-inclusive cap room, overlapping proposals, existing breaches, drawdown hysteresis/latching and exact threshold/recovery boundaries, scalar volatility estimates, warm-up, no leverage, cash scaling, failed-policy state isolation, history mutation/replay rejection, 90-day lot preservation, portfolio attribution, future-price independence, and adversarial strategies that relabel holdings, inflate the planning snapshot, mutate during `needs_refit`, or return hostile numeric types. It also checks that an injected universe row or non-string membership symbol cannot pass the engine's membership gate, that a hostile holdings mapping is rejected, that policy parameters are canonicalized, that large equal-weight books converge under the cash cap, and that coordinator attribution failures, `finish()` failures, and partial `advance()` failures leave the run `FAILED` without committed fills. The full regression suite currently contains 201 tests. `audits/Testing.ipynb` includes an interactive example.

The earlier five-agent audit in `audits/README.md` predates this extension; it must not be read as independent certification of the risk module. Tests do not establish research-model quality, source vintages, or actual risk-target achievement.
