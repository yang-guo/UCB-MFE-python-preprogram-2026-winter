# Learning guide: explaining the selected strategy

**Research question.** Can a simple relationship between relative price behavior,
risk and profitability rank stocks usefully over a 90-day holding period?
Separate three objects: the predictor produces scores, the portfolio rule turns
scores into orders, and the ledger records the economic result.

1. **A dated row is an information set.** A fiscal quarter ending in March is
   not known in March if the company files in May. Use filing+1, and ensure the
   next-session entry follows the signal. Training also needs completed labels.
2. **A rank is relative.** A momentum percentile of 0.9 means approximately the
   top tenth of peers on that date, not a forecast of 90% return. It can change
   when peer values change even if that stock's own momentum stays constant.
3. **Ridge regularizes correlated predictors.** It penalizes squared coefficient
   size. Standardization makes the penalty comparable across inputs; alpha1000
   was a development choice. More features or a nonlinear model need evidence,
   not an assumption that complexity must help.
4. **Fit and predict are different operations.** Fit learns medians, scales,
   coefficients and target bounds. Predict applies those saved quantities to new
   observations. This project refreshes inputs monthly without refitting.
5. **Target and trade must agree.** For signal t, entry e is the next session;
   maturity m is e+90 calendar days, moved to the next market session. The target
   is P(m)/P(e)-1, not a return beginning at t. Overlapping lots remain separate.
6. **A ledger is a conservation check.** NAV = cash + marked holdings + cash
   receivables. Buys reduce cash by notional plus cost; sells add notional minus
   cost. Corporate conversions require unit consistency and cannot create wealth
   just because a ticker or price scale changes.
7. **Performance is not prediction accuracy.** Rankings, idle cash, concentration,
   costs and the holding rule determine realized returns. Annual fresh-cash tests
   are diagnostics and must not be compounded into a continuous portfolio result.
8. **A test proves its stated condition.** Hand cases check arithmetic and timing;
   an independent ledger checks the full account; source comparisons check some
   data. None proves that every source row is true or that future returns follow
   the backtest. Keep unresolved evidence visible and avoid tuning on the final
   comparison.

Metric conventions: daily simple return r(t)=NAV(t)/NAV(t-1)-1; CAGR uses elapsed
calendar years from first evaluation close to final close; drawdown is
NAV(t)/running_max(NAV)-1. Sharpe = sqrt(252) × (mean(r)-rf_daily)/sample_sd(r),
where rf_daily=(1.02)^(1/252)-1. An initial pre-trade row includes entry costs in
strategy performance; benchmark purchase at the first test close has no initial
market return.

To review independently, explain why a missing future label is retained at
prediction time, trace one buy to its maturity, and reconcile a day's cash and
holdings. Then explain why the 5%/20% entry rules permit later weight drift.
These are the most useful interview exercises for this implementation.

Next research would need new chronological evidence: rolling stability of the
rank coefficients, broader issuer-aware data quality checks, and execution/cost
stress. Do not select another strategy from this reported final comparison.
