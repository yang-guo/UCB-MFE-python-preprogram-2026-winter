"""Final test: Apr 1 2016 - Sep 30 2026 with the configuration frozen in
pipeline.CHOSEN (the submission).

    ../../../.venv/bin/python run_final.py               # submission -> results/final_*
    ../../../.venv/bin/python run_final.py --variant v2  # post-test variant -> results/v2_check_*

Writes results/ (metrics, annual returns, accounting checks, chart) and the
full fill/daily/event logs to outputs/ (git-ignored).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtester import performance
import argparse

from features import FEATURES
from pipeline import CHOSEN, OUTPUTS, RESULTS, TEST_BACKTEST, V2_CONFIG, compare, run_strategy, setup


def accounting_checks(result, data, nav, end):
    """Independent re-derivations of the ledger from the fill log and raw prices."""
    fills, daily, ledger = result.fills, result.daily, result.ledger
    cal = data.calendar
    checks = {}
    # 1. Cash reconciles: initial + all fill cash flows == final cash; never negative.
    checks["cash_reconciles"] = np.isclose(ledger.initial_cash + fills.cash_change.sum(), ledger.cash, atol=1e-4)
    checks["cash_never_negative"] = bool((fills.cash_after >= -1e-6).all() and (daily.cash >= -1e-6).all())
    # 2. NAV = cash + open lots at their last observed price (end date).
    open_value = (result.open_lots.units * result.open_lots.last_mark).sum() if len(result.open_lots) else 0.0
    checks["final_nav_equals_cash_plus_open_lots"] = np.isclose(daily.nav.iloc[-1], ledger.cash + open_value)
    # 3. PnL: realized + unrealized (after fees) == NAV change.
    unreal = ((result.open_lots.last_mark - result.open_lots.entry_price) * result.open_lots.units
              - result.open_lots.entry_fee).sum() if len(result.open_lots) else 0.0
    checks["pnl_reconciles"] = np.isclose(fills.realized_pnl.sum() + unreal, daily.nav.iloc[-1] - ledger.initial_cash,
                                          atol=1e-3)
    # 4. Fill prices are the adjusted close of the fill date; signal strictly precedes buys.
    t_idx = cal.get_indexer(fills.date)
    n_idx = data.symbols.get_indexer(fills.symbol)
    checks["fill_prices_match_close"] = bool(np.allclose(fills.price, data.adj_close[t_idx, n_idx], rtol=1e-6))
    buys, sells = fills[fills.side == "BUY"], fills[fills.side == "SELL"]
    checks["signal_is_previous_session"] = bool(
        (cal.get_indexer(buys.signal_date) == cal.get_indexer(buys.date) - 1).all())
    # 5. Exits: never before 90 calendar days; at the first session with a quote on/after maturity.
    held = (sells.date - sells.entry_date).dt.days
    checks["no_exit_before_90_days"] = bool((held >= 90).all())
    first_quote = []
    for _, s in sells.iterrows():
        n = data.symbols.get_loc(s.symbol)
        start = cal.searchsorted(s.due_date)
        q = np.flatnonzero(np.isfinite(data.adj_close[start:, n]))
        first_quote.append(cal[start + q[0]] == s.date)
    checks["exit_at_first_quote_on_or_after_maturity"] = bool(all(first_quote))
    # 6. Each buy's lot was sold exactly once or is still open.
    sold_ids = set(sells.lot_id)
    open_ids = set(result.open_lots.lot_id) if len(result.open_lots) else set()
    checks["every_lot_closed_once_or_open"] = (len(sells.lot_id) == len(sold_ids)
                                               and sold_ids | open_ids == set(buys.lot_id)
                                               and not sold_ids & open_ids)
    # 7. No data after the test end.
    checks["no_dates_after_end"] = bool(fills.date.max() <= pd.Timestamp(end) and cal[-1] <= pd.Timestamp(end))
    # 8. Daily NAV recomputed from scratch on a sample of dates.
    sample = daily.index[:: max(1, len(daily) // 25)]
    ok = []
    for day in sample:
        d = cal.get_loc(day)
        bought = fills[(fills.side == "BUY") & (fills.date <= day)]
        closed = set(fills.loc[(fills.side == "SELL") & (fills.date <= day), "lot_id"])
        live = bought[~bought.lot_id.isin(closed)]
        cash = ledger.initial_cash + fills.loc[fills.date <= day, "cash_change"].sum()
        value = 0.0
        frozen = result.frozen[result.frozen.date <= day].set_index("lot_id").frozen_mark
        for _, lot in live.iterrows():
            if lot.lot_id in frozen.index:          # quotes after a long gap are not trusted
                value += lot.units * frozen[lot.lot_id]
                continue
            n = data.symbols.get_loc(lot.symbol)
            q = np.flatnonzero(np.isfinite(data.adj_close[: d + 1, n]))
            value += lot.units * float(data.adj_close[q[-1], n])  # float(): numpy 2 keeps float32 otherwise
        ok.append(np.isclose(cash + value, daily.loc[day, "nav"], rtol=1e-9))
    checks["daily_nav_recomputed_on_sample_dates"] = bool(all(ok))
    return pd.Series(checks, name="passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["submitted", "v2"], default="submitted")
    args = parser.parse_args()
    config = CHOSEN if args.variant == "submitted" else V2_CONFIG
    prefix = "final_" if args.variant == "submitted" else "v2_check_"
    OUTPUTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data, feats, target, label_end, panel = setup()
    start, end = TEST_BACKTEST
    print("Frozen configuration:", config)
    result, strategy, schedule, nav, bench = run_strategy(data, feats, panel, start, end, **config)

    metrics = compare(nav, bench, "Strategy" if args.variant == "submitted" else "v2 variant (post-test)")
    print("\nFinal test", TEST_BACKTEST, "\n", metrics.to_string())
    metrics.to_csv(RESULTS / f"{prefix}metrics.csv")
    s, b = metrics.iloc[:, 0], metrics.iloc[:, 1]
    beat = s.final_value > b.final_value
    print(f"\n{'Beat' if beat else 'Did not beat'} the benchmark: final value "
          f"${s.final_value:,.0f} vs ${b.final_value:,.0f} ({s.final_value / b.final_value - 1:+.1%}); "
          f"CAGR {s.CAGR:.2%} vs {b.CAGR:.2%} ({(s.CAGR - b.CAGR) * 100:+.2f} pp)")

    # Calendar-year returns (2016 and 2026 are partial years).
    yearly = pd.DataFrame({"strategy": nav, "benchmark": bench}).resample("YE").last()
    yearly = yearly.pct_change().fillna(yearly.iloc[0] / 1_000_000 - 1)
    yearly.index = yearly.index.year
    yearly["excess"] = yearly.strategy - yearly.benchmark
    print("\nCalendar-year returns:\n", yearly.round(4).to_string())
    yearly.to_csv(RESULTS / f"{prefix}annual_returns.csv")

    checks = accounting_checks(result, data, nav, end)
    print("\nAccounting checks:\n", checks.to_string())
    checks.to_csv(RESULTS / f"{prefix}accounting_checks.csv")
    assert checks.all(), "An accounting check failed"

    # Diagnostics.
    d = result.daily
    fills = result.fills
    buys = fills[fills.side == "BUY"]
    diag = pd.Series({
        "cohorts_formed": len(strategy.log),
        "buy_fills": len(buys), "sell_fills": int((fills.side == "SELL").sum()),
        "distinct_stocks_bought": buys.symbol.nunique(),
        "total_fees": fills.fee.sum(),
        "turnover_bought_per_year_over_avg_nav": buys.notional.sum() / d.nav.mean() / ((d.index[-1] - d.index[0]).days / 365.25),
        "average_cash_fraction": (d.cash / d.nav).mean(),
        "average_names_held": d.names.mean(),
        "average_slots_filled_agreement_cutoff": pd.DataFrame(strategy.log).slots_filled.mean(),
        "open_lots_at_end": len(result.open_lots),
        "rejected_buys_no_quote": int((result.events.event == "buy_rejected_no_quote").sum()),
        "late_exits_over_4_days": int((result.events.event == "late_exit").sum()),
        "stale_mark_incidents": int((result.events.event == "stale_mark").sum()),
        "lots_frozen_after_quote_gap": len(result.frozen),
        "model_refits": len(schedule.fit_log),
    })
    # Lots still open at the end because their stock stopped quoting before maturity.
    if len(result.open_lots):
        stuck = result.open_lots[result.open_lots.due_date <= pd.Timestamp(end)]
        diag["stuck_lots_past_maturity"] = len(stuck)
        diag["stuck_value_at_last_mark"] = float((stuck.units * stuck.last_mark).sum())
        stuck.to_csv(RESULTS / f"{prefix}stuck_lots.csv", index=False)
    print("\nDiagnostics:\n", diag.to_string())
    diag.to_csv(RESULTS / f"{prefix}diagnostics.csv")

    # Sensitivity: value any stuck lot at zero (no recovery) instead of its last mark.
    if len(result.open_lots) and diag.get("stuck_lots_past_maturity", 0):
        zero = nav.copy()
        zero.iloc[-1] -= diag.stuck_value_at_last_mark
        print(f"\nIf stuck lots were worth $0 at the end: final value ${zero.iloc[-1]:,.0f}, "
              f"CAGR {performance(zero).CAGR:.2%}")

    # Post-hoc diagnosis (not used for any choice): did the model still rank stocks
    # out of sample, and how did its universe do versus the cap-weighted index?
    # Same Ridge, refitted each January on matured labels, scored weekly.
    from model import daily_ic, walk_forward
    oos = walk_forward("ridge", panel, data, range(2016, 2027), features=config.get("features", FEATURES))
    oos = oos[oos.date.ge("2016-03-31") & oos.tradeable]
    oos["r"] = oos.groupby("date").score.rank(ascending=False, method="first")
    by_date = pd.DataFrame({"ic": daily_ic(oos),
                            "top30": oos[oos.r <= 30].groupby("date").target.mean(),
                            "universe_ew": oos.groupby("date").target.mean()})
    by_date["top30_minus_universe"] = by_date.top30 - by_date.universe_ew
    bench_90 = []
    for day in by_date.index:
        t = data.calendar.get_loc(day)
        due = data.calendar.searchsorted(data.calendar[t + 1] + pd.Timedelta(days=90))
        bench_90.append(data.benchmark.iloc[due] / data.benchmark.iloc[t + 1] - 1 if due < len(data.calendar) else np.nan)
    by_date["sp500tr"] = bench_90
    oos_table = by_date.groupby(by_date.index.year).mean()
    oos_table.loc["all"] = by_date.mean()
    print("\nOut-of-sample predictive diagnostics (weekly signals, mean 90-day returns):\n",
          oos_table.round(4).to_string())
    oos_table.to_csv(RESULTS / f"{prefix}oos_prediction_diagnostics.csv")

    # Save full logs (git-ignored) and the cohort log (small).
    d.to_csv(OUTPUTS / f"{prefix}daily.csv")
    fills.to_csv(OUTPUTS / f"{prefix}fills.csv", index=False)
    result.events.to_csv(OUTPUTS / f"{prefix}events.csv", index=False)
    pd.DataFrame(strategy.log).drop(columns=["top_scores", "top_agreement", "top_picked"]).to_csv(
        RESULTS / f"{prefix}cohorts.csv", index=False)
    pd.DataFrame(schedule.fit_log).to_csv(RESULTS / f"{prefix}model_refits.csv", index=False)
    buys.head(30).to_csv(RESULTS / f"{prefix}first_cohort_fills.csv", index=False)

    if args.variant != "submitted":
        return
    # Chart: cumulative return of both portfolios.
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[3, 1])
    ax.plot(nav.index, nav / nav.iloc[0] - 1, label="Strategy (Ridge, top-30, 3 cohorts)", color="#1f77b4")
    ax.plot(bench.index, bench / bench.iloc[0] - 1, label="S&P 500 total return (^SP500TR)", color="#444444")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Cumulative return")
    ax.set_title(f"$1M from {nav.index[1].date()} to {nav.index[-1].date()} (after 10 bps per side)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    for series, color in [(nav, "#1f77b4"), (bench, "#444444")]:
        ax2.plot(series.index, series / series.cummax() - 1, color=color)
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax2.set_ylabel("Drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "cumulative_returns.png", dpi=130)
    print("\nSaved chart to", RESULTS / "cumulative_returns.png")


if __name__ == "__main__":
    main()
