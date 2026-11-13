"""HW4 backtest: semi-annual equal-weight DCA, run through the course framework (backtest.py)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from backtest import (BacktestAnalytics, Backtester, OrderManagementSystem, Strategy)
from course_data import load_data, load_membership
from hw4_model import (DB, DEV_END, OUT, REBALANCE_MONTHS, TEST_END, TEST_START, HOLDING_DAYS)


class TailHistoryBacktester(Backtester):
    """Same simulation as Backtester; only the signal-date row of history is passed to the strategy."""
    def market_history(self, signal_date):
        return self.quotes.loc[:signal_date].tail(1).copy()


class SemiAnnualEqualWeight(Strategy):
    """First session of Jun/Dec: split ALL cash equally over the universe (optionally the model's top X%).

    Month-change scheduling adapted from MonthlyTopNStrategy.generate_orders; membership checks
    adapted from MomentumStrategy.score_stocks (selection_date <= signal_date).
    """
    def __init__(self, membership, holding_days=HOLDING_DAYS, months=REBALANCE_MONTHS,
                 scores=None, top_fraction=None):
        super().__init__(holding_days)
        if top_fraction is not None and (scores is None or not 0 < top_fraction <= 1):
            raise ValueError("top_fraction needs scores and 0 < fraction <= 1")
        m = membership[["quarter_start", "symbol", "selection_date"]].copy()
        for c in ["quarter_start", "selection_date"]:
            m[c] = pd.to_datetime(m[c])
        if m.isna().any().any() or m.duplicated(["quarter_start", "symbol"]).any():
            raise ValueError("Membership needs dates and one row per quarter and symbol")
        self.membership, self.months, self.top_fraction = m, tuple(months), top_fraction
        self.scores = {}
        if scores is not None:
            s = scores[["date", "symbol", "score"]].copy()
            s["date"] = pd.to_datetime(s.date)
            self.scores = {d: g.set_index("symbol").score for d, g in s.groupby("date")}

    def generate_orders(self, signal_date, execution_date, portfolio, history):
        if (execution_date.month not in self.months
                or execution_date.to_period("M") == signal_date.to_period("M")):
            return pd.Series(dtype=float)
        assert history.index[-1] == signal_date, "history must end at the signal close"
        quarter = execution_date.to_period("Q").start_time
        members = self.membership.loc[self.membership.quarter_start.eq(quarter)
                                      & self.membership.selection_date.le(signal_date), "symbol"]
        last = history.iloc[-1]                       # prices known at the signal close only
        tradable = last.index[last.notna() & last.index.isin(members)]
        if self.top_fraction is not None:
            s = self.scores.get(signal_date)
            if s is None:
                return pd.Series(dtype=float)
            s = s.reindex(tradable).dropna().sort_index().sort_values(ascending=False, kind="mergesort")
            tradable = s.index[:max(1, int(np.ceil(self.top_fraction * len(s))))] if len(s) else s.index
        budget = max(portfolio.cash - 1e-6, 0.)
        if len(tradable) == 0 or budget <= 0:
            return pd.Series(dtype=float)
        return pd.Series(budget / len(tradable), index=tradable, name="budget", dtype=float)


def load_market(database=DB, end=TEST_END):
    prices, _, _, benchmark = load_data(database, end=end)
    return prices, benchmark


def run(strategy, start, end, market, fee_bps=5, initial_cash=1_000_000, max_stale_sessions=10_000):
    """max_stale_sessions is large on purpose: a delisted holding is carried at its last observed
    price instead of crashing the run (HeldPriceGapError). See 'Open policy question'."""
    prices, benchmark = market
    bt = TailHistoryBacktester(prices, benchmark, strategy,
                               OrderManagementSystem(initial_cash, fee_bps),
                               max_stale_sessions=max_stale_sessions)
    return bt.run_backtest(start, end)


def audit_result(res, end, months=REBALANCE_MONTHS, holding_days=HOLDING_DAYS, initial_cash=1_000_000):
    """Accounting and timing checks (reconciliation asserts adapted from 1_backtesting.ipynb)."""
    h, f = res.history, res.fills
    np.testing.assert_allclose(h.nav, h.cash + h.holdings_value)
    cash_changes = f.groupby("date").cash_change.sum().reindex(h.index, fill_value=0)
    np.testing.assert_allclose(h.cash, initial_cash + cash_changes.cumsum(), atol=1e-5)
    assert h.cash.min() >= -1e-6, "negative cash"
    buys, sells = f[f.side.eq("BUY")], f[f.side.eq("SELL")]
    assert buys.signal_date.lt(buys.date).all(), "fill not after signal"
    assert buys.date.dt.month.isin(months).all()
    assert (buys.signal_date.dt.month != buys.date.dt.month).all(), "not first session of month"
    entry = buys.set_index("lot_id").date
    held_days = (sells.date - sells.lot_id.map(entry)).dt.days
    assert (held_days >= holding_days).all(), "sold before 90 calendar days"
    assert sells.date.ge(sells.due_date).all()
    assert f.date.max() <= pd.Timestamp(end) and h.index.max() <= pd.Timestamp(end)
    assert np.isclose(res.pnl.total_pnl, res.pnl.realized_pnl + res.pnl.unrealized_pnl)
    assert np.isclose(res.pnl.total_fees, f.fee.sum())
    unsold = res.open_lots[res.open_lots.due_date <= pd.Timestamp(end)]
    return pd.Series({"buys": len(buys), "sells": len(sells), "max_hold_days": held_days.max(),
                      "median_hold_days": held_days.median(), "open_lots_end": len(res.open_lots),
                      "matured_but_unsold_lots": len(unsold),
                      "rejected_missing_quote": int(res.diagnostics.event.eq("rejected_missing_quote").sum()),
                      "min_cash": h.cash.min()})


def check_signal_timing(market, membership, signal_date, execution_date, n_symbols=200):
    """Orders must not change when prices AFTER the signal date are altered."""
    prices, benchmark = market
    keep = sorted(prices.symbol.unique())[:n_symbols]
    p1 = prices[prices.symbol.isin(keep)].copy()
    p2 = p1.copy()
    p2.loc[p2.date > signal_date, "adj_close"] *= 3
    out = []
    for p in (p1, p2):
        s = SemiAnnualEqualWeight(membership)
        bt = TailHistoryBacktester(p, benchmark, s, OrderManagementSystem())
        out.append(s.generate_orders(signal_date, execution_date, bt.oms.snapshot({}),
                                     bt.market_history(signal_date)))
    pd.testing.assert_series_equal(out[0], out[1])
    return out[0]


def plot_cumulative(res, path):
    h = res.history
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, height_ratios=[3, 1])
    (h[["nav", "benchmark_nav"]] / h.nav.iloc[0] - 1).rename(
        columns={"nav": "Strategy (net of fees)", "benchmark_nav": "S&P 500 total return"}).plot(ax=ax[0])
    ax[0].set_ylabel("Cumulative return")
    ax[0].set_title("Semi-annual equal-weight DCA vs S&P 500 TR")
    (h.cash / h.nav).plot(ax=ax[1], label="cash / NAV")
    ax[1].legend()
    plt.tight_layout()
    fig.savefig(path, dpi=150)


def main(chosen="all_stocks"):
    market, membership = load_market(), load_membership(DB, end=TEST_END)

    # 1) Development-period strategy choice (no 2016+ results used for this decision).
    dev_scores = pd.read_csv(OUT / "dev_scores.csv", parse_dates=["date", "fit_date"])
    variants = {"all_stocks": None, "top_50pct": .5, "top_20pct": .2}
    dev_rows = {}
    for name, frac in variants.items():
        strat = SemiAnnualEqualWeight(membership, scores=dev_scores if frac else None, top_fraction=frac)
        res = run(strat, "2012-01-01", DEV_END, market)
        audit_result(res, DEV_END)
        s = BacktestAnalytics(res).summary()
        dev_rows[name] = s.strategy
    dev_rows["S&P 500 TR"] = s["S&P 500 total return"]
    print(pd.DataFrame(dev_rows))
    pd.DataFrame(dev_rows).to_csv(OUT / "dev_strategy_comparison.csv")

    # 2) Final test: run ONCE with the strategy chosen above.
    test_scores = pd.read_csv(OUT / "test_scores.csv", parse_dates=["date", "fit_date"])
    frac = variants[chosen]
    strat = SemiAnnualEqualWeight(membership, scores=test_scores if frac else None, top_fraction=frac)
    res = run(strat, TEST_START, TEST_END, market)
    print(audit_result(res, TEST_END))
    an = BacktestAnalytics(res)
    print(an.summary()); print(an.diagnostics()); print(res.pnl)
    an.summary().to_csv(OUT / "test_summary.csv")
    res.history.to_csv(OUT / "test_nav.csv"); res.fills.to_csv(OUT / "test_fills.csv", index=False)
    res.open_lots.to_csv(OUT / "test_open_lots.csv", index=False)
    plot_cumulative(res, OUT / "cumulative_return_vs_sp500tr.png")


if __name__ == "__main__":
    main()