"""Step 2: run the chosen strategy once on the final test, Apr 1 2016 - Sep 30 2026.

Run after validate.py:  python final_test.py

The settings come from results/chosen_config.json, written by validate.py.
The model is refit at the last signal of every March, on outcomes that had already ended.
"""

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import research
from backtester import Backtester, benchmark_nav, performance
from pipeline import RESULTS, load, log, scores_for, sectors, tradeable

TEST_START, TEST_END = "2016-04-01", "2026-09-30"


def main():
    cfg = json.loads((RESULTS / "chosen_config.json").read_text())
    md, panel = load()
    X = research.rank_features(panel)

    # Fit dates: the last March signal of 2016, 2017, ..., 2026 (2016-03-31 first).
    march = panel.loc[panel.date.dt.month.eq(3) & panel.date.ge("2016-03-01"), "date"]
    fit_dates = sorted(march.groupby(march.dt.year).max())
    preds, fits = research.walk_forward(panel, X, fit_dates, cfg["model"], cfg["target"])
    fits.to_csv(RESULTS / "test_fit_schedule.csv", index=False)
    log(f"fits:\n{fits.to_string()}")
    test_rows = panel.date.ge(fit_dates[0])
    q = research.signal_quality(panel.loc[test_rows], preds[test_rows], top_n=cfg["top_n"])
    q.to_csv(RESULTS / "test_signal_quality.csv", header=["value"])
    print(q.round(4).to_string())

    scores = scores_for(panel, preds, tradeable(panel, cfg["min_dollar_volume"]) & test_rows)
    prices = md.prices_frame()
    strategy = research.WeeklyTopN(scores, sectors(md), top_n=cfg["top_n"],
                                    cohort_fraction=cfg["cohort_fraction"], name_cap=cfg["name_cap"],
                                    sector_cap=cfg["sector_cap"])
    runs = {}
    for policy in ["last", "zero"]:  # value stocks that stop trading at last price, or at zero
        runs[policy] = Backtester(prices, cost_bps=cfg["cost_bps"], dark_policy=policy).run(
            strategy, TEST_START, TEST_END)
    res = runs["last"]
    bench = benchmark_nav(md.benchmark, TEST_START, TEST_END)
    assert bench.index[0] == res.nav.index[0], "strategy and benchmark must start at the same close"

    table = pd.DataFrame({"Strategy": performance(res.nav),
                          "Strategy (stopped stocks at $0)": performance(runs["zero"].nav),
                          "S&P 500 TR": performance(bench)})
    table.to_csv(RESULTS / "test_performance.csv")
    print(table.to_string())

    res.daily.to_csv(RESULTS / "test_daily_nav.csv")
    res.trades.to_csv(RESULTS.parent / "outputs" / "test_trades.csv", index=False)
    res.open_lots.to_csv(RESULTS / "test_open_lots_at_end.csv", index=False)
    res.events.groupby("event").size().to_csv(RESULTS / "test_events_count.csv", header=["count"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(res.nav / 1e6 - 1, label="My strategy")
    ax.plot(bench / 1e6 - 1, label="S&P 500 total return")
    ax.set_title("Cumulative return, Apr 2016 - Sep 2026 ($1M start)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "cumulative_returns.png", dpi=120)

    check_timing(res, prices, fits)
    check_held_prices(res, prices)
    log("saved results/")


def check_timing(res, prices, fits):
    """Check every real trade against the price table and the timing rules."""
    cal = prices.index
    t = res.trades
    buys, sells = t[t.side == "BUY"], t[t.side == "SELL"]
    nxt = cal[cal.get_indexer(buys.signal_date) + 1]
    px = prices.stack()
    checks = {
        "buys": len(buys),
        "sells": len(sells),
        "buy fills on the session after its signal": bool((buys.date.to_numpy() == nxt.to_numpy()).all()),
        "buy price = repaired adjusted close on fill day": bool(np.allclose(buys.price, px.loc[list(zip(buys.date, buys.symbol))])),
        "due date = entry + 90 calendar days": bool(((buys.due_date - buys.date).dt.days == 90).all()),
        "sell price = repaired adjusted close on sell day": bool(np.allclose(sells.price, px.loc[list(zip(sells.date, sells.symbol))])),
        "no sale before due date": bool((sells.date >= sells.due_date).all()),
        "sale is the first quoted session on/after due date": bool(all(
            prices.loc[d:s, sym].first_valid_index() == s
            for d, s, sym in zip(sells.due_date, sells.date, sells.symbol))),
        "no trades after test end": bool(t.date.max() <= pd.Timestamp(TEST_END)),
        "every fit used only outcomes ended before the fit date": bool((fits.last_label_end < fits.fit_date).all()),
        "cash never negative": bool((res.daily.cash >= -1e-6).all()),
        "NAV = cash + holdings every day": bool(np.allclose(res.daily.nav, res.daily.cash + res.daily.holdings_value)),
    }
    pd.Series(checks).to_csv(RESULTS / "test_timing_checks.csv", header=["result"])
    print(pd.Series(checks).to_string())

    # Worked examples for the README: the first lot, an AAPL lot whose due
    # date fell on a weekend (so it sells the next session), and the last exit.
    lots = buys.drop(columns="pnl").merge(sells[["lot_id", "date", "price", "pnl"]], on="lot_id",
                                          suffixes=("_buy", "_sell"))
    shifted = lots[lots.due_date.map(lambda d: d not in cal) & lots.symbol.eq("AAPL")]
    picks = pd.concat([lots.head(1), shifted.head(1), lots.tail(1)])
    ex = picks[["symbol", "signal_date", "date_buy", "price_buy", "units", "fee", "due_date", "date_sell",
                "price_sell", "pnl"]]
    ex.to_csv(RESULTS / "test_trade_examples.csv", index=False)
    print(ex.to_string())


def check_held_prices(res, prices):
    """Flag closed lots with a huge one-day move while we held them.

    research.clean_prices repairs the breaks it can recognize (missed splits,
    bad prints). This checks what is left and how much profit it explains.
    """
    t = res.trades
    lots = t[t.side == "BUY"].drop(columns="pnl").merge(t.loc[t.side == "SELL", ["lot_id", "date", "price", "pnl"]],
                                     on="lot_id", suffixes=("_buy", "_sell"))
    jumps = []
    for row in lots.itertuples():
        path = prices.loc[row.date_buy:row.date_sell, row.symbol].dropna()
        jumps.append(np.log(path / path.shift(1)).abs().max())
    lots["max_abs_log_move"] = jumps
    lots["lot_return"] = lots.price_sell / lots.price_buy - 1
    # 2.5x, a bit stricter than the 3x entry screen, so near-misses show up.
    flagged = lots[lots.max_abs_log_move >= np.log(2.5)]
    summary = pd.Series({"closed lots": len(lots), "lots with a >2.5x one-day move while held": len(flagged),
                         "PnL of those lots": flagged.pnl.sum(), "total realized PnL": lots.pnl.sum(),
                         "largest single-lot return": lots.lot_return.max(),
                         "smallest single-lot return": lots.lot_return.min()})
    summary["symbols of those lots"] = ", ".join(sorted(flagged.symbol.unique()))
    summary.to_csv(RESULTS / "test_held_price_audit.csv", header=["value"])
    flagged[["symbol", "date_buy", "date_sell", "lot_return", "pnl"]].to_csv(RESULTS / "test_flagged_lots.csv", index=False)

    # Capital stuck in stocks whose quotes stopped (mostly takeovers: no payout in the data).
    end = res.daily.index[-1]
    o = res.open_lots
    dark = o[o.last_quote_date < end - pd.Timedelta(days=7)]
    pd.Series({"open lots at end": len(o), "lots with no quote in the last week": len(dark),
               "securities": dark.symbol.nunique(), "value at last price": dark.market_value.sum(),
               "share of final NAV": dark.market_value.sum() / res.nav.iloc[-1],
               "total fees paid": res.trades.fee.sum(),
               "average cash share": (res.daily.cash / res.daily.nav).mean()}
              ).to_csv(RESULTS / "test_stuck_capital.csv", header=["value"])
    cols = ["symbol", "date_buy", "date_sell", "price_buy", "price_sell", "lot_return", "pnl", "max_abs_log_move"]
    lots.nlargest(10, "pnl")[cols].to_csv(RESULTS / "test_biggest_winners.csv", index=False)
    lots.nsmallest(10, "pnl")[cols].to_csv(RESULTS / "test_biggest_losers.csv", index=False)
    print(summary.to_string())
    print(lots.nlargest(5, "pnl")[cols].to_string())


if __name__ == "__main__":
    main()
