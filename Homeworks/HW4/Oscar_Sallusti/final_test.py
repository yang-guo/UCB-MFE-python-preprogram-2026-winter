"""Final out-of-sample test, Apr 1 2016 - Sep 30 2026, with the locked config.

    .venv/bin/python Homeworks/HW4/Oscar_Sallusti/final_test.py [path/to/database]

Writes small result files to results/ (committed) and the full trade log and
daily history to outputs/ (ignored by Git).

Benchmark alignment: both portfolios hold $1,000,000 cash at the 2016-03-31
close. The strategy's first purchases and the benchmark's single purchase
both happen at the 2016-04-01 close. Both are valued through 2026-09-30.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
from backtester import Backtester, check_accounting, performance_metrics  # noqa: E402
from features import build_panel, load_fundamental_events, load_market, signal_dates  # noqa: E402
from model import prediction_report, walk_forward  # noqa: E402
from strategy import TopScoreStrategy  # noqa: E402

DEFAULT_DB = HERE.parents[2] / "Lectures/modeling/data/equities_top2000.duckdb"
RESULTS, OUT = HERE / "results", HERE / "outputs"


def refit_dates(calendar):
    sig = signal_dates(calendar, C.FIRST_SIGNAL, C.TEST_END)
    years = range(2016, pd.Timestamp(C.TEST_END).year + 1)
    dates = [sig[sig >= pd.Timestamp(f"{y}-03-31")][0] for y in years
             if (sig >= pd.Timestamp(f"{y}-03-31")).any()]
    assert dates[0] == pd.Timestamp(C.FIRST_SIGNAL)
    return dates


def hand_check_lots(db, fills, n=5, seed=0):
    """Re-derive a few round trips straight from the database."""
    con = duckdb.connect(str(db), read_only=True)
    buys = fills[fills.side == "BUY"].sample(n, random_state=seed)
    rows = []
    for b in buys.itertuples():
        s = fills[(fills.lot_id == b.lot_id) & (fills.side == "SELL")]
        px = con.execute("""SELECT date, adj_close FROM prices WHERE symbol = ? AND volume > 0
                            AND date >= ? ORDER BY date""", [b.symbol, b.date]).fetchdf()
        px["date"] = pd.to_datetime(px.date)
        px = px.set_index("date").adj_close
        due = b.date + pd.Timedelta(days=90)
        exp_exit = px.index[px.index >= due][0] if (px.index >= due).any() else pd.NaT
        units = (b.notional + b.fee) / (px[b.date] * (1 + C.FEE_BPS / 1e4))
        row = dict(lot_id=b.lot_id, symbol=b.symbol, signal=b.signal_date.date(),
                   fill=b.date.date(), db_entry_px=px[b.date], bt_entry_px=b.price,
                   budget=b.notional + b.fee, units_by_hand=units, bt_units=b.units,
                   due=due.date(), expected_exit=getattr(exp_exit, "date", lambda: None)())
        if len(s):
            s = s.iloc[0]
            gross = s.units * px[s.date]
            row.update(bt_exit=s.date.date(), db_exit_px=px[s.date], bt_exit_px=s.price,
                       cash_back_by_hand=gross * (1 - C.FEE_BPS / 1e4), bt_cash_back=s.cash_change)
        rows.append(row)
    return pd.DataFrame(rows)


def plot(strategy_nav, bench_nav, path):
    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for nav, color, label in [(strategy_nav, blue, "Strategy"), (bench_nav, orange, "S&P 500 TR")]:
        cum = nav / nav.iloc[0] - 1
        ax.plot(cum.index, cum * 100, color=color, lw=2, label=label)
        ax.annotate(f"{label} {cum.iloc[-1]:+.0%}", (cum.index[-1], cum.iloc[-1] * 100),
                    xytext=(6, 0), textcoords="offset points", va="center", color=ink, fontsize=10)
    ax.axhline(0, color=muted, lw=0.8)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(muted)
    ax.tick_params(colors=muted)
    ax.set_ylabel("Cumulative return (%)", color=muted)
    ax.set_title("Cumulative return, $1M invested at the 2016-04-01 close (through 2026-09-30)",
                 color=ink, loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper left", labelcolor=ink)
    ax.margins(x=0.01)
    fig.subplots_adjust(right=0.83)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(db=DEFAULT_DB):
    RESULTS.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    t = time.time()
    m = load_market(db, end=C.TEST_END)
    panel = build_panel(m, load_fundamental_events(db, end=C.TEST_END), "2010-01-01", C.TEST_END)
    print(f"panel {panel.shape} in {time.time() - t:.0f}s")

    fits = refit_dates(m["calendar"])
    pred, log = walk_forward(panel, fits, C.TEST_END, C.MODEL)
    log.to_csv(RESULTS / "refits.csv", index=False)
    print("refits:\n", log.to_string(index=False))
    pred = pred[pred.date >= pd.Timestamp(C.FIRST_SIGNAL)]
    prediction_report(pred, max_rank=C.MAX_UNIVERSE_RANK).to_csv(RESULTS / "test_prediction_report.csv")

    prices = m["A"].loc[:C.TEST_END]
    strat = TopScoreStrategy(pred, top_n=C.TOP_N, tranches=C.TRANCHES, max_weight=C.MAX_WEIGHT,
                             max_sector_share=C.MAX_SECTOR_SHARE, min_dollar_vol=C.MIN_DOLLAR_VOL,
                             max_universe_rank=C.MAX_UNIVERSE_RANK)
    bt = Backtester(prices, initial_cash=C.INITIAL_CASH, fee_bps=C.FEE_BPS)
    res = bt.run(strat, C.FIRST_SIGNAL, C.TEST_END)
    assert res.fills.date.min() == pd.Timestamp(C.TEST_START)

    # Benchmark: cash at the 2016-03-31 close, all-in at the 2016-04-01 close.
    b = m["bench"].loc[C.FIRST_SIGNAL:C.TEST_END]
    bench_nav = C.INITIAL_CASH * b / b.loc[C.TEST_START]
    bench_nav.loc[C.FIRST_SIGNAL] = C.INITIAL_CASH
    nav = res.history.nav
    assert nav.index.equals(bench_nav.index)

    metrics = pd.DataFrame({"Strategy": performance_metrics(nav),
                            "S&P 500 TR": performance_metrics(bench_nav)})
    metrics.loc["final_value_if_stale_positions_worth_zero", "Strategy"] = res.history.nav_ex_stale.iloc[-1]
    metrics.to_csv(RESULTS / "metrics.csv")

    years = pd.DataFrame({"Strategy": nav, "S&P 500 TR": bench_nav}).resample("YE").last()
    years = years.pct_change().fillna(years.iloc[0] / C.INITIAL_CASH - 1)
    years.index = years.index.year
    years["excess"] = years.Strategy - years["S&P 500 TR"]
    years.to_csv(RESULTS / "yearly_returns.csv")

    checks = check_accounting(res, prices)
    checks["fills"] = len(res.fills)
    checks["buys"] = int((res.fills.side == "BUY").sum())
    checks["cancelled_orders"] = len(res.cancelled)
    checks["total_fees"] = res.fills.fee.sum()
    checks["max_stale_value"] = res.history.stale_value.max()
    checks["open_lots_at_end"] = len(res.open_lots)
    checks["open_lots_not_quoted_on_last_day"] = int((res.open_lots.last_quote < nav.index[-1]).sum())
    checks["days_cash_above_5pct"] = int((res.history.cash / nav > 0.05).sum())
    checks["mean_cash_weight"] = (res.history.cash / nav).mean()
    checks.to_csv(RESULTS / "accounting_checks.csv", header=["value"])
    hand = hand_check_lots(db, res.fills)
    hand.to_csv(RESULTS / "hand_checked_lots.csv", index=False)

    res.history.to_csv(OUT / "daily_history.csv")
    res.fills.to_csv(OUT / "fills.csv", index=False)
    res.cancelled.to_csv(OUT / "cancelled.csv", index=False)
    res.open_lots.to_csv(RESULTS / "open_lots_at_end.csv", index=False)
    plot(nav, bench_nav, RESULTS / "cumulative_returns.png")

    pd.set_option("display.width", 200)
    print("\nmetrics:\n", metrics.to_string())
    print("\nyearly:\n", years.round(4).to_string())
    print("\naccounting:\n", checks.to_string())
    print("\nhand-checked lots:\n", hand.T.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DB)
