"""Independently verify the saved full-run ledger against the raw database."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from strategy.data import ROOT

HERE = Path(__file__).resolve().parent


def main():
    config = json.loads((HERE / "results/locked_config.json").read_text())
    folder = HERE / "outputs/results"
    daily = pd.read_csv(folder / "daily.csv", parse_dates=["date"]).set_index("date")
    trades = pd.read_csv(folder / "trades.csv", parse_dates=["date", "signal_date", "entry", "maturity"])
    lots = pd.read_csv(folder / "open_lots.csv")
    buys = trades.loc[trades.side.eq("buy")].copy()
    sells = trades.loc[trades.side.eq("sell")].copy()
    with duckdb.connect(str(ROOT / config["database"]), read_only=True) as con:
        con.register("ledger", trades)
        checked = con.execute("""SELECT l.*,p.adj_close AS raw_price FROM ledger l
            LEFT JOIN prices p ON p.symbol=l.symbol AND p.date=l.date"""
        ).df()
        assert np.allclose(checked.price, checked.raw_price)
        calendar = pd.DatetimeIndex(pd.to_datetime(con.execute("SELECT date FROM calendar ORDER BY date").df().date))
        assert np.array_equal(calendar.get_indexer(buys.date), calendar.get_indexer(buys.signal_date)+1)
        con.register("buys", buys)
        count = con.execute("""SELECT count(*) FROM buys b JOIN universe_membership u
            ON u.symbol=b.symbol AND u.quarter_start=date_trunc('quarter',b.date)
            WHERE u.selection_date<=b.signal_date"""
        ).fetchone()[0]
        assert count == len(buys)
        con.register("sales", sells)
        # An earlier valid market quote after maturity would require an earlier exit.
        earlier = con.execute("""SELECT count(*) FROM sales s JOIN prices p ON p.symbol=s.symbol
            JOIN calendar c ON c.date=p.date
            WHERE p.date>=s.maturity AND p.date<s.date AND p.adj_close>0 AND isfinite(p.adj_close)"""
        ).fetchone()[0]
        assert earlier == 0
    assert sells.date.ge(sells.maturity).all()
    assert (trades.maturity-trades.entry).dt.days.eq(90).all()
    cash_flows = trades.groupby("date").cash_flow.sum().reindex(daily.index, fill_value=0).cumsum()
    assert np.allclose(config["capital"]+cash_flows, daily.cash, atol=1e-5)
    assert daily.cash.ge(-1e-6).all()
    assert np.allclose(daily.cash+daily.holdings, daily.value)
    assert np.isclose(daily.value.iloc[-1], daily.cash.iloc[-1]+(lots.units*lots.mark).sum())
    gross = trades.units*trades.price
    assert np.allclose(trades.fee, gross*config["fee"])
    expected_flow = np.where(trades.side.eq("buy"), -gross-trades.fee, gross-trades.fee)
    assert np.allclose(trades.cash_flow, expected_flow)
    assert buys.date.min() == pd.Timestamp("2016-04-01")
    assert trades.date.max() <= pd.Timestamp("2026-09-30")
    lines = ["# Full-run accounting audit", "", f"PASS: {len(buys):,} buys and {len(sells):,} sales checked.", "",
             "- Every fill matches the raw database adjusted close.",
             "- Every purchase uses the preceding market session's signal.",
             "- Every purchase has entry-quarter membership available at the signal close.",
             "- Every lot matures exactly 90 calendar days after entry.",
             "- Every sale is the first valid quoted market close on or after maturity.",
             "- Daily cash reconciles with the complete transaction ledger; cash never goes negative.",
             "- Fees and cash flows match units times fill price independently.",
             "- Daily and final NAV reconcile to cash plus holdings.",
             "- First buy is April 1, 2016; no trade uses a post-test date.", "",
             "These checks verify accounting and timing, not the economic accuracy of source prices or stale marks."]
    (HERE / "results/ACCOUNTING_CHECKS.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
