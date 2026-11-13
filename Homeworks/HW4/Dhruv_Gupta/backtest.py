"""HW4 section 3: long-only ledger for the frozen 90-day scores.

Orders are dollar budgets chosen at the previous close. Fills use the next
session's adjusted close. Each purchase is its own lot. Cash interest is zero.
A 5 bp fee is reserved inside each buy budget and charged again on the sale.

The fee-inside-budget idea follows Lectures/modeling/backtest.py. This file
does not call that backtester. The monthly cohort limit is the allocation
chosen here so three 90-day books can be funded without reusing the same cash.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

INITIAL_CASH = 1_000_000.0
HOLDING_DAYS = 90
FEE_BPS = 5.0
TOP_N = 30
COHORT_FRACTION = 1.0 / 3.0
NAME_CAP = 0.04
MAX_STALE_SESSIONS = 5


@dataclass
class Lot:
    lot_id: int
    symbol: str
    units: float
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp


class Ledger:
    """Cash, open lots, and the fill log. Long only."""

    def __init__(self, cash=INITIAL_CASH, fee_bps=FEE_BPS):
        if not np.isfinite(cash) or cash <= 0:
            raise ValueError("Starting capital must be positive")
        if not np.isfinite(fee_bps) or fee_bps < 0:
            raise ValueError("The fee must be nonnegative")
        self.initial_cash = float(cash)
        self.cash = float(cash)
        self.fee_rate = float(fee_bps) / 10_000.0
        self.lots = {}
        self.fills = []
        self._next_id = 1

    def units_for_budget(self, budget, price):
        """Shares bought when `budget` must also pay the entry fee."""
        if budget < 0 or price <= 0 or not np.isfinite([budget, price]).all():
            raise ValueError("Budget must be nonnegative and price positive")
        return budget / (price * (1.0 + self.fee_rate))

    def buy(self, budget, price, symbol, date, signal_date):
        date = pd.Timestamp(date)
        signal_date = pd.Timestamp(signal_date)
        if signal_date >= date:
            raise ValueError("The signal close must precede the fill")
        units = self.units_for_budget(budget, price)
        if units <= 0:
            return None
        notional = units * price
        fee = notional * self.fee_rate
        if notional + fee > self.cash + 1e-8:
            raise ValueError(f"Buy of {symbol} needs {notional + fee:.2f} but cash is {self.cash:.2f}")
        lot = Lot(
            self._next_id, symbol, units, signal_date, date, float(price), fee,
            date + pd.Timedelta(days=HOLDING_DAYS),
        )
        self._next_id += 1
        self.cash -= notional + fee
        if abs(self.cash) < 1e-8:
            self.cash = 0.0
        self.lots[lot.lot_id] = lot
        self.fills.append(self._fill(lot, "buy", date, units, price, fee, -(notional + fee)))
        return lot.lot_id

    def sell(self, lot_id, price, date):
        lot = self.lots[lot_id]
        date = pd.Timestamp(date)
        if date < lot.entry_date:
            raise ValueError("A sale cannot precede its purchase")
        notional = lot.units * price
        fee = notional * self.fee_rate
        cash_change = notional - fee
        self.cash += cash_change
        self.fills.append(self._fill(lot, "sell", date, lot.units, price, fee, cash_change))
        del self.lots[lot_id]
        return cash_change

    def holdings_value(self, marks):
        total = 0.0
        by_symbol = {}
        for lot in self.lots.values():
            if lot.symbol not in marks:
                raise ValueError(f"No mark for open lot {lot.symbol}")
            value = lot.units * marks[lot.symbol]
            total += value
            by_symbol[lot.symbol] = by_symbol.get(lot.symbol, 0.0) + value
        return total, by_symbol

    def _fill(self, lot, side, date, units, price, fee, cash_change):
        return {
            "date": pd.Timestamp(date),
            "signal_date": lot.signal_date,
            "lot_id": lot.lot_id,
            "symbol": lot.symbol,
            "side": side,
            "price": float(price),
            "units": float(units),
            "notional": float(units * price),
            "fee": float(fee),
            "cash_change": float(cash_change),
            "due_date": lot.due_date,
        }


def allocate_cohort(scores, cash, nav, held_symbols, top_n=TOP_N, cohort_fraction=COHORT_FRACTION,
                    name_cap=NAME_CAP, fee_rate=FEE_BPS / 10_000.0):
    """Equal fee-inclusive budgets for the highest scores, capped by cash and NAV.

    Names already held are skipped. Any budget the name cap does not use stays
    in cash. `scores` is a Series indexed by symbol.
    """
    if cash <= 1e-8 or nav <= 0 or scores is None or len(scores) == 0:
        return pd.Series(dtype=float)
    ranked = scores.rename("score").rename_axis("symbol").reset_index()
    ranked = ranked.loc[~ranked["symbol"].isin(held_symbols) & np.isfinite(ranked["score"])]
    ranked = ranked.sort_values(["score", "symbol"], ascending=[False, True]).head(top_n)
    budget = min(float(cash), float(nav) * cohort_fraction)
    if ranked.empty or budget <= 1e-8:
        return pd.Series(dtype=float)
    per_name = budget / len(ranked)
    max_spend = name_cap * float(nav) * (1.0 + fee_rate)
    chosen = {}
    remaining = budget
    for symbol in ranked["symbol"]:
        spend = min(per_name, remaining, max_spend)
        if spend > 1e-8:
            chosen[symbol] = float(spend)
            remaining -= spend
    out = pd.Series(chosen, dtype=float)
    if out.sum() > cash + 1e-8:
        raise ValueError("Cohort budgets exceed available cash")
    return out


def _as_index(values):
    return pd.DatetimeIndex(pd.to_datetime(values)).tz_localize(None)


def run_backtest(quotes, calendar, scores_by_date, entry_dates, start, end,
                 initial_cash=INITIAL_CASH, fee_bps=FEE_BPS, max_stale_sessions=MAX_STALE_SESSIONS,
                 top_n=TOP_N, cohort_fraction=COHORT_FRACTION, name_cap=NAME_CAP):
    """Mark every session from `start` through `end`.

    `quotes` is a DataFrame of adjusted closes indexed by session, with one
    column per symbol. Missing quotes stay missing. `scores_by_date` maps a
    signal session to a Series of model scores. Buys are placed only on
    `entry_dates`, and each budget was computable from the prior close.
    """
    calendar = _as_index(calendar).sort_values().unique()
    quotes = quotes.copy()
    quotes.index = _as_index(quotes.index)
    if quotes.index.has_duplicates:
        raise ValueError("Quote dates must be unique")
    quotes = quotes.sort_index().reindex(calendar)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start not in calendar or end not in calendar:
        raise ValueError("Start and end must be market sessions")
    if calendar.get_loc(start) == 0:
        raise ValueError("The first entry needs a prior signal session")
    entry_dates = set(_as_index(entry_dates))
    ledger = Ledger(initial_cash, fee_bps)
    last_marks = {}
    last_seen = {}
    diagnostics = []
    history = []
    noted_stale = set()
    noted_waiting = set()
    days = calendar[(calendar >= start) & (calendar <= end)]

    def record(day, holdings_value, stale_marks):
        history.append({
            "date": day,
            "cash": ledger.cash,
            "holdings_value": holdings_value,
            "nav": ledger.cash + holdings_value,
            "open_lots": len(ledger.lots),
            "stale_marks": stale_marks,
        })

    prior = calendar[calendar.get_loc(start) - 1]
    record(prior, 0.0, 0)

    for day in days:
        day_index = int(calendar.get_loc(day))
        signal_date = calendar[day_index - 1]
        holdings_value, by_symbol = _value(ledger, last_marks)
        orders = pd.Series(dtype=float)
        if day in entry_dates:
            orders = allocate_cohort(
                scores_by_date.get(pd.Timestamp(signal_date), pd.Series(dtype=float)),
                ledger.cash, ledger.cash + holdings_value, set(by_symbol),
                top_n=top_n, cohort_fraction=cohort_fraction, name_cap=name_cap,
                fee_rate=ledger.fee_rate,
            )
            if float(orders.sum()) > ledger.cash + 1e-8:
                raise ValueError("Orders planned at the signal close exceed cash")

        observed = quotes.loc[day].dropna() if day in quotes.index else pd.Series(dtype=float)
        observed = observed[np.isfinite(observed) & observed.gt(0)]
        for symbol, price in observed.items():
            last_marks[symbol] = float(price)
            last_seen[symbol] = day_index

        stale_marks = 0
        for symbol in sorted({lot.symbol for lot in ledger.lots.values()}):
            gap = day_index - last_seen[symbol]
            if gap > 0:
                stale_marks += 1
            if gap > max_stale_sessions and symbol not in noted_stale:
                noted_stale.add(symbol)
                diagnostics.append({"date": day, "symbol": symbol, "event": "stale_mark"})

        for lot in list(ledger.lots.values()):
            if day >= lot.due_date and lot.symbol in observed.index:
                ledger.sell(lot.lot_id, float(observed[lot.symbol]), day)
            elif day >= lot.due_date and lot.symbol not in observed.index and lot.lot_id not in noted_waiting:
                noted_waiting.add(lot.lot_id)
                diagnostics.append({"date": day, "symbol": lot.symbol, "event": "exit_waiting_for_quote"})

        for symbol, budget in orders.items():
            if symbol not in observed.index:
                diagnostics.append({"date": day, "symbol": symbol, "event": "rejected_missing_quote"})
                continue
            ledger.buy(float(budget), float(observed[symbol]), symbol, day, signal_date)

        holdings_value, _ = _value(ledger, last_marks)
        record(day, holdings_value, stale_marks)

    history = pd.DataFrame(history).set_index("date")
    fills = pd.DataFrame(ledger.fills)
    open_lots = pd.DataFrame([asdict(lot) for lot in ledger.lots.values()])
    diagnostics = pd.DataFrame(diagnostics, columns=["date", "symbol", "event"])
    check_accounting(history, fills, open_lots, ledger.initial_cash)
    return history, fills, open_lots, diagnostics


def _value(ledger, marks):
    if not ledger.lots:
        return 0.0, {}
    return ledger.holdings_value(marks)


def check_accounting(history, fills, open_lots, initial_cash):
    """Cash is the sum of fills, and NAV is cash plus the recorded holdings."""
    if abs(history["nav"].iloc[0] - initial_cash) > 1e-6:
        raise AssertionError("The book does not start at the stated capital")
    if (history["cash"] < -1e-6).any():
        raise AssertionError("Cash went negative")
    if not np.allclose(history["nav"], history["cash"] + history["holdings_value"]):
        raise AssertionError("NAV is not cash plus holdings")
    cash_from_fills = initial_cash
    if len(fills):
        cash_from_fills += fills["cash_change"].sum()
    if abs(cash_from_fills - history["cash"].iloc[-1]) > 1e-6:
        raise AssertionError("Ending cash does not match the fill log")
    if len(fills):
        buys = fills.loc[fills["side"].eq("buy")]
        if (buys["signal_date"] >= buys["date"]).any():
            raise AssertionError("A buy was filled at or before its signal")
        if (buys["due_date"] - buys["date"] != pd.Timedelta(days=HOLDING_DAYS)).any():
            raise AssertionError("A holding period is not 90 calendar days")
        sold = set(fills.loc[fills["side"].eq("sell"), "lot_id"])
        bought = set(buys["lot_id"])
        if not sold <= bought:
            raise AssertionError("A sale has no purchase")
        if len(open_lots) and set(open_lots["lot_id"]) & sold:
            raise AssertionError("A sold lot is still open")
    if (history.index > history.index.max()).any():
        raise AssertionError("History goes past the sample end")


def accounting_evidence(history, fills, open_lots, initial_cash=INITIAL_CASH):
    """Record the identities a reader can check without rerunning the book."""
    check_accounting(history, fills, open_lots, initial_cash)
    nav_gap = float((history["nav"] - history["cash"] - history["holdings_value"]).abs().max())
    cash_gap = float(initial_cash + fills["cash_change"].sum() - history["cash"].iloc[-1])
    buys = fills.loc[fills["side"].eq("buy")]
    sells = fills.loc[fills["side"].eq("sell")]
    rows = [
        ("starts_at_capital", history["nav"].iloc[0], "March 31, 2016 mark is the starting cash"),
        ("cash_minimum", float(history["cash"].min()), "Never negative"),
        ("nav_minus_cash_minus_holdings", nav_gap, "Largest absolute break in NAV = cash + holdings"),
        ("ending_cash_minus_fill_sum", cash_gap, "Ending cash equals capital plus every fill"),
        ("buys", len(buys), "Each buy is signaled on an earlier session"),
        ("sales", len(sells), "Each sale matches a buy, and that lot is not still open"),
        ("open_lots", len(open_lots), "Buys minus sales"),
        ("earliest_signal", str(pd.Timestamp(buys["signal_date"].min()).date()), "Before the first fill"),
        ("first_fill", str(pd.Timestamp(buys["date"].min()).date()), "First test close"),
        ("shortest_hold_days", int((buys["due_date"] - buys["date"]).dt.days.min()), "Due date is entry plus 90 calendar days"),
        ("longest_hold_days", int((buys["due_date"] - buys["date"]).dt.days.max()), "Due date is entry plus 90 calendar days"),
        ("last_history_date", str(pd.Timestamp(history.index.max()).date()), "No price after the sample end"),
    ]
    return pd.DataFrame(rows, columns=["check", "value", "note"])


def write_accounting_evidence(history, fills, open_lots):
    import model

    evidence = accounting_evidence(history, fills, open_lots)
    model.RESULTS.mkdir(parents=True, exist_ok=True)
    evidence.to_csv(model.RESULTS / "accounting_check.csv", index=False)
    return evidence


def load_saved_book():
    import model

    history = pd.read_csv(model.RESULTS / "backtest_history.csv", parse_dates=["date"]).set_index("date")
    fills = pd.read_csv(
        model.RESULTS / "backtest_fills.csv",
        parse_dates=["date", "signal_date", "due_date"],
    )
    open_lots = pd.read_csv(
        model.RESULTS / "backtest_open_lots.csv",
        parse_dates=["signal_date", "entry_date", "due_date"],
    )
    return history, fills, open_lots


def run_frozen_portfolio():
    """Score each month-end with the frozen model and run the test-period book."""
    import json

    import duckdb

    import model

    model.log("Fitting the frozen April 2016 model")
    choice = json.loads(model.CHOICE_PATH.read_text())
    boundary = model.first_on_or_after(model.TEST_START)
    train_raw = model.read_panel(model.TRAIN_START, boundary - pd.Timedelta(days=1), month_start_only=True)
    train = model.matured_training_rows(model.add_peer_features(train_raw), boundary)
    pipe, _, _ = model.fit_regressor(model.make_tree(choice["params"]), train)
    if train["label_end"].max() >= boundary:
        raise ValueError("The portfolio model saw a label that had not ended")

    with duckdb.connect(str(model.FULL_DB), read_only=True) as con:
        benchmark = con.execute(
            "SELECT CAST(date AS TIMESTAMP) AS date, adj_close FROM benchmark WHERE date <= ? ORDER BY date",
            [model.TEST_END],
        ).fetchdf()
    calendar = pd.DatetimeIndex(pd.to_datetime(benchmark["date"]))
    entries = month_entry_dates(calendar, model.TEST_START, model.TEST_END)
    signals = pd.DatetimeIndex([calendar[calendar.get_loc(day) - 1] for day in entries])
    model.log(f"Scoring {len(signals)} month-end signals for {len(entries)} entries")

    con = duckdb.connect()
    raw = con.execute(
        "SELECT * FROM read_parquet(?) WHERE date IN (SELECT UNNEST(?))",
        [str(model.PANEL_PATH), [pd.Timestamp(day) for day in signals]],
    ).fetchdf()
    con.close()
    scored = model.add_peer_features(model._normalize(raw))
    scored["score"] = choice["score_sign"] * pipe.predict(scored[model.MODEL_COLUMNS])
    scores_by_date = {
        pd.Timestamp(date): group.set_index("symbol")["score"]
        for date, group in scored.groupby("date", sort=False)
    }

    symbols = sorted(scored["symbol"].unique())
    model.log(f"Loading adjusted closes for {len(symbols):,} names")
    with duckdb.connect(str(model.FULL_DB), read_only=True) as con:
        prices = con.execute(
            """
            SELECT symbol, CAST(date AS TIMESTAMP) AS date, adj_close
            FROM prices
            WHERE date >= ? AND date <= ?
              AND adj_close > 0 AND isfinite(adj_close)
              AND symbol IN (SELECT UNNEST(?))
            """,
            [signals.min(), model.TEST_END, symbols],
        ).fetchdf()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.loc[prices["date"].isin(calendar)]
    if prices.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate adjusted closes")
    quotes = prices.pivot(index="date", columns="symbol", values="adj_close")

    model.log("Running the ledger")
    history, fills, open_lots, diagnostics = run_backtest(
        quotes, calendar, scores_by_date, entries, model.TEST_START, model.TEST_END,
    )
    model.RESULTS.mkdir(parents=True, exist_ok=True)
    history.to_csv(model.RESULTS / "backtest_history.csv")
    fills.to_csv(model.RESULTS / "backtest_fills.csv", index=False)
    open_lots.to_csv(model.RESULTS / "backtest_open_lots.csv", index=False)
    diagnostics.to_csv(model.RESULTS / "backtest_diagnostics.csv", index=False)
    fees = float(fills["fee"].sum()) if len(fills) else 0.0
    summary = pd.Series({
        "final_nav": float(history["nav"].iloc[-1]),
        "final_cash": float(history["cash"].iloc[-1]),
        "final_holdings": float(history["holdings_value"].iloc[-1]),
        "open_lots": int(len(open_lots)),
        "buys": int(fills["side"].eq("buy").sum()) if len(fills) else 0,
        "sells": int(fills["side"].eq("sell").sum()) if len(fills) else 0,
        "fees": fees,
        "stale_names": int(diagnostics["event"].eq("stale_mark").sum()) if len(diagnostics) else 0,
        "rejected_missing_quotes": int(diagnostics["event"].eq("rejected_missing_quote").sum()) if len(diagnostics) else 0,
        "exits_waiting": int(diagnostics["event"].eq("exit_waiting_for_quote").sum()) if len(diagnostics) else 0,
        "first_entry": str(pd.Timestamp(entries[0]).date()),
        "last_date": str(pd.Timestamp(history.index[-1]).date()),
    })
    summary.to_csv(model.RESULTS / "backtest_summary.csv", header=["value"])
    write_accounting_evidence(history, fills, open_lots)
    _plot_nav(history, model.RESULTS / "backtest_nav.png")
    compare_to_benchmark(history)
    model.log(
        f"Final NAV {summary['final_nav']:.2f}; cash {summary['final_cash']:.2f}; "
        f"open lots {summary['open_lots']}; fees {summary['fees']:.2f}"
    )
    return history, summary


def performance(nav, annual_rf=0.02):
    """CAGR on calendar time, drawdown on the level, Sharpe on daily simple returns.

    The 2% hurdle is converted to a daily rate with 252 sessions, matching the
    convention in Lectures/modeling/backtest.py. CAGR uses 365.25-day years.
    """
    nav = nav.sort_index().astype(float)
    if len(nav) < 2 or not np.isfinite(nav).all() or nav.le(0).any():
        raise ValueError("Need at least two positive portfolio values")
    returns = nav.pct_change().dropna()
    daily_rf = (1.0 + annual_rf) ** (1.0 / 252.0) - 1.0
    excess = returns - daily_rf
    volatility = float(returns.std(ddof=1))
    years = (nav.index[-1] - nav.index[0]).total_seconds() / (365.25 * 24 * 60 * 60)
    sharpe = np.sqrt(252) * float(excess.mean()) / volatility if volatility > 1e-12 else np.nan
    return pd.Series({
        "final_value": float(nav.iloc[-1]),
        "CAGR": float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0),
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
        "Sharpe": float(sharpe),
    })


def compare_to_benchmark(history=None, annual_rf=0.02):
    """Buy and hold ^SP500TR from the first test close through the same end date."""
    import duckdb

    import model

    if history is None:
        history = pd.read_csv(model.RESULTS / "backtest_history.csv", parse_dates=["date"]).set_index("date")
    history.index = pd.to_datetime(history.index)
    first_close = pd.Timestamp(history.index[1])
    last_close = pd.Timestamp(history.index[-1])
    with duckdb.connect(str(model.FULL_DB), read_only=True) as con:
        benchmark = con.execute(
            """
            SELECT CAST(date AS TIMESTAMP) AS date, adj_close
            FROM benchmark
            WHERE date >= ? AND date <= ?
            ORDER BY date
            """,
            [first_close, last_close],
        ).fetchdf()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark = benchmark.set_index("date")["adj_close"].reindex(history.index[1:])
    if benchmark.isna().any() or not (benchmark > 0).all():
        raise ValueError("The benchmark is missing a test-session close")
    benchmark_nav = pd.Series(INITIAL_CASH, index=history.index, dtype=float)
    benchmark_nav.iloc[1:] = INITIAL_CASH * benchmark / benchmark.iloc[0]
    strategy = performance(history["nav"], annual_rf)
    index = performance(benchmark_nav, annual_rf)
    comparison = pd.DataFrame({"strategy": strategy, "^SP500TR": index})
    gap = float(strategy["final_value"] - index["final_value"])
    comparison.loc["final_value_gap"] = [gap, 0.0]
    model.RESULTS.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(model.RESULTS / "benchmark_comparison.csv")
    curves = pd.DataFrame({
        "strategy": history["nav"] / INITIAL_CASH - 1.0,
        "^SP500TR": benchmark_nav / INITIAL_CASH - 1.0,
    })
    curves.to_csv(model.RESULTS / "cumulative_returns.csv")
    _plot_cumulative(curves, model.RESULTS / "cumulative_returns.png")
    model.log(
        f"Strategy {strategy['final_value']:.0f} vs ^SP500TR {index['final_value']:.0f} "
        f"(gap {gap:.0f})"
    )
    return comparison


def _plot_cumulative(curves, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(curves.index, curves["strategy"] * 100, color="#1f4e79", label="Strategy")
    ax.plot(curves.index, curves["^SP500TR"] * 100, color="#6b4c9a", label="S&P 500 total return")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("Growth of $1,000,000 from the first test close")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_nav(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(history.index, history["nav"], color="#1f4e79", label="Portfolio value")
    ax.plot(history.index, history["cash"], color="#c47b17", linewidth=1, label="Cash")
    ax.set_ylabel("Dollars")
    ax.set_title("Long-only book, including open positions")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def month_entry_dates(calendar, start, end):
    """First market session of each month inside the test window."""
    calendar = _as_index(calendar)
    days = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    return pd.DatetimeIndex(days.to_series().groupby(days.to_period("M")).min())


if __name__ == "__main__":
    import sys

    if "--compare" in sys.argv:
        compare_to_benchmark()
    elif "--evidence" in sys.argv:
        history, fills, open_lots = load_saved_book()
        print(write_accounting_evidence(history, fills, open_lots).to_string(index=False))
    else:
        import test_backtest

        test_backtest.main()
        run_frozen_portfolio()
