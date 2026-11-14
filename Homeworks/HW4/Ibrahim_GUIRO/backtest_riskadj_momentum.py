from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


DATABASE = Path("Lectures/modeling/data/equities_top2000.duckdb")
CAPITAL = 1_000_000.0
FEE = 0.001        # 10 basis points per purchase and per sale
FINAL_START = pd.Timestamp("2016-04-01")
FINAL_END = pd.Timestamp("2026-09-30")
OUTPUT = Path(__file__).resolve().parent / "results_riskadj_momentum"
MAX_RANK = 500
TOP_N = 30
SLEEVES = 3
SLEEVE_GAP = 21
MIN_ORDER = 2_000.0
STALE_SESSIONS = 5
PERIODS = {
    "development": (pd.Timestamp("2010-01-04"), pd.Timestamp("2015-12-31")),
    "final": (FINAL_START, FINAL_END),
}


def load_benchmark(db, index):
    """$1,000,000 in ^SP500TR at the first session of `index`."""
    rows = db.execute(
        "SELECT date, adj_close FROM benchmark WHERE symbol = '^SP500TR'"
    ).df()
    rows["date"] = pd.to_datetime(rows["date"])
    prices = rows.set_index("date")["adj_close"].reindex(index)
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("The benchmark has an invalid or missing session price.")
    return CAPITAL * prices / prices.iloc[0]


def metrics(equity):
    """Final value, CAGR, max drawdown and Sharpe (2% annual risk-free rate)."""
    wealth = np.r_[CAPITAL, equity.to_numpy()]
    changes = wealth[1:] / wealth[:-1] - 1
    excess = changes - (1.02 ** (1 / 252) - 1)
    elapsed = (equity.index[-1] - equity.index[0]).days / 365.25
    daily_risk = excess.std(ddof=1)
    return {
        "final_value": wealth[-1],
        "total_return": wealth[-1] / CAPITAL - 1,
        "cagr": (wealth[-1] / CAPITAL) ** (1 / elapsed) - 1,
        "max_drawdown": (wealth / np.maximum.accumulate(wealth) - 1).min(),
        "sharpe": excess.mean() / daily_risk * np.sqrt(252) if daily_risk > 0 else np.nan,
    }


def load_inputs(db):
    calendar = pd.DatetimeIndex(db.execute(
        "SELECT date FROM calendar WHERE date >= DATE '2008-01-01' ORDER BY date"
    ).df()["date"])
    rows = db.execute(
        """
        SELECT date, symbol, adj_close FROM prices
        WHERE date >= DATE '2008-01-01' AND isfinite(adj_close) AND adj_close > 0
        """
    ).df()
    rows["date"] = pd.to_datetime(rows["date"])
    # Off-calendar rows are dropped by the reindex, as the data guide requires.
    prices = rows.pivot(index="date", columns="symbol", values="adj_close")
    prices = prices.reindex(calendar)

    # Warrants, notes, units and SPACs can enter the top 500 because the cap
    # proxy multiplies their price by the issuer's share count. Keep one row per
    # issuer name (the best-ranked one) and drop debt-like and SPAC listings.
    members = db.execute(
        r"""
        SELECT m.quarter_start, m.symbol
        FROM universe_membership AS m JOIN universe AS u USING (symbol)
        WHERE m.universe_rank <= ?
          AND NOT regexp_matches(u.name,
              '(?i)\bnotes?\b|debenture|warrant|acquisition corp|%')
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY m.quarter_start, lower(trim(u.name))
            ORDER BY m.universe_rank
        ) = 1
        """,
        [MAX_RANK],
    ).df()
    members["quarter_start"] = pd.to_datetime(members["quarter_start"])
    members = {
        quarter: set(group["symbol"])
        for quarter, group in members.groupby("quarter_start")
    }
    return calendar, prices, members


def make_scores(prices):
    """Scores known after each close; no value uses a later price."""
    daily = prices.pct_change(fill_method=None)
    momentum = prices.shift(21) / prices.shift(252) - 1
    volatility = daily.rolling(63, min_periods=60).std()
    usable = (
        prices.notna()
        & (prices.notna().rolling(252).sum() >= 240)
        & (daily.abs().gt(0.5).rolling(252, min_periods=1).sum() == 0)
        & (volatility > 0)
    )
    return (momentum / volatility).where(usable)


def pick(scores, members, signal_day, entry_day):
    quarter = entry_day.to_period("Q").start_time
    row = scores.loc[signal_day]
    row = row[row.index.isin(members.get(quarter, set()))].dropna()
    # Sort by score, then symbol, so ties do not depend on column order.
    ranked = row.reset_index().sort_values(
        [signal_day, "symbol"], ascending=[False, True]
    )
    return ranked["symbol"].head(TOP_N).tolist()


def simulate(calendar, prices, scores, members, start, end):
    sessions = calendar[(calendar >= start) & (calendar <= end)]
    quotes = prices.loc[sessions]
    sleeve_cash = [CAPITAL / SLEEVES] * SLEEVES
    sleeve_start = [sessions[k * SLEEVE_GAP] for k in range(SLEEVES)]
    lots = []          # dicts: sleeve, symbol, units, entry, maturity, cost
    pending = {}       # sleeve -> (names, budget) to fill at the next close
    marks = {}         # symbol -> (price, session index)
    fills, history, incidents = [], [], []

    for i, day in enumerate(sessions):
        row = quotes.iloc[i]

        # 1. Sell matured lots at today's close, or the next observed close.
        still_open = []
        for lot in lots:
            price = row[lot["symbol"]]
            if day >= lot["maturity"] and pd.notna(price):
                if (day - lot["entry"]).days < 90:
                    raise RuntimeError(f"{lot['symbol']} sold before 90 days.")
                proceeds = lot["units"] * price * (1 - FEE)
                sleeve_cash[lot["sleeve"]] += proceeds
                fills.append((day, lot["sleeve"], lot["symbol"], "SELL",
                              lot["units"], price, proceeds))
            else:
                if day == lot["maturity"]:
                    incidents.append((day, lot["symbol"], "missing_exit_quote"))
                still_open.append(lot)
        lots = still_open

        # 2. Fill orders planned at yesterday's close. Budgets include fees.
        for sleeve, (names, budget) in pending.items():
            per_name = budget / len(names)
            maturity_idx = calendar.searchsorted(day + pd.Timedelta(days=90))
            # NaT means the maturity falls after the last supplied session.
            maturity = (calendar[maturity_idx] if maturity_idx < len(calendar)
                        else pd.NaT)
            for symbol in names:
                price = row[symbol]
                if pd.isna(price):
                    incidents.append((day, symbol, "missing_entry_quote"))
                    continue  # the unspent cash stays in the sleeve
                units = per_name / (price * (1 + FEE))
                sleeve_cash[sleeve] -= per_name
                lots.append(dict(sleeve=sleeve, symbol=symbol, units=units,
                                 entry=day, maturity=maturity, cost=per_name))
                fills.append((day, sleeve, symbol, "BUY", units, price, -per_name))
        pending = {}
        if min(sleeve_cash) < -1e-6:
            raise RuntimeError(f"Negative cash on {day.date()}.")

        # 3. Mark open lots; carry the last quote and flag stale marks.
        value = stale = 0.0
        weights = {}
        for lot in lots:
            price = row[lot["symbol"]]
            if pd.notna(price):
                marks[lot["symbol"]] = (price, i)
            mark, seen = marks[lot["symbol"]]
            lot_value = lot["units"] * mark
            value += lot_value
            weights[lot["symbol"]] = weights.get(lot["symbol"], 0.0) + lot_value
            if i - seen > STALE_SESSIONS:
                stale += lot_value
        cash = sum(sleeve_cash)
        nav = cash + value
        history.append((day, nav, cash, value, stale,
                        max(weights.values(), default=0.0) / nav, len(weights)))

        # 4. Plan tomorrow's purchases from today's close.
        if i + 1 < len(sessions):
            entry_day = sessions[i + 1]
            for sleeve in range(SLEEVES):
                if day >= sleeve_start[sleeve] and sleeve_cash[sleeve] >= MIN_ORDER:
                    names = pick(scores, members, day, entry_day)
                    if names:
                        pending[sleeve] = (names, sleeve_cash[sleeve])

    daily = pd.DataFrame(history, columns=[
        "date", "equity", "cash", "positions", "stale_value",
        "max_weight", "n_names",
    ]).set_index("date")
    fills = pd.DataFrame(fills, columns=[
        "date", "sleeve", "symbol", "side", "units", "price", "cash_flow",
    ])
    open_lots = pd.DataFrame(lots)
    incidents = pd.DataFrame(incidents, columns=["date", "symbol", "event"])
    return daily, fills, open_lots, incidents


def check_accounting(daily, fills, open_lots, prices):
    """Rebuild cash and the final value from the fill log."""
    cash = CAPITAL + fills["cash_flow"].sum()
    assert np.isclose(cash, daily["cash"].iloc[-1]), "cash does not reconcile"
    last = daily.index[-1]
    if not open_lots.empty:
        marks = prices.loc[:last, open_lots["symbol"]].ffill().iloc[-1]
        held = (open_lots["units"] * marks.to_numpy()).sum()
    else:
        held = 0.0
    assert np.isclose(cash + held, daily["equity"].iloc[-1]), "NAV does not reconcile"

    buys = fills[fills["side"] == "BUY"].iloc[0]
    sells = fills[fills["side"] == "SELL"]
    match = sells[(sells["symbol"] == buys["symbol"]) & (sells["date"] > buys["date"])]
    if not match.empty:
        sale = match.iloc[0]
        held_days = (sale["date"] - buys["date"]).days
        assert held_days >= 90, "a lot was sold before 90 calendar days"
        print(f"Hand check: bought {buys['symbol']} on {buys['date'].date()} at "
              f"{buys['price']:.4f}, sold {sale['date'].date()} ({held_days} days) "
              f"at {sale['price']:.4f}; return after fees "
              f"{sale['cash_flow'] / -buys['cash_flow'] - 1:+.4%}")
    print("Accounting reconciles: cash and final NAV rebuilt from the fill log.")


def run_period(db, label, calendar, prices, scores, members):
    start, end = PERIODS[label]
    daily, fills, open_lots, incidents = simulate(
        calendar, prices, scores, members, start, end
    )
    check_accounting(daily, fills, open_lots, prices)
    daily["benchmark"] = load_benchmark(db, daily.index)

    summary = pd.DataFrame({
        "Risk-adj momentum": metrics(daily["equity"]),
        "S&P 500 TR": metrics(daily["benchmark"]),
    }).T
    print(f"\n{label}: {start.date()} to {end.date()}")
    print(summary.round(4).to_string())
    print(f"Average cash {daily['cash'].div(daily['equity']).mean():.1%}, "
          f"peak stale value {daily['stale_value'].div(daily['equity']).max():.1%}, "
          f"largest name weight {daily['max_weight'].max():.1%}, "
          f"incidents {len(incidents)}")

    out = OUTPUT / label
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "portfolio_metrics.csv")
    daily.to_csv(out / "daily.csv")
    fills.to_csv(out / "fills.csv", index=False)
    open_lots.to_csv(out / "open_positions.csv", index=False)
    incidents.to_csv(out / "incidents.csv", index=False)

    wealth = daily[["equity", "benchmark"]].rename(columns={
        "equity": "Risk-adj momentum", "benchmark": "S&P 500 TR",
    })
    ax = (wealth / CAPITAL - 1).plot(title=f"Cumulative returns ({label})")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.figure.savefig(out / "cumulative_returns.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)


def main():
    with duckdb.connect(str(DATABASE), read_only=True) as db:
        calendar, prices, members = load_inputs(db)
        if calendar[-1] != FINAL_END:
            raise ValueError("Use the complete homework database through 2026-09-30.")
        scores = make_scores(prices)
        for label in PERIODS:
            run_period(db, label, calendar, prices, scores, members)


if __name__ == "__main__":
    main()
