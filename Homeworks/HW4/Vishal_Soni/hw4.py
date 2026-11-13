"""Monthly, point-in-time 90-day equity model and independent backtester."""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FEATURES = ["mom21", "mom63", "mom252_21", "vol63"]
INITIAL = 1_000_000.0
FEE = 0.001
HOLD_DAYS = 90


def load_samples(con):
    """Build monthly signal rows in ticker batches; quotes are never forward-filled."""
    benchmark = con.execute("SELECT date, adj_close FROM benchmark ORDER BY date").df()
    benchmark.date = pd.to_datetime(benchmark.date)
    calendar = pd.DatetimeIndex(benchmark.date)
    monthly = pd.Series(calendar, index=calendar).groupby(calendar.to_period("M")).first()
    entries = pd.DatetimeIndex(monthly[(monthly >= "2010-01-01") & (monthly <= "2026-09-30")])
    signal_ix = calendar.get_indexer(entries) - 1
    valid = signal_ix >= 0
    entries, signal_ix = entries[valid], signal_ix[valid]
    signals = calendar[signal_ix]
    membership = con.execute("SELECT symbol, quarter_start, selection_date, universe_rank FROM universe_membership WHERE universe_rank <= 200").df()
    membership.quarter_start = pd.to_datetime(membership.quarter_start)
    membership.selection_date = pd.to_datetime(membership.selection_date)
    members = {(r.symbol, r.quarter_start): (r.selection_date, r.universe_rank)
               for r in membership.itertuples(index=False)}
    symbols = sorted(membership.symbol.unique())
    pieces = []
    for start in range(0, len(symbols), 100):
        batch = symbols[start:start + 100]
        quotes = con.execute("""SELECT symbol, date, adj_close FROM prices
            WHERE symbol IN (SELECT unnest(?)) AND date <= '2026-09-30'
            ORDER BY symbol, date""", [batch]).df()
        quotes.date = pd.to_datetime(quotes.date)
        for symbol, group in quotes.groupby("symbol", sort=False):
            px = group.set_index("date").adj_close.reindex(calendar)
            px = px.where(np.isfinite(px) & (px > 0))
            rets = px.pct_change(fill_method=None)
            f = pd.DataFrame({"mom21": px / px.shift(21) - 1,
                              "mom63": px / px.shift(63) - 1,
                              "mom252_21": px.shift(21) / px.shift(252) - 1,
                              "vol63": rets.rolling(63, min_periods=63).std() * np.sqrt(252)}, index=calendar)
            f = f.reindex(signals).copy()
            f["entry_date"] = entries
            f["signal_date"] = signals
            f["symbol"] = symbol
            info = [members.get((symbol, d.to_period("Q").start_time)) for d in entries]
            f["member"] = [v is not None and v[0] <= s for v, s in zip(info, signals)]
            f = f.loc[f.member & f[FEATURES].notna().all(axis=1)].copy()
            if f.empty:
                continue
            entry_px = px.reindex(pd.DatetimeIndex(f.entry_date)).to_numpy()
            target, label_end = [], []
            for entry, buy in zip(f.entry_date, entry_px):
                due_ix = calendar.searchsorted(entry + pd.Timedelta(days=HOLD_DAYS))
                if not np.isfinite(buy) or due_ix >= len(calendar):
                    target.append(np.nan); label_end.append(pd.NaT); continue
                # Censor targets if no quoted close within five sessions of maturity.
                future = px.iloc[due_ix:min(due_ix + 6, len(calendar))].dropna()
                if future.empty:
                    target.append(np.nan); label_end.append(pd.NaT)
                else:
                    target.append(float(future.iloc[0] / buy - 1))
                    label_end.append(future.index[0])
            f["target"] = target
            f["label_end"] = label_end
            pieces.append(f.reset_index(drop=True))
        print(f"Features: {min(start + 100, len(symbols))}/{len(symbols)} symbols", flush=True)
    return pd.concat(pieces, ignore_index=True), benchmark


def fit_and_predict(samples):
    """Freeze model specification before the final period; no test retraining."""
    model = lambda: make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    train = samples[(samples.signal_date >= "2010-01-01") & (samples.label_end < "2014-01-01") & samples.target.notna()]
    valid = samples[(samples.entry_date >= "2014-01-01") & (samples.entry_date <= "2015-09-30") & (samples.label_end <= "2015-12-31") & samples.target.notna()]
    assert train.label_end.max() < valid.signal_date.min()
    m = model().fit(train[FEATURES], train.target)
    pred = m.predict(valid[FEATURES])
    validation = {"train_rows": len(train), "validation_rows": len(valid),
                  "rmse": float(np.sqrt(np.mean((pred - valid.target) ** 2))),
                  "baseline_rmse": float(np.sqrt(np.mean((train.target.mean() - valid.target) ** 2)))}
    final_train = samples[(samples.signal_date >= "2010-01-01") & (samples.label_end <= "2015-12-31") & samples.target.notna()]
    final = samples[(samples.entry_date >= "2016-04-01") & (samples.entry_date <= "2026-09-30")].copy()
    assert final_train.label_end.max() < final.signal_date.min()
    m = model().fit(final_train[FEATURES], final_train.target)
    final["score"] = m.predict(final[FEATURES])
    return final, validation, len(final_train)


def run_backtest(calendar, quotes, orders, start="2016-04-01", end="2026-09-30", initial=INITIAL, fee=FEE, slots=15, allow_stale=False):
    """Signal-close order budgets; observed next-close fills; individual 90-day lots."""
    calendar = pd.DatetimeIndex(calendar)
    dates = calendar[(calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))]
    book = quotes.pivot(index="date", columns="symbol", values="adj_close").reindex(calendar)
    orders = orders.sort_values(["signal_date", "score", "symbol"], ascending=[True, False, True])
    schedules = {d: g.head(slots).symbol.tolist() for d, g in orders.groupby("signal_date")}
    lots, fills, rows = [], [], []
    cash = float(initial)
    pending = None
    stale_max = 0
    # Include previous close to schedule the first test-day orders.
    previous = calendar[calendar.get_loc(dates[0]) - 1]
    for day in pd.DatetimeIndex([previous]).append(dates):
        today = book.loc[day]
        if day != previous:
            survivors = []
            for lot in lots:
                p = today.get(lot["symbol"], np.nan)
                if pd.notna(p) and np.isfinite(p) and p > 0:
                    lot["mark"], lot["stale"] = float(p), 0
                else:
                    lot["stale"] += 1
                    stale_max = max(stale_max, lot["stale"])
                    if lot["stale"] > 5 and not allow_stale:
                        raise ValueError(f"Unresolved held-stock price gap: {lot['symbol']} on {day.date()}")
                if day >= lot["maturity"] and lot["stale"] == 0:
                    proceeds = lot["units"] * lot["mark"] * (1 - fee)
                    cash += proceeds
                    fills.append((day, "sell", lot["symbol"], lot["units"], lot["mark"], proceeds))
                else:
                    survivors.append(lot)
            lots = survivors
            if pending is not None:
                names, each = pending
                for symbol in names:
                    price = today.get(symbol, np.nan)
                    if pd.isna(price) or not np.isfinite(price) or price <= 0:
                        continue  # no invented fill; leave the allocated cash unspent
                    spend = min(each, cash)
                    units = spend / (float(price) * (1 + fee))
                    if units <= 0:
                        continue
                    cash -= spend
                    lots.append({"symbol": symbol, "units": units, "mark": float(price),
                                 "maturity": day + pd.Timedelta(days=HOLD_DAYS), "stale": 0})
                    fills.append((day, "buy", symbol, units, float(price), spend))
                pending = None
            assert cash >= -1e-7
            value = cash + sum(lot["units"] * lot["mark"] for lot in lots)
            rows.append((day, cash, value, len(lots)))
        if day in schedules:
            names = schedules[day]
            equity = cash + sum(lot["units"] * lot["mark"] for lot in lots)
            budget = min(cash, equity / 3)
            pending = (names, budget / len(names)) if names else None
    return (pd.DataFrame(rows, columns=["date", "cash", "value", "open_lots"]).set_index("date"),
            pd.DataFrame(fills, columns=["date", "side", "symbol", "units", "price", "cash_flow"]), lots, stale_max)


def statistics(values, initial=INITIAL):
    values = values.astype(float)
    daily = values.pct_change(fill_method=None)
    daily.iloc[0] = values.iloc[0] / initial - 1
    wealth = pd.concat([pd.Series([initial]), values.reset_index(drop=True)], ignore_index=True)
    days = (values.index[-1] - values.index[0]).days
    return {"final_value": float(values.iloc[-1]),
            "cagr": float((values.iloc[-1] / initial) ** (365.25 / days) - 1),
            "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
            "sharpe": float((daily - (1.02 ** (1 / 252) - 1)).mean() / daily.std(ddof=1) * np.sqrt(252))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-stale", action="store_true", help="Exploratory only: carry last marks beyond five sessions; report unresolved lots")
    parser.add_argument("--database", type=Path, default=ROOT / "Lectures/modeling/data/equities_top2000.duckdb")
    args = parser.parse_args()
    with duckdb.connect(str(args.database), read_only=True) as con:
        samples, benchmark = load_samples(con)
        scores, validation, ntrain = fit_and_predict(samples)
        # Selecting names uses only model scores, not target or future-price availability.
        selected = scores.sort_values(["signal_date", "score", "symbol"], ascending=[True, False, True]).groupby("signal_date").head(15)
        symbols = selected.symbol.unique().tolist()
        quotes = con.execute("SELECT symbol,date,adj_close FROM prices WHERE symbol IN (SELECT unnest(?)) AND date BETWEEN '2016-03-31' AND '2026-09-30'", [symbols]).df()
    quotes.date = pd.to_datetime(quotes.date)
    quotes.loc[~np.isfinite(quotes.adj_close) | (quotes.adj_close <= 0), "adj_close"] = np.nan
    portfolio, fills, lots, stale = run_backtest(benchmark.date, quotes, scores, allow_stale=args.allow_stale)
    benchmark = benchmark.set_index("date").loc[portfolio.index]
    bench_values = INITIAL * benchmark.adj_close / benchmark.adj_close.iloc[0]
    strat_stats, bench_stats = statistics(portfolio.value), statistics(bench_values)
    print("Validation:", validation, "Final training rows:", ntrain)
    print("Strategy:", strat_stats)
    print("Benchmark:", bench_stats)
    print(f"Trades: {len(fills)}; open lots: {len(lots)}; max stale sessions: {stale}")
    print("Unresolved lots:", [(lot["symbol"], lot["maturity"].date(), lot["stale"], round(lot["units"] * lot["mark"], 2)) for lot in lots if lot["stale"] > 5])
    print(f"Difference: ${strat_stats['final_value'] - bench_stats['final_value']:,.2f}")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(portfolio.index, portfolio.value / INITIAL - 1, label="Strategy")
    ax.plot(bench_values.index, bench_values / INITIAL - 1, label="S&P 500 total return")
    ax.set(title="Exploratory returns (stale holding): 2016–2026" if args.allow_stale else "Cumulative return: 2016–2026", ylabel="Cumulative return", xlabel="Year")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(HERE / ("provisional_returns.png" if args.allow_stale else "cumulative_returns.png"), dpi=160)
    pd.DataFrame([strat_stats, bench_stats], index=["Strategy", "S&P 500 TR"]).to_csv(HERE / ("provisional_metrics.csv" if args.allow_stale else "metrics.csv"))


if __name__ == "__main__":
    main()
