import argparse
import sys
import warnings
from pathlib import Path

import duckdb
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

# course helpers (Lectures/modeling/course_data.py)
ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / "Lectures" / "modeling"))
import course_data as cd

warnings.filterwarnings("ignore", category=FutureWarning)

END = "2026-09-30"
TRAIN_START = "2010-01-01"
DEV_END = "2015-12-31"
TEST_START = "2016-04-01"
TOP_N = 20
COHORT_FRACTION = 1 / 3
FEE_BPS = 5
CAPITAL = 1_000_000
HOLDING_DAYS = 90
MAX_STALE = 5
MAX_RANK = 1000
MIN_PRICE = 5


def month_end_sessions(calendar):
    s = pd.Series(calendar, index=calendar)
    return pd.DatetimeIndex(s.groupby(calendar.to_period("M")).max())


def build_panel(db, calendar, signal_dates, batch_size=250):
    with duckdb.connect(str(db), read_only=True) as con:
        symbols = con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchdf().symbol.tolist()

    panels, test_prices = [], []
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        print(f"  batch {i // batch_size + 1}/{(len(symbols) - 1) // batch_size + 1}")
        prices, fundamentals, universe, _ = cd.load_data(db, end=END, symbols=batch)
        if prices.empty:
            continue
        p = cd.price_features(prices, calendar)
        p = p.loc[p.date.isin(signal_dates)]
        p = cd.join_financials(p, cd.financial_events(fundamentals), universe)
        p = cd.make_labels(p, prices, calendar, holding_days=HOLDING_DAYS)
        panels.append(p)
        test_prices.append(prices.loc[prices.date >= "2016-03-01", ["symbol", "date", "adj_close"]])

    panel = pd.concat(panels, ignore_index=True)
    panel = cd.apply_membership(panel, cd.load_membership(db, end=END), calendar)
    prices = pd.concat(test_prices).pivot(index="date", columns="symbol", values="adj_close")
    return panel, prices.reindex(calendar[calendar >= "2016-03-01"])


def training_rows(panel, fit_date):
    # only outcomes already finished at the fit date; target = rank of the return in the month
    rows = panel.loc[panel.date.ge(TRAIN_START) & panel.eligible
                     & panel.target.notna() & panel.label_end.lt(fit_date)].copy()
    rows["y"] = rows.groupby("date").target.rank(pct=True)
    return rows


def can_buy(rows):
    return rows.universe_rank.le(MAX_RANK) & rows.close.ge(MIN_PRICE)


def choose_model(panel):
    dev = panel.loc[panel.date.le(DEV_END) & panel.eligible].reset_index(drop=True)
    candidates = {
        "ridge": Ridge(alpha=100),
        "boosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                                  max_leaf_nodes=15, random_state=0),
    }
    rows = []
    for name, model in candidates.items():
        for train_ix, valid_ix in cd.date_folds(dev, validation_years=(2014, 2015)):
            train = training_rows(dev.iloc[train_ix], dev.iloc[valid_ix].date.min())
            valid = dev.iloc[valid_ix].copy()
            pipe = cd.make_pipeline(model)
            pipe.fit(train[cd.FEATURES], train.y)
            valid["score"] = pipe.predict(valid[cd.FEATURES])
            m = cd.prediction_metrics(valid, valid.score)
            buyable = valid.loc[can_buy(valid)]
            top20 = buyable.groupby("date").apply(lambda x: x.nlargest(TOP_N, "score").target.mean(),
                                                   include_groups=False)
            rows.append({"model": name, "year": valid.date.dt.year.iloc[0],
                         "rank_IC": m.mean_daily_rank_IC, "top20_return": top20.mean(),
                         "all_stocks_return": buyable.groupby("date").target.mean().mean()})
    results = pd.DataFrame(rows)
    best = results.groupby("model").rank_IC.mean().idxmax()
    return candidates[best], best, results


def predict_test(panel, model, test_signals):
    scores, current = [], None
    for i, s in enumerate(test_signals):
        # refit at the start and then once a year
        if current is None or s.year != test_signals[i - 1].year:
            train = training_rows(panel, s)
            current = cd.make_pipeline(model)
            current.fit(train[cd.FEATURES], train.y)
            print(f"  refit on {s.date()} with {len(train):,} rows")
        rows = panel.loc[panel.date.eq(s) & panel.eligible]
        rows = rows.loc[can_buy(rows)]
        if len(rows):
            scores.append(rows[["date", "symbol"]].assign(score=current.predict(rows[cd.FEATURES])))
    return pd.concat(scores, ignore_index=True), current


class Backtester:

    def __init__(self, prices, scores, capital=CAPITAL, fee_bps=FEE_BPS):
        self.prices = prices
        self.scores = scores.groupby("date")
        self.signal_dates = set(scores.date)
        self.fee = fee_bps / 10_000
        self.cash = float(capital)
        self.capital = float(capital)
        self.lots = []
        self.trades = []
        self.history = []
        self.last_price = {}
        self.last_seen = {}

    def sell(self, lot, day, price):
        proceeds = lot["units"] * price
        fee = proceeds * self.fee
        self.cash += proceeds - fee
        self.trades.append({"date": day, "symbol": lot["symbol"], "side": "SELL", "price": price,
                            "units": lot["units"], "fee": fee, "cash_change": proceeds - fee,
                            "entry_date": lot["entry_date"], "due_date": lot["due_date"]})

    def buy(self, symbol, day, signal_date, price, budget):
        units = budget / (price * (1 + self.fee))
        cost = units * price
        fee = cost * self.fee
        if cost + fee > self.cash + 1e-6:
            raise RuntimeError("Not enough cash")
        self.cash -= cost + fee
        due = day + pd.Timedelta(days=HOLDING_DAYS)
        self.lots.append({"symbol": symbol, "units": units, "entry_date": day,
                          "entry_price": price, "signal_date": signal_date, "due_date": due})
        self.trades.append({"date": day, "symbol": symbol, "side": "BUY", "price": price,
                            "units": units, "fee": fee, "cash_change": -(cost + fee),
                            "signal_date": signal_date, "entry_date": day, "due_date": due})

    def plan_orders(self, day, nav):
        budget = min(self.cash, nav * COHORT_FRACTION)
        top = self.scores.get_group(day).nlargest(TOP_N, "score")
        if budget <= 0 or top.empty:
            return []
        return [(s, day, budget / len(top)) for s in top.symbol]

    def run(self, start, end):
        days = self.prices.loc[start:end].index
        for symbol, px in self.prices.loc[:start].iloc[:-1].items():
            px = px.dropna()
            if len(px):
                self.last_price[symbol] = px.iloc[-1]
                self.last_seen[symbol] = -1

        prev = self.prices.index[self.prices.index.get_loc(days[0]) - 1]
        orders = self.plan_orders(prev, self.capital) if prev in self.signal_dates else []
        self.history.append({"date": prev, "cash": self.cash, "nav": self.capital,
                             "positions": 0, "stale": 0})

        for t, day in enumerate(days):
            row = self.prices.loc[day]
            for symbol, price in row.dropna().items():
                self.last_price[symbol] = price
                self.last_seen[symbol] = t

            # sell lots that reached 90 days (first close available)
            still_open = []
            for lot in self.lots:
                price = row.get(lot["symbol"], np.nan)
                if day >= lot["due_date"] and np.isfinite(price):
                    self.sell(lot, day, price)
                else:
                    still_open.append(lot)
            self.lots = still_open

            # buy the orders decided yesterday at today's close
            for symbol, signal_date, budget in orders:
                price = row.get(symbol, np.nan)
                if np.isfinite(price) and price > 0:
                    self.buy(symbol, day, signal_date, price, budget)
                else:
                    self.trades.append({"date": day, "symbol": symbol, "side": "SKIPPED (no price)",
                                        "price": np.nan, "units": 0, "fee": 0, "cash_change": 0,
                                        "entry_date": pd.NaT, "due_date": pd.NaT})

            # value cash + positions (last price if missing today)
            stale = sum(1 for lot in self.lots if self.last_seen[lot["symbol"]] < t)
            too_old = sum(1 for lot in self.lots if t - self.last_seen[lot["symbol"]] > MAX_STALE)
            nav = self.cash + sum(lot["units"] * self.last_price[lot["symbol"]] for lot in self.lots)
            self.history.append({"date": day, "cash": self.cash, "nav": nav,
                                 "positions": len(self.lots), "stale": stale, "stale_over_5": too_old})

            orders = self.plan_orders(day, nav) if day in self.signal_dates and day < days[-1] else []

        return pd.DataFrame(self.history).set_index("date"), pd.DataFrame(self.trades)


def performance(nav, rf=0.02):
    returns = nav.pct_change().dropna()
    excess = returns - ((1 + rf) ** (1 / 252) - 1)
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return pd.Series({
        "Final value": nav.iloc[-1],
        "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
        "Max drawdown": (nav / nav.cummax() - 1).min(),
        "Sharpe": np.sqrt(252) * excess.mean() / returns.std(),
    })


def accounting_checks(history, trades, prices, calendar):
    out = []
    buys = trades.loc[trades.side.eq("BUY")]
    sells = trades.loc[trades.side.eq("SELL")]
    out.append(f"Cash never negative: {history.cash.min() >= -1e-6} (min {history.cash.min():,.2f})")
    out.append(f"Buys: {len(buys):,}, sales: {len(sells):,}, "
               f"skipped orders (no price): {int(trades.side.str.startswith('SKIPPED').sum())}")
    cash_from_trades = CAPITAL + trades.cash_change.sum()
    out.append(f"Final cash from the trade list: {cash_from_trades:,.2f} vs book {history.cash.iloc[-1]:,.2f}")
    out.append(f"All sales on or after maturity (90 days): {bool((sells.date >= sells.due_date).all())}")
    out.append(f"Max delay after maturity: {(sells.date - sells.due_date).dt.days.max()} calendar days")

    first = buys.iloc[0]
    sale = sells.loc[(sells.symbol == first.symbol) & (sells.entry_date == first.date)].iloc[0]
    buy_cost = first.units * first.price * (1 + FEE_BPS / 10_000)
    sale_cash = sale.units * sale.price * (1 - FEE_BPS / 10_000)
    out.append(f"Example lot {first.symbol}: bought {first.units:.4f} units at {first.price:.4f} "
               f"on {first.date.date()} (cost {buy_cost:,.2f} with fee), sold at {sale.price:.4f} "
               f"on {sale.date.date()} for {sale_cash:,.2f}. Return {sale_cash / buy_cost - 1:.2%}; "
               f"price check: {prices.loc[first.date, first.symbol]:.4f} and "
               f"{prices.loc[sale.date, sale.symbol]:.4f}")

    previous_session = calendar[calendar.get_indexer(buys.date) - 1]
    out.append(f"Every buy is filled the session after its signal: "
               f"{bool((previous_session == buys.signal_date.values).all())}")
    out.append(f"Days with a held price older than 5 sessions: {int((history.get('stale_over_5', 0) > 0).sum())}")
    return out


def md_table(df):
    header = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |" for i, r in zip(df.index, df.values)]
    return "\n".join([header, sep] + rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=str(ROOT / "Lectures/modeling/data/equities_top2000.duckdb"))
    parser.add_argument("--out", default=str(Path(__file__).parent / "results"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    with duckdb.connect(args.db, read_only=True) as con:
        calendar = con.execute("SELECT date FROM calendar WHERE date<=? ORDER BY date", [END]).fetchdf().date
        bench = con.execute("SELECT date, adj_close FROM benchmark WHERE date<=? ORDER BY date",
                            [END]).fetchdf()
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar).astype("datetime64[ns]"))
    bench = bench.set_index(pd.to_datetime(bench.date).astype("datetime64[ns]")).adj_close
    signals = month_end_sessions(calendar)
    signals = signals[signals >= "2009-01-01"]

    print("1) Features and labels")
    panel, prices = build_panel(args.db, calendar, signals)
    print(f"  {len(panel):,} stock-month rows, {int(panel.eligible.sum()):,} eligible")

    print("2) Model choice on 2014-2015")
    model, name, dev_results = choose_model(panel)
    dev_results.to_csv(out / "validation.csv", index=False)
    print(dev_results.round(4).to_string(index=False))
    print(f"  chosen: {name}")

    print("3) Test predictions")
    first_test = calendar[calendar >= TEST_START][0]
    first_signal = calendar[calendar.get_loc(first_test) - 1]
    test_signals = signals[(signals >= first_signal) & (signals < END)]
    if first_signal not in test_signals:
        test_signals = test_signals.insert(0, first_signal)
    scores, last_model = predict_test(panel, model, test_signals)
    joblib.dump(last_model, out / "model.joblib")

    print("4) Backtest")
    bt = Backtester(prices, scores)
    history, trades = bt.run(first_test, END)
    history.to_csv(out / "daily_nav.csv")
    trades.to_csv(out / "trades.csv", index=False)

    bench_nav = CAPITAL * bench.loc[first_test:END] / bench.loc[first_test]
    nav = history.nav.loc[first_test:]
    summary = pd.DataFrame({"Strategy": performance(nav), "S&P 500 TR": performance(bench_nav)})
    summary.to_csv(out / "summary.csv")
    print(summary.round(4).to_string())
    diff = nav.iloc[-1] - bench_nav.iloc[-1]
    print(f"  {'Beat' if diff > 0 else 'Did not beat'} the benchmark by ${diff:,.0f}")

    checks = accounting_checks(history, trades, prices, calendar)
    (out / "accounting_checks.txt").write_text("\n".join(checks))
    print("\n".join(checks))

    plt.figure(figsize=(10, 5))
    plt.plot(nav / CAPITAL - 1, label="Strategy")
    plt.plot(bench_nav / CAPITAL - 1, label="S&P 500 TR")
    plt.title("Cumulative return")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(out / "cumulative_returns.png", dpi=120, bbox_inches="tight")

    open_lots = pd.DataFrame(bt.lots)
    stuck = 0 if open_lots.empty else int((open_lots.due_date <= pd.Timestamp(END)).sum())
    table = pd.DataFrame({
        col: [f"${summary.loc['Final value', col]:,.0f}", f"{summary.loc['CAGR', col]:.2%}",
              f"{summary.loc['Max drawdown', col]:.2%}", f"{summary.loc['Sharpe', col]:.2f}"]
        for col in summary.columns}, index=summary.index)
    lines = ["# Results", "",
             f"Test period: {first_test.date()} to {END}. Model: {name}.", "",
             md_table(table), "",
             f"{'Beat' if diff > 0 else 'Did not beat'} the benchmark by ${diff:,.0f}.", "",
             f"Open lots at the end: {len(open_lots)} (of which {stuck} past maturity with no new price, "
             "valued at their last price).", "",
             "## Validation (2014 and 2015)", "", md_table(dev_results.round(4).set_index("model")), "",
             "## Accounting checks", ""] + [f"- {c}" for c in checks] + [
             "", "![Cumulative return](cumulative_returns.png)"]
    (out / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
