import argparse
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_absolute_error, r2_score

START = "2016-04-01"
END = "2026-09-30"
CAPITAL = 1_000_000
FEE = 0.0005
TOP_N = 10
FEATURES = ["return_5", "return_10", "return_21", "return_63", "return_252_skip_21", "volatility_63"]


def load_market(database):
    with duckdb.connect(str(database), read_only=True, config={"threads": 2}) as con:
        benchmark = con.execute("""
            SELECT date, adj_close FROM benchmark
            WHERE date BETWEEN '2008-01-01' AND ?
            ORDER BY date
        """, [END]).df().set_index("date").adj_close
        prices = con.execute("""
            SELECT symbol, date, adj_close FROM prices
            WHERE date BETWEEN '2008-01-01' AND ?
              AND isfinite(adj_close) AND adj_close > 0
        """, [END]).df()
        members = con.execute("""
            SELECT symbol, quarter_start, selection_date FROM universe_membership
            WHERE quarter_start BETWEEN '2010-01-01' AND ?
        """, [END]).df()
    benchmark.index = pd.DatetimeIndex(benchmark.index)
    assert benchmark.index.is_unique and benchmark.gt(0).all()
    quotes = prices.pivot(index="date", columns="symbol", values="adj_close")
    return quotes.reindex(benchmark.index), benchmark, members

# Find the course repository and use the script’s default database.


def monthly_features(values, daily, symbols, signal):
    return pd.DataFrame(
        {
            "return_5": values[signal] / values[signal - 5] - 1,
            "return_10": values[signal] / values[signal - 10] - 1,
            "return_21": values[signal] / values[signal - 21] - 1,
            "return_63": values[signal] / values[signal - 63] - 1,
            "return_252_skip_21": values[signal - 21] / values[signal - 252] - 1,
            "volatility_63": daily.iloc[signal - 62 : signal + 1]
            .std(skipna=False)
            .to_numpy(),
        },
        index=symbols,
    )


def add_return_labels(panel, quotes, entry, exit_row):
    panel["label_end"] = pd.NaT
    panel["target"] = np.nan

    entry_price = quotes.iloc[entry].reindex(panel.index)
    dates = quotes.index

    for row in range(exit_row, min(exit_row + 6, len(dates))):
        exit_price = quotes.iloc[row].reindex(panel.index)
        valid = panel["target"].isna() & entry_price.notna() & exit_price.notna()

        panel.loc[valid, "label_end"] = dates[row]
        panel.loc[valid, "target"] = exit_price[valid] / entry_price[valid] - 1

    return panel


def monthly_panel(quotes, members):
    dates = quotes.index
    months = dates.to_period("M")

    # purchase after signal date, and hold for 90 days. The signal date is the last day of the month.
    entries = np.flatnonzero(months[1:] != months[:-1]) + 1
    entries = entries[dates[entries] >= "2010-01-01"]
    signals = entries - 1
    exit_rows = dates.searchsorted(dates[entries] + pd.Timedelta(days=90))

    daily = quotes.pct_change(fill_method=None)
    values = quotes.to_numpy()
    parts = []

    for entry, signal, exit_row in zip(entries, signals, exit_rows):
        if signal < 252:
            continue

        signal_date = dates[signal]
        entry_date = dates[entry]
        quarter = entry_date.to_period("Q").start_time

        # Select the eligible symbols for the current quarter and signal date
        eligible = members.loc[
            members["quarter_start"].eq(quarter)
            & members["selection_date"].le(signal_date),
            "symbol",
        ]

        panel = monthly_features(values, daily, quotes.columns, signal)
        panel = panel.loc[panel.index.isin(eligible)]
        panel = panel.replace([np.inf, -np.inf], np.nan)
        panel = panel.dropna(subset=FEATURES)

        panel["signal"] = signal_date
        panel["entry"] = entry_date
        panel = add_return_labels(panel, quotes, entry, exit_row)

        parts.append(panel.reset_index())

    return pd.concat(parts, ignore_index=True)


def select_ridge_alpha(development):
    # Fit each fold using only return labels available before validation.
    scores = {}
    validation_rows = []
    for alpha in (1, 100, 1000):
        errors = []
        for year in (2013, 2014, 2015):
            cutoff = pd.Timestamp(year=year, month=1, day=1)
            train = development.loc[development.label_end.lt(cutoff)]
            valid = development.loc[
                development.signal.dt.year.eq(year)
                & development.label_end.lt("2016-01-01")
            ]
            assert train.label_end.max() < valid.signal.min()
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(train[FEATURES], train.target)
            predicted = model.predict(valid[FEATURES])
            rmse = np.sqrt(np.mean((predicted - valid.target) ** 2))
            errors.append(rmse)
            ridge = model.named_steps["ridge"]
            validation_rows.append({
                "alpha": alpha,
                "validation_year": year,
                "training_rows": len(train),
                "validation_rows": len(valid),
                "rmse": rmse,
                "intercept": ridge.intercept_,
                **dict(zip(FEATURES, ridge.coef_)),
            })
        scores[alpha] = np.mean(errors)
    return min(scores, key=scores.get), scores, pd.DataFrame(validation_rows)


def rank_test_stocks(model, test):
    # Rank candidates by predicted return within each entry date.
    test["score"] = model.predict(test[FEATURES])
    ranked = test.sort_values(["entry", "score", "symbol"], ascending=[True, False, True])
    picks = {date: list(group.symbol.head(TOP_N)) for date, group in ranked.groupby("entry")}
    return picks


def development_samples(panel):
    return panel.loc[
        panel.signal.between("2010-01-01", "2015-12-31") & panel.entry.lt("2016-01-01")
        & panel.label_end.lt(START)
    ]


def fit_and_rank(panel):
    development = development_samples(panel)
    alpha, scores, validation_results = select_ridge_alpha(development)
    # Refit the selected setting on all eligible development samples.
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(development[FEATURES], development.target)
    test = panel.loc[panel.entry.between(START, END)].copy()
    assert development.label_end.max() <= test.signal.min()
    assert test.signal.lt(test.entry).all()
    picks = rank_test_stocks(model, test)
    return picks, scores, alpha, len(development), development.label_end.max(), validation_results, model


@dataclass
class Lot:
    symbol: str
    units: float
    entry: pd.Timestamp
    due: pd.Timestamp
    mark: float
    stale: int = 0


def backtest(quotes, picks, capital=CAPITAL, fee=FEE, start=START):
    cash = capital
    nav = capital
    lots, fills, records = {}, [], []
    next_id = 0
    missing_entries = 0
    max_stale = 0
    gaps = {}
    sessions = quotes.loc[start:]
    initial_date = quotes.index[quotes.index.get_loc(sessions.index[0]) - 1]
    records.append((initial_date, cash, 0.0, nav, 0))
    for date, observed in sessions.iterrows():
        symbols = picks.get(date, [])
        # Size orders from the previous close before observing execution prices.
        budget = min(cash, nav / 3) / len(symbols) if symbols else 0
        for lot_id, lot in list(lots.items()):
            price = observed[lot.symbol]
            if pd.isna(price):
                lot.stale += 1
                max_stale = max(max_stale, lot.stale)
                if lot.stale > 5:
                    gaps[lot.symbol] = max(gaps.get(lot.symbol, 0), lot.stale)
                continue
            lot.mark, lot.stale = price, 0
            if date >= lot.due:
                notional = lot.units * price
                charge = notional * fee
                cash += notional - charge
                fills.append((date, lot_id, lot.symbol, "SELL", notional - charge, charge, lot.due))
                del lots[lot_id]
        for symbol in symbols:
            if budget <= 1e-8:
                continue
            price = observed[symbol]
            if pd.isna(price):
                missing_entries += 1
                continue
            notional = budget / (1 + fee)
            due = date + pd.Timedelta(days=90)
            lots[next_id] = Lot(symbol, notional / price, date, due, price)
            cash -= budget
            fills.append((date, next_id, symbol, "BUY", -budget, notional * fee, due))
            next_id += 1
        holdings = sum(lot.units * lot.mark for lot in lots.values())
        nav = cash + holdings
        records.append((date, cash, holdings, nav, len(lots)))
    history = pd.DataFrame(records, columns=["date", "cash", "holdings", "nav", "open_lots"]).set_index("date")
    ledger = pd.DataFrame(fills, columns=["date", "lot", "symbol", "side", "cash_change", "fee", "due"])
    changes = ledger.groupby("date").cash_change.sum().reindex(history.index, fill_value=0)
    np.testing.assert_allclose(history.cash, capital + changes.cumsum(), rtol=0, atol=1e-6)
    np.testing.assert_allclose(history.nav, history.cash + history.holdings)
    assert history.cash.min() >= -1e-7
    sales = ledger.loc[ledger.side.eq("SELL")]
    assert sales.date.ge(sales.due).all()
    return history, ledger, lots, missing_entries, max_stale, gaps



def performance(nav):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    returns = nav.pct_change().dropna()
    rf = 1.02 ** (1 / 252) - 1
    volatility = returns.std()
    return {
        "Final value": nav.iloc[-1],
        "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
        "Max drawdown": (nav / nav.cummax() - 1).min(),
        "Sharpe": np.sqrt(252) * (returns.mean() - rf) / volatility if volatility > 0 else np.nan,
    }



def main():
    parser = argparse.ArgumentParser(description="Run the HW4 Ridge backtest.")
    root = Path(__file__).resolve().parents[3]
    relative = Path("Lectures/modeling/data/equities_top2000.duckdb")
    fallback = Path.home() / "Desktop/dev/ucb-mfe/UCB-MFE-python-preprogram-2026-winter"
    database = root / relative
    if not database.is_file():
        database = fallback / relative
    parser.add_argument("--database", type=Path, default=database)
    parser.add_argument("--chart", type=Path, default=Path(__file__).with_name("cumulative_return.png"))
    args = parser.parse_args()

    quotes, benchmark, members = load_market(args.database)
    panel = monthly_panel(quotes, members)
    picks, errors, alpha, rows, last_label, validation_results, final_model = fit_and_rank(panel)
    print("Development validation RMSE and standardized coefficients")
    print(validation_results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("Validation RMSE", {k: round(float(v), 4) for k, v in errors.items()})
    print("Selected alpha", alpha, "Training rows", rows, "Last training label", last_label.date())
    ridge = final_model.named_steps["ridge"]
    print(f"\nFinal model (alpha={alpha}, standardized features)")
    print(pd.Series(ridge.coef_, index=FEATURES, name="coefficient").to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"Intercept: {ridge.intercept_:.6f}")

    train = development_samples(panel)
    test = panel.loc[panel.entry.between(START, END) & panel.target.notna()].copy()

    forecast_metrics = {}
    for name, sample in [("Train", train), ("Test", test)]:
        actual = sample.target
        forecast = final_model.predict(sample[FEATURES])
        forecast_metrics[name] = {
            "MAE (percentage points)": 100 * mean_absolute_error(actual, forecast),
            "RMSE (percentage points)": 100 * np.sqrt(np.mean((forecast - actual) ** 2)),
            "R²": r2_score(actual, forecast),
            "Direction accuracy (%)": 100 * ((forecast > 0) == (actual > 0)).mean(),
            "Always-up baseline (%)": 100 * actual.gt(0).mean(),
        }
    accuracy = pd.DataFrame(forecast_metrics).T
    print(accuracy.round(3).to_string())

    test["forecast"] = final_model.predict(test[FEATURES])
    rank_ic = test.groupby("entry")[["target", "forecast"]].corr(method="spearman")
    rank_ic = rank_ic.xs("forecast", level=1).target
    print(f"Mean monthly test Rank IC: {rank_ic.mean():.4f}")

    # Compare the original picks with the scored pool using available returns.
    selected = pd.Series(picks).rename_axis("entry").explode().rename("symbol")
    is_pick = pd.MultiIndex.from_frame(test[["entry", "symbol"]]).isin(
        pd.MultiIndex.from_frame(selected.reset_index())
    )
    pool = test.groupby("entry").target
    top = test.loc[is_pick].groupby("entry").target
    monthly = pd.DataFrame(index=pd.DatetimeIndex(picks.keys(), name="entry"))
    monthly["Top picks (%)"] = 100 * top.mean()
    monthly["Pool (%)"] = 100 * pool.mean()
    monthly["Difference (pp)"] = monthly["Top picks (%)"] - monthly["Pool (%)"]
    monthly["Picks with returns"] = top.size().reindex(monthly.index, fill_value=0)
    monthly["Pool with returns"] = pool.size().reindex(monthly.index, fill_value=0)
    print(f"Average monthly difference: {monthly['Difference (pp)'].mean():.3f} pp")
    print(monthly.round(3).to_string())

    history, fills, lots, rejected, stale, gaps = backtest(quotes, picks)
    market = CAPITAL * benchmark.reindex(history.index) / benchmark.loc[START]
    market.iloc[0] = CAPITAL
    nav = pd.DataFrame({"Ridge strategy": history.nav, "S&P 500 total return": market})
    metrics = pd.DataFrame({name: performance(nav[name]) for name in nav})
    print(metrics.to_string())
    print("Buys:", fills.side.eq("BUY").sum(), "Sales:", fills.side.eq("SELL").sum())
    print("Fees:", fills.fee.sum(), "Average cash fraction:", (history.cash / history.nav).iloc[1:].mean())
    print("Open lots:", len(lots), "Rejected entries:", rejected, "Max stale sessions:", stale, "Long gaps:", gaps)
    print("Stale open lots:", [(lot.symbol, lot.stale, lot.units * lot.mark) for lot in lots.values() if lot.stale])
    print(f"Stale marks as share of final NAV: {sum(lot.units * lot.mark for lot in lots.values() if lot.stale) / history.nav.iloc[-1]:.2%}")

    import matplotlib.pyplot as plt

    ax = nav.div(CAPITAL).sub(1).mul(100).plot(figsize=(8, 4))
    ax.set(title="Final test cumulative returns", ylabel="Cumulative return (%)", xlabel="")
    ax.grid(alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(args.chart, dpi=150)
    plt.close(ax.figure)


if __name__ == "__main__":
    main()
