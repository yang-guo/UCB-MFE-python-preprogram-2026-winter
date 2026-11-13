"""Evaluate the HW4 strategy on the full database."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from hw4_backtest import audit, run_backtest

REPO = next(path for path in Path(__file__).resolve().parents
            if (path / "Lectures/modeling/course_data.py").is_file())
MODELING = REPO / "Lectures/modeling"
sys.path.insert(0, str(MODELING))
from course_data import apply_membership, load_membership, make_labels, price_features

START = pd.Timestamp("2016-04-01")
END = pd.Timestamp("2026-09-30")
CAPITAL = 1_000_000
FEE = 0.001
BUDGET = 250_000
FEATURES = ["momentum_63", "volatility_63"]

CASH_EVENTS = {
    ("TPTX", pd.Timestamp("2022-08-17")): (76.0, "CASH_MERGER"),
    ("GBT", pd.Timestamp("2022-10-05")): (68.50, "CASH_MERGER"),
    ("CIFRW", pd.Timestamp("2025-12-26")): (0.01, "WARRANT_REDEMPTION"),
}


def read_prices(connection, symbols, start, end, calendar):
    prices = connection.execute("""
        SELECT symbol, date, adj_close, close, volume FROM prices
        WHERE symbol IN (SELECT unnest(?)) AND date BETWEEN ? AND ?
        ORDER BY symbol, date
    """, [symbols, start, end]).fetchdf()
    prices["date"] = pd.to_datetime(prices["date"]).astype("datetime64[ns]")
    valid = prices["date"].isin(calendar) & np.isfinite(prices["adj_close"]) & prices["adj_close"].gt(0)
    return prices.loc[valid].copy()


def build_samples(connection, membership, calendar):
    sessions = calendar[(calendar >= START) & (calendar <= END)]
    signals = sessions[~sessions.to_period("M").duplicated()]
    symbols = connection.execute("SELECT symbol FROM universe ORDER BY symbol").fetchnumpy()["symbol"].tolist()
    training, candidates = [], []
    for offset in range(0, len(symbols), 128):
        batch = symbols[offset:offset + 128]
        prices = read_prices(connection, batch, "2009-01-01", END, calendar)
        if prices.empty:
            continue
        panel = price_features(prices, calendar)
        panel = apply_membership(panel, membership[membership["symbol"].isin(batch)], calendar)
        eligible = panel.loc[panel["eligible"]]
        development = eligible.loc[eligible["date"].between("2010-01-01", "2015-12-31")]
        if not development.empty:
            labels = make_labels(development, prices, calendar, holding_days=90, max_exit_delay=0)
            known = labels["label_end"].lt(START) & labels["target"].notna()
            training.append(labels.loc[known, ["date", "symbol", "label_end", "target", *FEATURES]])
        candidates.append(eligible.loc[eligible["date"].isin(signals), ["date", "symbol", *FEATURES]])
    train = pd.concat(training, ignore_index=True)
    test = pd.concat(candidates, ignore_index=True)
    assert train["label_end"].max() < START and train["date"].max() < START
    assert np.isfinite(train[[*FEATURES, "target"]]).all().all()
    assert np.isfinite(test[FEATURES]).all().all()
    return train, test


def metrics(nav):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    daily = nav.pct_change().dropna()
    excess = daily - ((1.02 ** (1 / 252)) - 1)
    volatility = daily.std(ddof=1)
    return {
        "final_value": nav.iloc[-1],
        "cumulative_return": nav.iloc[-1] / CAPITAL - 1,
        "cagr": (nav.iloc[-1] / CAPITAL) ** (1 / years) - 1,
        "max_drawdown": (nav / nav.cummax() - 1).min(),
        "sharpe": np.sqrt(252) * excess.mean() / volatility if volatility > 0 else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=MODELING / "data/equities_top2000.duckdb")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs/volume_capped")
    args = parser.parse_args()
    expected = json.loads((MODELING / "data/manifest.json").read_text())["files"]["equities_top2000.duckdb"]
    if not args.database.is_file():
        parser.error(f"Download equities_top2000.duckdb from bCourses to {args.database}.")
    with args.database.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if args.database.stat().st_size != expected["size_bytes"] or checksum != expected["sha256"]:
        raise ValueError("The database does not match the full course delivery manifest.")

    with duckdb.connect(str(args.database), read_only=True) as connection:
        calendar = pd.DatetimeIndex(connection.sql("SELECT date FROM calendar ORDER BY date").df()["date"]).as_unit("ns")
        benchmark = connection.sql("SELECT date, adj_close FROM benchmark ORDER BY date").df()
        benchmark["date"] = pd.to_datetime(benchmark["date"]).astype("datetime64[ns]")
        membership = load_membership(args.database, end=END.date().isoformat())
        train, candidates = build_samples(connection, membership, calendar)

        model = make_pipeline(StandardScaler(), Ridge(alpha=100))
        model.fit(train[FEATURES], train["target"])
        candidates["prediction"] = model.predict(candidates[FEATURES])
        winners = candidates.sort_values(
            ["date", "prediction", "symbol"], ascending=[True, False, True]
        ).drop_duplicates("date")[["date", "symbol", "prediction"]]
        execution_prices = read_prices(
            connection, sorted(winners["symbol"].unique()), START, END, calendar
        )
        if "GLYC" in set(winners["symbol"]):
            # GLYC prices are already on the CBIO post-split basis.
            rename_date = pd.Timestamp("2025-06-16")
            successor = read_prices(connection, ["CBIO"], rename_date, END, calendar)
            alias = successor.assign(symbol="GLYC")
            execution_prices = pd.concat([
                execution_prices.loc[
                    ~execution_prices["symbol"].eq("GLYC") | execution_prices["date"].lt(rename_date)
                ],
                alias,
            ], ignore_index=True)

    history, fills, open_lots, skipped = run_backtest(
        winners, execution_prices, calendar, START, END,
        initial_cash=CAPITAL, order_budget=BUDGET, fee_rate=FEE, cash_events=CASH_EVENTS,
    )
    audit(history, fills, open_lots, execution_prices, calendar, cash_events=CASH_EVENTS)
    assert history.index[0] == START and history.index[-1] == END
    benchmark_price = benchmark.set_index("date")["adj_close"].reindex(history.index)
    assert benchmark_price.notna().all() and benchmark_price.gt(0).all()
    equity = pd.DataFrame({
        "Ridge strategy": history["equity"],
        "S&P 500 total return": CAPITAL * benchmark_price / benchmark_price.iloc[0],
    })
    comparison = pd.DataFrame({name: metrics(equity[name]) for name in equity}).T
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "metrics.csv")
    equity.join(history[["cash", "position_value", "open_lots", "stale_lots"]]).to_csv(output / "daily_equity.csv")
    fills.to_csv(output / "fills.csv", index=False)
    open_lots.to_csv(output / "open_lots.csv", index=False)
    skipped.to_csv(output / "skipped_buys.csv", index=False)
    winners.to_csv(output / "monthly_selections.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    cumulative = equity / CAPITAL - 1
    ax.plot(cumulative.index, cumulative.iloc[:, 0], label=equity.columns[0], color="#176680", linewidth=1.8)
    ax.plot(cumulative.index, cumulative.iloc[:, 1], label=equity.columns[1], color="#a95127", linewidth=1.8)
    ax.axhline(0, color="0.7", linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(title="HW4 | April 2016–September 2026", ylabel="Cumulative return")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "cumulative_return.png", dpi=160)
    plt.close(fig)

    print(comparison.round(4).to_string())
    print(f"Ledger verified for {len(history):,} sessions.")


if __name__ == "__main__":
    main()
