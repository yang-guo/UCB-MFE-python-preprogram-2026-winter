
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model import load_development_data
from train_model import FEATURES, fit_final_model


HERE = Path(__file__).resolve().parent
DB_PATH = (
    HERE.parents[2]
    / "Lectures"
    / "modeling"
    / "data"
    / "equities_top2000.duckdb"
)

START = pd.Timestamp("2016-04-01")
END = pd.Timestamp("2026-09-30")

INITIAL_CAPITAL = 1_000_000.0
NUM_STOCKS = 10
HOLDING_DAYS = 90
RISK_FREE_RATE = 0.02


def performance_metrics(df, column):
    values = df[column].astype(float)
    daily_returns = values.pct_change().dropna()

    years = (
        df["date"].iloc[-1] - df["date"].iloc[0]
    ).days / 365.25

    total_return = values.iloc[-1] / values.iloc[0] - 1

    cagr = (
        (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
        if years > 0 and values.iloc[-1] > 0
        else np.nan
    )

    drawdown = values / values.cummax() - 1

    daily_risk_free = (
        (1 + RISK_FREE_RATE) ** (1 / 252) - 1
    )

    sharpe = (
        (daily_returns - daily_risk_free).mean()
        / daily_returns.std()
        * np.sqrt(252)
        if len(daily_returns) > 1
        and daily_returns.std() > 0
        else np.nan
    )

    return {
        "final_value": float(values.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()),
        "sharpe_2pct_rf": float(sharpe),
    }


def main():
    print("TRAINING FINAL MODEL")

    development = load_development_data()

    model, feature_lower, feature_upper = fit_final_model(
        development
    )

    del development

    print("Model trained.")

    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("\nLOADING BENCHMARK AND TRADING CALENDAR")

    benchmark_columns = [
        row[1]
        for row in con.execute(
            "PRAGMA table_info('benchmark')"
        ).fetchall()
    ]

    benchmark_price_column = (
        "adj_close"
        if "adj_close" in benchmark_columns
        else "close"
    )

    benchmark = con.execute(
        f"""
        SELECT date,
               {benchmark_price_column} AS benchmark_price
        FROM benchmark
        WHERE date BETWEEN ? AND ?
        ORDER BY date
        """,
        [START, END]
    ).df()

    benchmark["date"] = pd.to_datetime(
        benchmark["date"]
    )

    benchmark = (
        benchmark[
            benchmark["benchmark_price"] > 0
        ]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )

    dates = pd.DatetimeIndex(benchmark["date"])

    assert len(dates) > 1
    assert dates[0] == START
    assert dates[-1] == END

    benchmark["benchmark_value"] = (
        INITIAL_CAPITAL
        * benchmark["benchmark_price"]
        / benchmark["benchmark_price"].iloc[0]
    )

    print("Trading sessions:", len(dates))

    print("\nLOADING FEATURES USING TRAINING DEFINITIONS")

    # Match model.py:
    # - calculate lags before filtering invalid prices
    # - use 20 valid daily returns for volatility
    # - use the complete historical lookback
    # - do not use future prices for features

    features = con.execute(
        """
        WITH daily_prices AS (
            SELECT
                symbol,
                date,
                adj_close,
                LAG(adj_close) OVER (
                    PARTITION BY symbol
                    ORDER BY date
                ) AS previous_close
            FROM prices
            WHERE date <= DATE '2026-09-30'
        ),
        price_features AS (
            SELECT
                symbol,
                date,
                adj_close,

                LAG(adj_close, 20) OVER (
                    PARTITION BY symbol
                    ORDER BY date
                ) AS price_20d_ago,

                LAG(adj_close, 60) OVER (
                    PARTITION BY symbol
                    ORDER BY date
                ) AS price_60d_ago,

                STDDEV_SAMP(
                    CASE
                        WHEN adj_close > 0
                         AND previous_close > 0
                        THEN adj_close / previous_close - 1
                        ELSE NULL
                    END
                ) OVER (
                    PARTITION BY symbol
                    ORDER BY date
                    ROWS BETWEEN 19 PRECEDING
                             AND CURRENT ROW
                ) AS volatility_20d,

                COUNT(
                    CASE
                        WHEN adj_close > 0
                         AND previous_close > 0
                        THEN 1
                        ELSE NULL
                    END
                ) OVER (
                    PARTITION BY symbol
                    ORDER BY date
                    ROWS BETWEEN 19 PRECEDING
                             AND CURRENT ROW
                ) AS valid_return_count

            FROM daily_prices
        )
        SELECT
            symbol,
            date AS feature_date,
            adj_close / price_20d_ago - 1 AS return_20d,
            adj_close / price_60d_ago - 1 AS return_60d,
            volatility_20d
        FROM price_features
        WHERE date BETWEEN ? AND ?
          AND adj_close > 0
          AND price_20d_ago > 0
          AND price_60d_ago > 0
          AND valid_return_count = 20
          AND volatility_20d IS NOT NULL
        """,
        [START, END]
    ).df()

    features["feature_date"] = pd.to_datetime(
        features["feature_date"]
    )

    print("Feature rows:", len(features))

    print("\nLOADING HISTORICAL UNIVERSE")

    membership = con.execute(
        """
        SELECT
            symbol,
            quarter_start,
            quarter_end,
            selection_date,
            effective_start
        FROM universe_membership
        WHERE quarter_end >= ?
          AND quarter_start <= ?
        """,
        [START, END]
    ).df()

    for column in [
        "quarter_start",
        "quarter_end",
        "selection_date",
        "effective_start",
    ]:
        membership[column] = pd.to_datetime(
            membership[column]
        )

    features["quarter"] = (
        features["feature_date"].dt.to_period("Q")
    )

    membership["quarter"] = (
        membership["quarter_start"].dt.to_period("Q")
    )

    eligible = features.merge(
        membership,
        on=["symbol", "quarter"],
        how="inner"
    )

    eligible = eligible[
        (eligible["selection_date"] <= eligible["feature_date"])
        & (eligible["effective_start"] <= eligible["feature_date"])
        & (eligible["quarter_start"] <= eligible["feature_date"])
        & (eligible["quarter_end"] >= eligible["feature_date"])
    ].copy()

    eligible = eligible.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna(subset=FEATURES)

    eligible_by_date = {
        pd.Timestamp(day): group
        for day, group in eligible.groupby("feature_date")
    }

    print("Eligible feature rows:", len(eligible))

    print("\nLOADING PRICES")

    prices = con.execute(
        """
        SELECT symbol, date, adj_close
        FROM prices
        WHERE date BETWEEN ? AND ?
          AND adj_close > 0
        ORDER BY symbol, date
        """,
        [START, END]
    ).df()

    con.close()

    prices["date"] = pd.to_datetime(prices["date"])

    price_lookup = {}

    for symbol, group in prices.groupby(
        "symbol", sort=False
    ):
        price_lookup[symbol] = pd.Series(
            group["adj_close"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(group["date"])
        )

    del prices

    def price_on(symbol, date):
        series = price_lookup.get(symbol)

        if series is None or date not in series.index:
            return None

        value = series.loc[date]

        if isinstance(value, pd.Series):
            value = value.iloc[-1]

        value = float(value)

        if not np.isfinite(value) or value <= 0:
            return None

        return value

    def latest_price(symbol, date):
        series = price_lookup.get(symbol)

        if series is None:
            return None, None

        index = series.index.searchsorted(
            date, side="right"
        ) - 1

        if index < 0:
            return None, None

        return (
            float(series.iloc[index]),
            pd.Timestamp(series.index[index])
        )

    print("\nRUNNING BACKTEST")

    cash = INITIAL_CAPITAL
    positions = {}
    pending_orders = []

    trades = []
    daily_rows = []
    accounting_rows = []
    events = []
    open_position_rows = []

    cycle = 0

    for i, date in enumerate(dates):
        date = pd.Timestamp(date)

        # Sell positions that have reached maturity.
        for symbol, position in list(positions.items()):
            if date < position["maturity_date"]:
                continue

            exit_price = price_on(symbol, date)

            if exit_price is None:
                continue

            proceeds = position["shares"] * exit_price
            cash += proceeds

            trades.append({
                "cycle": position["cycle"],
                "symbol": symbol,
                "signal_date": position["signal_date"],
                "entry_date": position["entry_date"],
                "maturity_date": position["maturity_date"],
                "exit_date": date,
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "shares": position["shares"],
                "allocation": position["allocation"],
                "proceeds": proceeds,
                "predicted_return": position["prediction"],
                "realized_return": (
                    proceeds / position["allocation"] - 1
                ),
                "exit_delayed": (
                    date > position["maturity_date"]
                ),
            })

            del positions[symbol]

        # Execute yesterday's orders at today's close.
        # No future price was used to choose the stocks.
        if pending_orders:
            orders = pending_orders
            pending_orders = []

            allocation = cash / len(orders)

            for order in orders:
                symbol = order["symbol"]

                if symbol in positions:
                    events.append({
                        "date": date,
                        "symbol": symbol,
                        "event": "already_held",
                    })
                    continue

                entry_price = price_on(symbol, date)

                if entry_price is None:
                    events.append({
                        "date": date,
                        "symbol": symbol,
                        "event": "missing_entry_price",
                    })
                    continue

                if allocation > cash + 0.000001:
                    raise AssertionError(
                        "Attempted to overspend cash."
                    )

                shares = allocation / entry_price
                cash -= allocation

                maturity_target = (
                    date + pd.Timedelta(days=HOLDING_DAYS)
                )

                maturity_index = dates.searchsorted(
                    maturity_target
                )

                maturity_date = (
                    pd.Timestamp(dates[maturity_index])
                    if maturity_index < len(dates)
                    else maturity_target
                )

                positions[symbol] = {
                    "cycle": order["cycle"],
                    "signal_date": order["signal_date"],
                    "entry_date": date,
                    "maturity_date": maturity_date,
                    "entry_price": entry_price,
                    "shares": shares,
                    "allocation": allocation,
                    "prediction": order["prediction"],
                }

        # Calculate daily portfolio value.
        holdings_value = 0.0
        stale_count = 0
        unpriced_count = 0

        for symbol, position in positions.items():
            current_price = price_on(symbol, date)

            if current_price is None:
                current_price, last_quote_date = latest_price(
                    symbol, date
                )

                if current_price is None:
                    unpriced_count += 1
                    continue

                stale_count += 1

            holdings_value += (
                position["shares"] * current_price
            )

        portfolio_value = cash + holdings_value

        assert cash >= -0.000001
        assert abs(
            portfolio_value - cash - holdings_value
        ) < 0.01

        daily_rows.append({
            "date": date,
            "cash": cash,
            "holdings_value": holdings_value,
            "portfolio_value": portfolio_value,
            "open_positions": len(positions),
            "stale_price_positions": stale_count,
            "unpriced_positions": unpriced_count,
        })

        accounting_rows.append({
            "date": date,
            "cash_nonnegative": cash >= -0.000001,
            "portfolio_identity_valid": abs(
                portfolio_value - cash - holdings_value
            ) < 0.01,
            "stale_price_positions": stale_count,
            "unpriced_positions": unpriced_count,
        })

        # Make a decision after today's close.
        # Execute the resulting orders next session.
        if (
            not positions
            and not pending_orders
            and i + 1 < len(dates)
        ):
            candidates = eligible_by_date.get(date)

            if candidates is None or candidates.empty:
                continue

            X = candidates[FEATURES].clip(
                lower=feature_lower,
                upper=feature_upper,
                axis=1
            )

            scored = candidates.copy()
            scored["prediction"] = model.predict(X)

            selected = (
                scored.sort_values(
                    "prediction",
                    ascending=False
                )
                .head(NUM_STOCKS)
            )

            if selected.empty:
                continue

            cycle += 1

            pending_orders = [
                {
                    "cycle": cycle,
                    "symbol": row.symbol,
                    "prediction": float(row.prediction),
                    "signal_date": date,
                }
                for row in selected.itertuples()
            ]

            print(
                f"Cycle {cycle:3d} | "
                f"Signal: {date.date()} | "
                f"Selected: {len(pending_orders)}"
            )

    # Include open positions at the final test close.
    for symbol, position in positions.items():
        final_price, quote_date = latest_price(
            symbol, END
        )

        open_position_rows.append({
            "symbol": symbol,
            "entry_date": position["entry_date"],
            "maturity_date": position["maturity_date"],
            "shares": position["shares"],
            "entry_price": position["entry_price"],
            "last_price": final_price,
            "last_quote_date": quote_date,
            "market_value": (
                position["shares"] * final_price
                if final_price is not None
                else np.nan
            ),
            "stale_at_end": quote_date != END,
        })

    history = pd.DataFrame(daily_rows)
    trades_df = pd.DataFrame(trades)
    checks_df = pd.DataFrame(accounting_rows)

    open_df = pd.DataFrame(
        open_position_rows,
        columns=[
            "symbol", "entry_date", "maturity_date",
            "shares", "entry_price", "last_price",
            "last_quote_date", "market_value",
            "stale_at_end"
        ]
    )

    events_df = pd.DataFrame(
        events,
        columns=["date", "symbol", "event"]
    )

    comparison = history.merge(
        benchmark[["date", "benchmark_value"]],
        on="date",
        how="left"
    )

    if comparison["benchmark_value"].isna().any():
        raise RuntimeError(
            "Benchmark missing on a trading date."
        )

    strategy_metrics = performance_metrics(
        comparison,
        "portfolio_value"
    )

    benchmark_metrics = performance_metrics(
        comparison,
        "benchmark_value"
    )

    metrics_df = pd.DataFrame([
        {
            "metric": key,
            "strategy": strategy_metrics[key],
            "benchmark": benchmark_metrics[key],
        }
        for key in strategy_metrics
    ])

    # Timing tests.
    assert history["date"].iloc[0] == START
    assert history["date"].iloc[-1] == END
    assert checks_df["cash_nonnegative"].all()
    assert checks_df["portfolio_identity_valid"].all()

    if not trades_df.empty:
        assert (
            trades_df["entry_date"]
            > trades_df["signal_date"]
        ).all()

        assert (
            trades_df["exit_date"]
            >= trades_df["maturity_date"]
        ).all()

        assert (
            trades_df["exit_date"]
            >= trades_df["entry_date"]
            + pd.Timedelta(days=HOLDING_DAYS)
        ).all()

    if not open_df.empty:
        assert (
            open_df["entry_date"] <= END
        ).all()

    # Output files.
    trades_df.to_csv(
        HERE / "backtest_trades.csv", index=False
    )

    history.to_csv(
        HERE / "backtest_portfolio.csv", index=False
    )

    comparison.to_csv(
        HERE / "backtest_comparison.csv", index=False
    )

    metrics_df.to_csv(
        HERE / "backtest_metrics.csv", index=False
    )

    checks_df.to_csv(
        HERE / "backtest_accounting_checks.csv",
        index=False
    )

    events_df.to_csv(
        HERE / "backtest_events.csv", index=False
    )

    open_df.to_csv(
        HERE / "backtest_open_positions.csv",
        index=False
    )

    # Cumulative-return chart.
    plt.figure(figsize=(12, 6))

    plt.plot(
        comparison["date"],
        comparison["portfolio_value"] / INITIAL_CAPITAL - 1,
        label="Ridge strategy"
    )

    plt.plot(
        comparison["date"],
        comparison["benchmark_value"] / INITIAL_CAPITAL - 1,
        label="S&P 500 Total Return"
    )

    plt.title(
        "HW4: Strategy vs. S&P 500 Total Return"
    )
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        HERE / "backtest_performance.png",
        dpi=150
    )

    plt.close()

    print("\n" + "=" * 60)
    print("FINAL BACKTEST RESULTS")
    print("=" * 60)

    print("Cycles initiated:", cycle)
    print("Completed trades:", len(trades_df))
    print("Final date:", history["date"].iloc[-1].date())
    print("Open positions:", len(open_df))
    print("Final cash:", round(cash, 2))

    print("\nMETRICS")
    print(metrics_df.to_string(index=False))

    print("\nACCOUNTING CHECKS")
    print(
        "Nonnegative cash:",
        checks_df["cash_nonnegative"].all()
    )
    print(
        "Portfolio identity:",
        checks_df["portfolio_identity_valid"].all()
    )

    print(
        "Maximum stale-price positions:",
        checks_df["stale_price_positions"].max()
    )

    print(
        "Stale positions at test end:",
        int(open_df["stale_at_end"].sum())
        if not open_df.empty else 0
    )

    print(
        "Missing-entry events:",
        len(events_df)
    )

    print("\nAll output files saved.")


if __name__ == "__main__":
    main()
