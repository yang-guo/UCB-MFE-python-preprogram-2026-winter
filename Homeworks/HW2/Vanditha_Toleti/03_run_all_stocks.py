
from pathlib import Path
from datetime import datetime
import sqlite3
import warnings

import numpy as np
import pandas as pd
import papermill as pm
import nbformat
import matplotlib.pyplot as plt
from tqdm import tqdm


# --------------------------------------------------
# 1. Paths and parameters
# --------------------------------------------------

HOMEWORK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = HOMEWORK_DIR.parents[2]

TEMPLATE_PATH = HOMEWORK_DIR / "02_stock_report_template.ipynb"
DB_PATH = PROJECT_ROOT / "course_file_export" / "data.db"

REPORTS_DIR = HOMEWORK_DIR / "reports"
RUN_DATE = datetime.now().strftime("%Y-%m-%d")
RUN_DIR = REPORTS_DIR / RUN_DATE
RUN_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2024-06-01"
END_DATE = "2024-11-30"
ROLLING_WINDOW = 21
RISK_FREE_RATE = 0.0


# --------------------------------------------------
# 2. Get all tickers from the cleaned database
# --------------------------------------------------

with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
    ticker_df = pd.read_sql_query(
        """
        SELECT DISTINCT ticker
        FROM ohlc_hw_cleaned
        WHERE ticker IS NOT NULL
        ORDER BY ticker
        """,
        conn,
    )

tickers = ticker_df["ticker"].dropna().tolist()

if not tickers:
    raise ValueError("No tickers found in ohlc_hw_cleaned.")

print(f"Tickers found: {len(tickers)}")
print(f"Requested period: {START_DATE} to {END_DATE}")
print(f"Output directory: {RUN_DIR}")


# --------------------------------------------------
# 3. Run the report notebook for every ticker
# --------------------------------------------------

results = []

for stock_ticker in tqdm(tickers, desc="Generating stock reports"):
    output_path = RUN_DIR / f"{stock_ticker}_report.ipynb"

    try:
        pm.execute_notebook(
            input_path=str(TEMPLATE_PATH),
            output_path=str(output_path),
            parameters={
                "ticker": stock_ticker,
                "start_date": START_DATE,
                "end_date": END_DATE,
                "rolling_window": ROLLING_WINDOW,
                "risk_free_rate": RISK_FREE_RATE,
            },
            progress_bar=False,
            log_output=True,
        )

        results.append({
            "ticker": stock_ticker,
            "status": "success",
            "report_path": str(output_path),
            "error": "",
        })

    except Exception as exc:
        results.append({
            "ticker": stock_ticker,
            "status": "failed",
            "report_path": "",
            "error": str(exc)[:500],
        })

results_df = pd.DataFrame(results)
results_path = RUN_DIR / "run_results.csv"
results_df.to_csv(results_path, index=False)

print("\nBatch execution finished.")
print(results_df["status"].value_counts())
print(f"Run log saved to: {results_path}")


# --------------------------------------------------
# 4. Calculate portfolio metrics from the database
# --------------------------------------------------
# Calculate directly from cleaned prices so the summary
# does not depend on notebook display formatting or paths.

summary_rows = []

successful_tickers = results_df.loc[
    results_df["status"] == "success", "ticker"
].tolist()

with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:

    for stock_ticker in successful_tickers:

        stock_df = pd.read_sql_query(
            """
            SELECT ts, ticker, adj_close
            FROM ohlc_hw_cleaned
            WHERE ticker = ?
              AND ts >= ?
              AND ts <= ?
            ORDER BY ts
            """,
            conn,
            params=(stock_ticker, START_DATE, END_DATE),
            parse_dates=["ts"],
        )

        if stock_df.empty:
            warnings.warn(
                f"No data available for {stock_ticker}; skipping summary."
            )
            continue

        stock_df["adj_close"] = pd.to_numeric(
            stock_df["adj_close"], errors="coerce"
        )
        stock_df = stock_df.dropna(subset=["adj_close"])
        stock_df = stock_df[
            np.isfinite(stock_df["adj_close"])
            & (stock_df["adj_close"] > 0)
        ].copy()

        if stock_df.empty:
            continue

        stock_df = stock_df.sort_values("ts").drop_duplicates(
            subset=["ts"], keep="last"
        )

        # Daily log returns
        stock_df["log_return"] = np.log(
            stock_df["adj_close"] / stock_df["adj_close"].shift(1)
        )

        # Rolling annualized volatility
        stock_df["rolling_volatility"] = (
            stock_df["log_return"]
            .rolling(ROLLING_WINDOW)
            .std()
            * np.sqrt(252)
        )

        # Drawdown from running peak
        stock_df["running_peak"] = stock_df["adj_close"].cummax()
        stock_df["drawdown"] = (
            stock_df["adj_close"] / stock_df["running_peak"] - 1
        )

        daily_returns = stock_df["log_return"].dropna()

        if len(daily_returns) >= 2:
            annualized_vol = daily_returns.std(ddof=1) * np.sqrt(252)

            if annualized_vol > 0 and np.isfinite(annualized_vol):
                sharpe_ratio = (
                    daily_returns.mean() * 252 - RISK_FREE_RATE
                ) / annualized_vol
            else:
                sharpe_ratio = np.nan
        else:
            sharpe_ratio = np.nan

        total_return = (
            stock_df["adj_close"].iloc[-1]
            / stock_df["adj_close"].iloc[0]
            - 1
        )

        summary_rows.append({
            "ticker": stock_ticker,
            "avg_daily_return": daily_returns.mean(),
            "avg_volatility": stock_df["rolling_volatility"].mean(),
            "max_drawdown": stock_df["drawdown"].min(),
            "sharpe_ratio": sharpe_ratio,
            "total_return": total_return,
            "report_path": str(
                RUN_DIR / f"{stock_ticker}_report.ipynb"
            ),
        })


# --------------------------------------------------
# 5. Save portfolio summary
# --------------------------------------------------

summary_df = pd.DataFrame(
    summary_rows,
    columns=[
        "ticker",
        "avg_daily_return",
        "avg_volatility",
        "max_drawdown",
        "sharpe_ratio",
        "total_return",
        "report_path",
    ],
)

summary_path = RUN_DIR / "portfolio_summary.csv"
summary_df.to_csv(summary_path, index=False)

print(f"\nPortfolio summary saved to: {summary_path}")
print(f"Successful reports: {len(successful_tickers)}")
print(f"Summary rows: {len(summary_df)}")

if summary_df.empty:
    raise RuntimeError(
        "No portfolio summary rows were generated. "
        "Check run_results.csv and the notebook errors."
    )


# --------------------------------------------------
# 6. Top and bottom 10 by Sharpe ratio
# --------------------------------------------------

ranked_df = summary_df.dropna(subset=["sharpe_ratio"]).copy()
ranked_df = ranked_df.sort_values(
    "sharpe_ratio", ascending=False
)

top_10 = ranked_df.head(10)
bottom_10 = ranked_df.tail(10).sort_values(
    "sharpe_ratio", ascending=True
)

top_10_path = RUN_DIR / "top_10_sharpe.csv"
bottom_10_path = RUN_DIR / "bottom_10_sharpe.csv"

top_10.to_csv(top_10_path, index=False)
bottom_10.to_csv(bottom_10_path, index=False)

print("\nTop 10 stocks by Sharpe ratio:")
print(top_10[["ticker", "sharpe_ratio", "total_return"]].to_string(index=False))

print("\nBottom 10 stocks by Sharpe ratio:")
print(
    bottom_10[
        ["ticker", "sharpe_ratio", "total_return"]
    ].to_string(index=False)
)


# --------------------------------------------------
# 7. Risk-return scatter plot
# --------------------------------------------------

plot_df = summary_df.dropna(
    subset=["avg_volatility", "total_return", "sharpe_ratio"]
).copy()

if not plot_df.empty:
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(
        plot_df["avg_volatility"],
        plot_df["total_return"],
        c=plot_df["sharpe_ratio"],
        cmap="viridis",
        s=85,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.4,
    )

    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Annualized Sharpe Ratio")

    # Annotate the five stocks with the highest total returns
    top_return_stocks = plot_df.nlargest(5, "total_return")

    for _, row in top_return_stocks.iterrows():
        ax.annotate(
            row["ticker"],
            (row["avg_volatility"], row["total_return"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Average Rolling Annualized Volatility")
    ax.set_ylabel("Total Return")
    ax.set_title("Risk-Return Profile of Portfolio")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    plot_path = RUN_DIR / "risk_return_profile.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nRisk-return plot saved to: {plot_path}")
else:
    print("\nNo valid rows available for the risk-return plot.")


# --------------------------------------------------
# 8. Final output
# --------------------------------------------------

print("\nAll done!")
print(f"Reports directory: {RUN_DIR}")
print(f"Portfolio summary: {summary_path}")
print(f"Top 10 Sharpe: {top_10_path}")
print(f"Bottom 10 Sharpe: {bottom_10_path}")
