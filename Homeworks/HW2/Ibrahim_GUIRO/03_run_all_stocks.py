"""Generate stock reports from the cleaned market database."""

from datetime import date
import logging
from pathlib import Path
import sqlite3

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import papermill as pm
from tqdm import tqdm


# Resolve paths relative to this script, regardless of the terminal directory.
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "data.db"
TEMPLATE_PATH = BASE_DIR / "02_stock_report_template.ipynb"


def load_batch_parameters(database_path):
    """Read available tickers and calculate the common analysis period."""
    if not database_path.is_file():
        raise FileNotFoundError(f"Database not found: {database_path}")

    database_uri = database_path.as_uri() + "?mode=ro"

    with sqlite3.connect(database_uri, uri=True) as connection:
        tickers = pd.read_sql_query(
            """
            SELECT DISTINCT ticker
            FROM ohlc_hw_cleaned
            ORDER BY ticker
            """,
            connection,
        )["ticker"].tolist()

        latest_date_value = connection.execute(
            "SELECT MAX(ts) FROM ohlc_hw_cleaned"
        ).fetchone()[0]

    if not tickers or latest_date_value is None:
        raise ValueError("The cleaned dataset is empty.")

    latest_date = pd.Timestamp(latest_date_value)

    # Exclude the month containing the latest available observation.
    latest_month_start = latest_date.normalize().replace(day=1)
    period_start = latest_month_start - pd.DateOffset(months=6)
    period_end = latest_month_start - pd.Timedelta(days=1)

    return (
        tickers,
        period_start.strftime("%Y-%m-%d"),
        period_end.strftime("%Y-%m-%d"),
        latest_date,
    )


def extract_report_results(notebook):
    """Retrieve the named JSON output from an executed notebook."""
    results = []

    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            payload = output.get("data", {}).get("application/json")

            if (
                isinstance(payload, dict)
                and payload.get("output_name") == "stock_report_results"
            ):
                results.append(dict(payload))

    if len(results) != 1:
        raise ValueError(
            f"Expected one stock_report_results output; found {len(results)}."
        )

    return results[0]


def create_portfolio_summary(summary, output_dir, start_date, end_date):
    """Print Sharpe rankings and save the portfolio risk-return chart."""
    ranked = summary.loc[
        np.isfinite(summary["sharpe_ratio"])
    ].sort_values("sharpe_ratio", ascending=False)

    ranking_columns = [
        "ticker",
        "sharpe_ratio",
        "total_return",
        "avg_volatility",
    ]

    print("\nTop 10 stocks by Sharpe ratio:")
    print(
        ranked.head(10)[ranking_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nBottom 10 stocks by Sharpe ratio:")
    print(
        ranked.tail(10).sort_values("sharpe_ratio")[
            ranking_columns
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    chart_columns = ["avg_volatility", "total_return", "sharpe_ratio"]
    eligible = summary.loc[
        np.isfinite(summary[chart_columns]).all(axis=1)
    ].copy()

    if eligible.empty:
        print("No stocks have all metrics required for the chart.")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    points = ax.scatter(
        eligible["avg_volatility"],
        eligible["total_return"],
        c=eligible["sharpe_ratio"],
        cmap="viridis",
        s=90,
        edgecolors="black",
        linewidths=0.5,
        alpha=0.85,
    )

    # Label the five highest total returns among plotted stocks.
    for _, row in eligible.nlargest(5, "total_return").iterrows():
        ax.annotate(
            row["ticker"],
            (row["avg_volatility"], row["total_return"]),
            xytext=(6, 6),
            textcoords="offset points",
        )

    fig.colorbar(points, ax=ax, label="Sharpe ratio")

    ax.set_title(
        "Risk-Return Profile of Portfolio\n"
        f"Requested period: {start_date} to {end_date}"
    )
    ax.set_xlabel("Mean annualized 20-session rolling volatility")
    ax.set_ylabel("Total return over available observations")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.grid(alpha=0.25)

    fig.tight_layout()

    chart_path = output_dir / "risk_return_profile.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nRisk-return chart saved to: {chart_path}")
    print("Actual coverage for each stock is recorded in portfolio_summary.csv.")


def main():
    """Execute each stock report and collect successful results."""
    if not TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"Notebook not found: {TEMPLATE_PATH}")

    tickers, start_date, end_date, latest_date = load_batch_parameters(
        DB_PATH
    )

    output_dir = BASE_DIR / "reports" / date.today().isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=output_dir / "execution.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )

    print(f"Latest database date: {latest_date.date()}")
    print(f"Analysis period: {start_date} to {end_date}")
    print(f"Stocks to process: {len(tickers)}")
    print(f"Reports directory: {output_dir}")

    successful_results = []
    failed_stocks = []

    for ticker in tqdm(tickers, desc="Generating reports", unit="stock"):
        report_path = output_dir / f"{ticker}_report.ipynb"

        try:
            executed_notebook = pm.execute_notebook(
                input_path=str(TEMPLATE_PATH),
                output_path=str(report_path),
                parameters={
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    # Relative to cwd below, so no local path ends up in reports.
                    "db_path": DB_PATH.relative_to(BASE_DIR).as_posix(),
                },
                kernel_name="python3",
                cwd=str(BASE_DIR),
                progress_bar=False,
                execution_timeout=300,
            )

            results = extract_report_results(executed_notebook)

            if (
                results["ticker"] != ticker
                or results["requested_start"] != start_date
                or results["requested_end"] != end_date
            ):
                raise ValueError("Report parameters do not match the batch.")

            results.pop("output_name")
            results["report_path"] = report_path.relative_to(
                BASE_DIR
            ).as_posix()

            successful_results.append(results)
            logging.info("Report completed: %s", ticker)

        except Exception as error:
            failed_stocks.append(ticker)
            logging.exception("Report failed: %s", ticker)
            tqdm.write(f"Failed: {ticker} ({type(error).__name__})")

    # Record failures from this execution, including an empty file if none.
    failure_path = output_dir / "failed_stocks.txt"
    failure_path.write_text(
        "".join(f"{ticker}\n" for ticker in failed_stocks),
        encoding="utf-8",
    )

    # Collect only results returned successfully during this execution.
    columns = [
        "ticker",
        "avg_daily_return",
        "avg_volatility",
        "max_drawdown",
        "sharpe_ratio",
        "total_return",
        "report_path",
        "requested_start",
        "requested_end",
        "actual_start",
        "actual_end",
        "observation_count",
        "eligible_return_count",
    ]

    summary = pd.DataFrame(successful_results, columns=columns)

    # Undefined metrics are stored as None; make sure the columns stay numeric.
    metric_columns = columns[1:6]
    summary[metric_columns] = summary[metric_columns].apply(pd.to_numeric)
    summary = summary.sort_values(
        "sharpe_ratio", ascending=False, na_position="last"
    )

    summary_path = output_dir / "portfolio_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\nSuccessfully generated {len(summary)}/{len(tickers)} reports")
    print(f"Failed reports: {len(failed_stocks)}")
    print(f"Summary saved to: {summary_path}")

    if summary.empty:
        print("No successful reports. Inspect execution.log for details.")
    else:
        create_portfolio_summary(
            summary, output_dir, start_date, end_date
        )


if __name__ == "__main__":
    main()