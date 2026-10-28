"""Generate single-stock reports and a portfolio summary for HW2."""

from contextlib import closing
from datetime import date
from pathlib import Path
import sqlite3
from matplotlib.ticker import PercentFormatter

import matplotlib

# Use a noninteractive backend when the script runs from the terminal.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

HOMEWORK_DIR = Path(__file__).resolve().parent
DATABASE_PATH = HOMEWORK_DIR / "data" / "data.db"
TEMPLATE_PATH = HOMEWORK_DIR / "02_stock_report_template.ipynb"
REPORTS_ROOT = HOMEWORK_DIR / "reports"

RUN_DATE = date.today().isoformat()


def validate_paths() -> None:
    """Verify that the database and report template are available."""
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(f"Database not found: {DATABASE_PATH}")

    if not TEMPLATE_PATH.is_file():
        raise FileNotFoundError(f"Report template not found: {TEMPLATE_PATH}")


def load_database_inputs() -> tuple[list[str], pd.Timestamp]:
    """Load all cleaned tickers and the latest database observation date."""
    database_uri = DATABASE_PATH.as_uri() + "?mode=ro"

    with closing(sqlite3.connect(database_uri, uri=True)) as connection:
        ticker_data = pd.read_sql_query(
            """
            SELECT DISTINCT ticker
            FROM ohlc_hw_cleaned
            WHERE ticker IS NOT NULL
            ORDER BY ticker
            """,
            connection,
        )

        latest_date_value = connection.execute("""
            SELECT MAX(ts)
            FROM ohlc_hw_cleaned
            """).fetchone()[0]

    tickers = ticker_data["ticker"].tolist()

    if not tickers:
        raise ValueError("The cleaned table does not contain any tickers.")

    if latest_date_value is None:
        raise ValueError("The cleaned table does not contain any dates.")

    latest_date = pd.Timestamp(latest_date_value)

    return tickers, latest_date


def determine_analysis_window(
    latest_date: pd.Timestamp,
) -> tuple[str, str]:
    """Return the six complete months before the latest observation month."""
    latest_month_start = pd.Timestamp(
        year=latest_date.year,
        month=latest_date.month,
        day=1,
    )

    start_date = latest_month_start - pd.DateOffset(months=6)

    end_date = latest_month_start - pd.Timedelta(days=1)

    return (
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )


def generate_reports(
    tickers: list[str],
    start_date: str,
    end_date: str,
    output_directory: Path,
) -> tuple[list[Path], dict[str, str]]:
    """Execute the report template once for each ticker."""
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful_reports = []
    failures = {}

    for ticker in tqdm(
        tickers,
        desc="Generating reports",
    ):
        report_path = output_directory / f"{ticker}_report.ipynb"

        try:
            pm.execute_notebook(
                input_path=str(TEMPLATE_PATH),
                output_path=str(report_path),
                parameters={
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "db_path": str(DATABASE_PATH.resolve()),
                },
                cwd=str(HOMEWORK_DIR),
                progress_bar=False,
                log_output=False,
            )

            successful_reports.append(report_path)

        except Exception as error:
            failures[ticker] = f"{type(error).__name__}: {error}"

    failed_file = output_directory / "failed_stocks.txt"

    failed_file.write_text(
        "\n".join(
            f"{ticker}: {message}" for ticker, message in failures.items()
        ),
        encoding="utf-8",
    )

    print(
        f"Successfully generated "
        f"{len(successful_reports)}/{len(tickers)} reports"
    )

    if failures:
        print(f"Failed tickers saved to: " f"{failed_file}")
    else:
        print("No report failures.")

    return successful_reports, failures


SUMMARY_COLUMNS = [
    "ticker",
    "avg_daily_return",
    "avg_volatility",
    "max_drawdown",
    "sharpe_ratio",
    "total_return",
    "report_path",
]


def collect_portfolio_summary(
    report_paths: list[Path],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Collect named metrics from successfully executed reports."""
    summary_rows = []
    collection_failures = {}

    for report_path in tqdm(
        report_paths,
        desc="Collecting results",
    ):
        try:
            notebook = sb.read_notebook(str(report_path))

            if not bool(notebook.scraps["data_quality_ok"].data):
                raise ValueError("Report data-quality checks failed.")

            summary_rows.append(
                {
                    "ticker": notebook.scraps["ticker"].data,
                    "avg_daily_return": notebook.scraps[
                        "avg_daily_return"
                    ].data,
                    "avg_volatility": notebook.scraps["avg_volatility"].data,
                    "max_drawdown": notebook.scraps["max_drawdown"].data,
                    "sharpe_ratio": notebook.scraps["sharpe_ratio"].data,
                    "total_return": notebook.scraps["total_return"].data,
                    "report_path": str(report_path.relative_to(HOMEWORK_DIR)),
                }
            )

        except Exception as error:
            collection_failures[
                report_path.stem.removesuffix("_report")
            ] = (
                f"{type(error).__name__}: {error}"
            )

    summary = pd.DataFrame(
        summary_rows,
        columns=SUMMARY_COLUMNS,
    )

    if not summary.empty:
        numeric_columns = [
            "avg_daily_return",
            "avg_volatility",
            "max_drawdown",
            "sharpe_ratio",
            "total_return",
        ]

        summary[numeric_columns] = summary[numeric_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )

        summary = summary.sort_values(
            "sharpe_ratio",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    return summary, collection_failures


def print_sharpe_rankings(
    summary: pd.DataFrame,
) -> None:
    """Print the top and bottom stocks with defined Sharpe ratios."""
    valid_rankings = summary.dropna(subset=["sharpe_ratio"])

    if valid_rankings.empty:
        print("No defined Sharpe ratios are available " "for ranking.")
        return

    ranking_columns = [
        "ticker",
        "sharpe_ratio",
        "total_return",
        "avg_volatility",
    ]

    top_stocks = valid_rankings.nlargest(
        10,
        "sharpe_ratio",
    )[ranking_columns]

    bottom_stocks = valid_rankings.nsmallest(
        10,
        "sharpe_ratio",
    )[ranking_columns]

    formatters = {
        "sharpe_ratio": "{:.2f}".format,
        "total_return": "{:.2%}".format,
        "avg_volatility": "{:.2%}".format,
    }

    print("\nTop stocks by Sharpe ratio:")
    print(
        top_stocks.to_string(
            index=False,
            formatters=formatters,
        )
    )

    print("\nBottom stocks by Sharpe ratio:")
    print(
        bottom_stocks.to_string(
            index=False,
            formatters=formatters,
        )
    )


def save_portfolio_outputs(
    summary: pd.DataFrame,
    output_directory: Path,
) -> tuple[Path | None, Path | None]:
    """Save the portfolio summary CSV and risk-return visualization."""
    if summary.empty:
        print(
            "No successful report results were available; "
            "portfolio outputs were not created."
        )
        return None, None

    summary_path = output_directory / "portfolio_summary.csv"

    summary.to_csv(
        summary_path,
        index=False,
    )

    plot_columns = [
        "avg_volatility",
        "total_return",
        "sharpe_ratio",
    ]

    finite_plot_values = np.isfinite(summary[plot_columns]).all(axis=1)

    plot_data = summary.loc[finite_plot_values].copy()

    if plot_data.empty:
        print(
            "Portfolio summary saved, but no finite "
            "risk-return observations were available to plot."
        )
        return summary_path, None

    figure, axis = plt.subplots(
        figsize=(10, 7),
        layout="constrained",
    )

    scatter = axis.scatter(
        plot_data["avg_volatility"],
        plot_data["total_return"],
        c=plot_data["sharpe_ratio"],
        cmap="viridis",
        s=70,
        alpha=0.8,
        edgecolor="black",
    )

    colorbar = figure.colorbar(
        scatter,
        ax=axis,
    )
    colorbar.set_label("Sharpe ratio")

    top_returns = plot_data.nlargest(
        5,
        "total_return",
    )

    for row in top_returns.itertuples():
        axis.annotate(
            row.ticker,
            (
                row.avg_volatility,
                row.total_return,
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    axis.set_title("Risk-Return Profile of Portfolio")
    axis.set_xlabel("Average annualized rolling volatility")
    axis.set_ylabel("Total return")
    axis.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axis.grid(True, alpha=0.3)

    plot_path = output_directory / "portfolio_risk_return.png"

    figure.savefig(
        plot_path,
        dpi=150,
    )
    plt.close(figure)

    print(f"Portfolio summary saved to: {summary_path}")
    print(f"Portfolio chart saved to: {plot_path}")

    return summary_path, plot_path

def main() -> None:
    """Run the complete multi-stock reporting pipeline."""
    validate_paths()

    tickers, latest_date = load_database_inputs()

    start_date, end_date = (
        determine_analysis_window(
            latest_date
        )
    )

    output_directory = (
        REPORTS_ROOT
        / RUN_DATE
    )

    print(f"Database latest date: {latest_date.date()}")
    print(
        f"Requested analysis window: "
        f"{start_date} to {end_date}"
    )
    print(f"Tickers to analyze: {len(tickers)}")
    print(f"Report directory: {output_directory}")

    successful_reports, report_failures = (
        generate_reports(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            output_directory=output_directory,
        )
    )

    summary, collection_failures = (
        collect_portfolio_summary(
            successful_reports
        )
    )

    # Include both execution and result-collection failures.
    all_failures = {
        **report_failures,
        **collection_failures,
    }

    failed_file = (
        output_directory
        / "failed_stocks.txt"
    )

    failed_file.write_text(
        "\n".join(
            f"{ticker}: {message}"
            for ticker, message
            in all_failures.items()
        ),
        encoding="utf-8",
    )

    if collection_failures:
        print(
            "Result collection failed for: "
            + ", ".join(
                sorted(collection_failures)
            )
        )

    if summary.empty:
        print(
            "No portfolio summary was created "
            "because no report results were available."
        )
        return

    print(
        f"Collected results from "
        f"{len(summary)} reports."
    )

    print_sharpe_rankings(summary)

    save_portfolio_outputs(
        summary,
        output_directory,
    )

    print("Multi-stock analysis complete.")

if __name__ == "__main__":
    main()
