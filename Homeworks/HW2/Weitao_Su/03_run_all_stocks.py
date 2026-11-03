"""Run the stock report template for every cleaned ticker."""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import papermill as pm
import scrapbook as sb
from tqdm import tqdm


base_dir = Path(__file__).resolve().parent


def load_stocks(db_path):
    """Return all tickers and the six calendar months before the latest month."""
    if not db_path.is_file():
        raise FileNotFoundError("Run 01_data_cleaning.ipynb first.")
    conn = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
    try:
        tickers = pd.read_sql_query(
            "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn
        )["ticker"].tolist()
        latest = pd.read_sql_query(
            "SELECT MAX(ts) AS latest FROM ohlc_hw_cleaned", conn
        )["latest"].iloc[0]
    finally:
        conn.close()
    if not tickers or pd.isna(latest):
        raise ValueError("ohlc_hw_cleaned is empty.")
    month_start = pd.Timestamp(latest).replace(day=1).normalize()
    start_date = (month_start - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    end_date = (month_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return tickers, start_date, end_date


def run_report(ticker, start_date, end_date, folder, db_path):
    """Execute one report and collect its named Scrapbook results."""
    output = folder / f"{ticker}_report.ipynb"
    pm.execute_notebook(
        str(base_dir / "02_stock_report_template.ipynb"), str(output),
        parameters={
            "ticker": ticker, "start_date": start_date, "end_date": end_date,
            "db_path": str(db_path),
        },
        kernel_name="python3", cwd=str(base_dir), progress_bar=False,
        log_output=False,
    )
    out = dict(sb.read_notebook(str(output)).scraps["stats"].data)
    if (
        out["ticker"] != ticker or out["start_date"] != start_date
        or out["end_date"] != end_date
    ):
        raise ValueError("Report parameters do not match its results.")
    out["report_path"] = output.relative_to(base_dir).as_posix()
    return out


def plot_summary(df, folder):
    """Save a portfolio scatter plot using numeric fractions."""
    out = df.loc[
        np.isfinite(df["avg_volatility"]) & np.isfinite(df["total_return"])
    ].copy()
    defined = np.isfinite(out["sharpe_ratio"])
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    if defined.any():
        points = ax.scatter(
            out.loc[defined, "avg_volatility"], out.loc[defined, "total_return"],
            c=out.loc[defined, "sharpe_ratio"], cmap="viridis", s=70,
        )
        fig.colorbar(points, ax=ax, label="Sharpe ratio")
    if (~defined).any():
        ax.scatter(
            out.loc[~defined, "avg_volatility"], out.loc[~defined, "total_return"],
            color="gray", marker="x", label="Undefined Sharpe",
        )
        ax.legend()
    if out.empty:
        ax.text(0.5, 0.5, "Insufficient history for this chart", ha="center", transform=ax.transAxes)
    for _, row in out.nlargest(5, "total_return").iterrows():
        ax.annotate(
            row["ticker"], (row["avg_volatility"], row["total_return"]),
            xytext=(5, 5), textcoords="offset points",
        )
    ax.set(
        title="Risk-Return Profile of Portfolio",
        xlabel="Average rolling volatility (annualized)", ylabel="Total return",
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(True, alpha=0.3)
    fig.savefig(folder / "portfolio_summary.png", dpi=150)
    plt.close(fig)


def main():
    """Generate reports, log failures, and summarize only this run's successes."""
    db_path = base_dir / "data/data.db"
    tickers, start_date, end_date = load_stocks(db_path)
    run_date = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    folder = base_dir / "reports" / run_date
    folder.mkdir(parents=True, exist_ok=True)
    for name in ["portfolio_summary.csv", "portfolio_summary.png"]:
        (folder / name).unlink(missing_ok=True)
    logging.basicConfig(
        filename=folder / "errors.log", filemode="w", level=logging.ERROR,
        format="%(asctime)s %(message)s", force=True,
    )
    print(f"Requested window for all tickers: {start_date} to {end_date}")

    rows = []
    failures = []
    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            rows.append(run_report(ticker, start_date, end_date, folder, db_path))
        except Exception:
            failures.append(ticker)
            logging.exception("Failed ticker: %s", ticker)
            tqdm.write(f"Failed {ticker}; see errors.log")

    (folder / "failed_stocks.txt").write_text(
        "".join(f"{ticker}\n" for ticker in failures), encoding="utf-8"
    )
    print(f"Successfully generated {len(rows)}/{len(tickers)} reports")
    if not rows:
        print("No successful reports. No portfolio summary was created.")
        return 1

    summary_df = pd.DataFrame(rows)
    metric_cols = ["avg_daily_return", "avg_volatility", "max_drawdown",
                   "sharpe_ratio", "total_return"]
    summary_df[metric_cols] = summary_df[metric_cols].apply(pd.to_numeric, errors="coerce")
    summary_df = summary_df.sort_values("sharpe_ratio", ascending=False, na_position="last")
    summary_df.to_csv(folder / "portfolio_summary.csv", index=False)

    ranked = summary_df.loc[np.isfinite(summary_df["sharpe_ratio"])]
    rank_cols = ["ticker", "sharpe_ratio", "total_return", "avg_volatility"]
    print("\nTop 10")
    print(ranked.head(10)[rank_cols].to_string(index=False))
    print("\nBottom 10")
    print(ranked.tail(10)[rank_cols].to_string(index=False))
    print("\nActual coverage")
    print(summary_df[["ticker", "actual_start_date", "actual_end_date", "trading_days"]].to_string(index=False))
    print("Histories may be shorter than the requested window; comparisons use different actual coverage.")
    print("Prices are not extrapolated beyond supplied observations.")
    plot_summary(summary_df, folder)
    print(f"Saved results to {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
