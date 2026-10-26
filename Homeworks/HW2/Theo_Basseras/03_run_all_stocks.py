"""Run the single-stock report for every ticker with papermill, then build the portfolio summary.

Usage (from Homeworks/HW2/Theo_Basseras, after 01_data_cleaning.ipynb has created ohlc_hw_cleaned):
    python 03_run_all_stocks.py [--db data/data.db] [--template 02_stock_report_template.ipynb] [--limit N]

Outputs, in reports/<run date YYYY-MM-DD>/:
    <TICKER>_report.ipynb      one executed report per ticker that succeeded
    failed_stocks.txt          tickers whose report failed (empty file when none failed)
    run.log                    one line per ticker: success, or the error message
    portfolio_summary.csv      metrics of the successful reports, sorted by Sharpe ratio
    risk_return.png            avg volatility vs total return, colored by Sharpe ratio
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no window is opened

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import papermill as pm  # noqa: E402
import scrapbook as sb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from tqdm import tqdm  # noqa: E402

HERE = Path(__file__).resolve().parent
METRICS = ["avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return"]


def load_tickers(db_path: Path) -> list[str]:
    """All tickers present in the cleaned table, sorted."""
    with sqlite3.connect(db_path) as con:
        rows = con.execute("SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker").fetchall()
    return [r[0] for r in rows]


def analysis_window(db_path: Path) -> tuple[str, str, str]:
    """The six complete calendar months before the month of the latest cleaned observation.

    The window is anchored to the data, not to the computer clock: if the latest observation is
    2024-12-06, the window is 2024-06-01 to 2024-11-30. The month containing the latest observation is
    always excluded, even when that observation falls on its last day (a final trading date does not
    prove the month's delivery is complete). Returns (start, end, latest observation) as ISO strings.
    """
    with sqlite3.connect(db_path) as con:
        latest = con.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()[0]
    first_of_latest_month = pd.Timestamp(latest).to_period("M").to_timestamp()
    end = first_of_latest_month - pd.Timedelta(days=1)
    start = (first_of_latest_month - pd.DateOffset(months=6)).normalize()
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), latest


def run_report(template: Path, output: Path, ticker: str, start: str, end: str, db_path: Path) -> dict:
    """Execute the template for one ticker and return its recorded metrics (scrapbook values).

    The paths are relative to this folder (main() runs from it), so the executed report stores no
    local absolute path, neither in its parameters nor in its papermill metadata.
    """
    pm.execute_notebook(
        template.as_posix(), output.as_posix(),
        parameters={"ticker": ticker, "start_date": start, "end_date": end, "db_path": db_path.as_posix()},
        cwd=".", progress_bar=False, log_output=False,
    )
    scraps = sb.read_notebook(output.as_posix()).scraps
    missing = [name for name in METRICS if name not in scraps]
    if missing:
        raise RuntimeError(f"report finished without recording {missing}")
    record = {name: scraps[name].data for name in ["ticker", "first_date", "last_date", "n_days"] + METRICS}
    record["report_path"] = output.as_posix()
    return record


def plot_risk_return(summary: pd.DataFrame, path: Path, window: str, window_last: str) -> None:
    """Scatter of average rolling volatility vs total return, colored by Sharpe, top 5 returns annotated.

    Circles are tickers with data over the whole window; squares are tickers whose data stops earlier,
    so their numbers cover a shorter period.
    """
    fig, ax = plt.subplots(figsize=(11, 7.5))
    norm = plt.Normalize(summary["sharpe_ratio"].min(), summary["sharpe_ratio"].max())
    full = summary["last_date"] == window_last
    for part, marker in [(summary[full], "o"), (summary[~full], "s")]:
        if len(part):
            ax.scatter(part["avg_volatility"], part["total_return"], c=part["sharpe_ratio"], cmap="RdYlGn",
                       norm=norm, marker=marker, s=80, edgecolor="black", linewidth=0.5)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdYlGn"), ax=ax, label="Sharpe ratio")
    for _, row in summary.nlargest(5, "total_return").iterrows():
        ax.annotate(row["ticker"], (row["avg_volatility"], row["total_return"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Average 20-day rolling volatility (annualized)")
    ax.set_ylabel("Total return over the window")
    short_ends = sorted(summary.loc[~full, "last_date"].unique())
    subtitle = f"Report window {window}"
    if short_ends:
        subtitle += f"; {int((~full).sum())} of {len(summary)} tickers have data only to {', '.join(short_ends)}"
    ax.set_title(subtitle, fontsize=10)
    marker_key = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="lightgray", markeredgecolor="black",
               markersize=9, label=f"data to {window_last} (whole window)"),
        Line2D([0], [0], marker="s", linestyle="", markerfacecolor="lightgray", markeredgecolor="black",
               markersize=9, label="data stops earlier (shorter period)"),
    ]
    ax.legend(handles=marker_key, loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.suptitle("Risk-Return Profile of Portfolio", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    """Generate one report per ticker, then the portfolio summary. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default="data/data.db", help="database with ohlc_hw_cleaned (relative to this folder)")
    parser.add_argument("--template", default="02_stock_report_template.ipynb", help="relative to this folder")
    parser.add_argument("--limit", type=int, default=None, help="only the first N tickers (for a quick test)")
    args = parser.parse_args()

    os.chdir(HERE)   # paths below are relative to this folder, whatever the caller's working directory
    db_path, template = Path(args.db), Path(args.template)
    out_dir = Path("reports") / date.today().isoformat()   # run date, independent of the data window
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=out_dir / "run.log", filemode="w", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("papermill").setLevel(logging.WARNING)   # keep run.log to one line per ticker
    tickers = load_tickers(db_path)[: args.limit]
    start, end, latest = analysis_window(db_path)
    print(f"Latest cleaned observation {latest}: window {start} to {end} for {len(tickers)} tickers")
    logging.info("window %s to %s (latest observation %s), %d tickers", start, end, latest, len(tickers))

    records, failed = [], []
    for ticker in tqdm(tickers, desc="Generating reports"):
        output = out_dir / f"{ticker}_report.ipynb"
        try:
            records.append(run_report(template, output, ticker, start, end, db_path))
            logging.info("%s: success", ticker)
        except Exception as exc:  # one bad ticker must not stop the batch
            failed.append(ticker)
            message = str(exc).strip().splitlines()[-1] if str(exc).strip() else type(exc).__name__
            logging.error("%s: failed: %s", ticker, message)
            output.unlink(missing_ok=True)   # never leave a half-run report that could be mistaken for a result

    (out_dir / "failed_stocks.txt").write_text("".join(f"{t}\n" for t in failed), encoding="utf-8")
    print(f"Successfully generated {len(records)}/{len(tickers)} reports")
    if not records:
        print("No report succeeded: no portfolio summary is produced. See run.log.")
        return 1

    summary = pd.DataFrame(records)
    ranked = summary[summary["sharpe_ratio"].notna()].sort_values("sharpe_ratio", ascending=False)
    unranked = summary[summary["sharpe_ratio"].isna()]
    summary = pd.concat([ranked, unranked], ignore_index=True)
    columns = ["ticker", *METRICS, "first_date", "last_date", "n_days", "report_path"]
    summary[columns].to_csv(out_dir / "portfolio_summary.csv", index=False)

    shown = ["ticker", "sharpe_ratio", "total_return", "avg_volatility", "max_drawdown", "last_date"]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print("\nTop 10 by Sharpe ratio:\n", ranked.head(10)[shown].to_string(index=False))
        print("\nBottom 10 by Sharpe ratio:\n", ranked.tail(10)[shown].to_string(index=False))
    if len(unranked):
        print(f"\nNot ranked (undefined Sharpe): {', '.join(unranked['ticker'])}")
    window_last = summary["last_date"].max()   # last trading day with data inside the window
    short = summary[summary["last_date"] < window_last]
    if len(short):
        ends = ", ".join(sorted(short["last_date"].unique()))
        print(f"\nCoverage note: {len(short)} of {len(summary)} tickers have no data after {ends} "
              f"(the others run to {window_last}); their metrics cover a shorter period, "
              "so compare them with care.")

    plot_risk_return(ranked if len(ranked) else summary, out_dir / "risk_return.png", f"{start} to {end}",
                     window_last)
    print(f"\nOutputs in {out_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
