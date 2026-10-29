"""Generate a risk report for every ticker with papermill, then build a portfolio summary.

Usage (from the repository root or this folder):
    python Homeworks/HW2/Antonio_Rivera/03_run_all_stocks.py

Steps:
1. Read all tickers from `ohlc_hw_cleaned` (created by 01_data_cleaning.ipynb).
2. Pick one common window for every ticker: the six complete calendar months before the
   month that contains the latest cleaned observation (anchored to the data, not the clock).
3. Execute 02_stock_report_template.ipynb once per ticker into reports/{run date}/.
4. Collect the scrapbook outputs of the reports that succeeded in THIS run, rank them by
   Sharpe ratio, plot risk vs return, and save portfolio_summary.csv.
"""

import logging
import sqlite3
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # write image files; no window needed
import matplotlib.pyplot as plt
import pandas as pd
import papermill as pm
import scrapbook as sb
from matplotlib.ticker import PercentFormatter
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "data" / "data.db"
TEMPLATE = HERE / "02_stock_report_template.ipynb"
REPORTS_ROOT = HERE / "reports"
SUMMARY_COLS = ["ticker", "avg_daily_return", "avg_volatility", "max_drawdown",
                "sharpe_ratio", "total_return", "report_path", "first_obs", "last_obs", "n_days"]

log = logging.getLogger("run_all_stocks")


def load_tickers(db_path: Path) -> list[str]:
    """Return the sorted list of tickers in ohlc_hw_cleaned."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker").fetchall()
    return [r[0] for r in rows]


def analysis_window(db_path: Path, months: int = 6) -> tuple[str, str, str]:
    """Return (start_date, end_date, latest_obs) for the last `months` complete months.

    The window ends on the last day of the month BEFORE the month containing the latest
    cleaned observation, even if that observation is itself a month-end: a last trading
    date alone does not prove the month's delivery is complete.
    Example: latest 2024-12-06 -> 2024-06-01 to 2024-11-30.
    """
    with sqlite3.connect(db_path) as conn:
        latest = conn.execute("SELECT MAX(ts) FROM ohlc_hw_cleaned").fetchone()[0]
    if latest is None:
        raise ValueError("ohlc_hw_cleaned is empty")
    latest_month = pd.Timestamp(latest).to_period("M")
    start = (latest_month - months).start_time.date()
    end = (latest_month - 1).end_time.date()
    return start.isoformat(), end.isoformat(), latest


def run_report(ticker: str, start_date: str, end_date: str, out_dir: Path) -> Path:
    """Execute the template for one ticker and return the output notebook path.

    Raises whatever papermill raises (e.g. PapermillExecutionError when a cell fails).
    """
    out_path = out_dir / f"{ticker}_report.ipynb"
    pm.execute_notebook(
        input_path=str(TEMPLATE),
        output_path=str(out_path),
        parameters={"ticker": ticker, "start_date": start_date, "end_date": end_date,
                    "db_path": str(DB_PATH.relative_to(HERE))},   # relative to cwd below
        cwd=str(HERE),
        progress_bar=False,
    )
    return out_path


def collect_summary(report_paths: dict[str, Path]) -> pd.DataFrame:
    """Read the scrapbook outputs of the given (successful) reports into one DataFrame.

    Only the reports passed in are read, never every notebook in the folder, so failed or
    stale reports from earlier runs cannot leak into the summary.
    """
    rows = []
    for ticker, path in report_paths.items():
        scraps = sb.read_notebook(str(path)).scraps
        row = {name: scraps[name].data for name in SUMMARY_COLS
               if name not in ("ticker", "report_path")}
        rows.append({"ticker": ticker, "report_path": str(path.relative_to(REPORTS_ROOT.parent)), **row})
    df = pd.DataFrame(rows, columns=SUMMARY_COLS)
    # Scrapbook stores a NaN (e.g. undefined Sharpe) as the string 'nan'; turn the metric
    # columns back into numbers so NaN is a real NaN again and is excluded from rankings.
    numeric = ["avg_daily_return", "avg_volatility", "max_drawdown", "sharpe_ratio", "total_return", "n_days"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df


def plot_risk_return(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Scatter of mean rolling volatility vs total return, colored by Sharpe ratio.

    Tickers with an undefined Sharpe ratio are drawn in gray. The top 5 by total return
    are labeled.
    """
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=(12, 8))
        ok = summary_df["sharpe_ratio"].notna()
        pts = ax.scatter(summary_df.loc[ok, "avg_volatility"], summary_df.loc[ok, "total_return"],
                         c=summary_df.loc[ok, "sharpe_ratio"], cmap="viridis", s=90,
                         edgecolor="white", linewidth=1.5, zorder=3)
        if (~ok).any():
            ax.scatter(summary_df.loc[~ok, "avg_volatility"], summary_df.loc[~ok, "total_return"],
                       color="#8a8a8a", s=90, edgecolor="white", linewidth=1.5, zorder=3,
                       label="Sharpe undefined")
            ax.legend(loc="lower right", frameon=True)
        if ok.any():
            fig.colorbar(pts, ax=ax, label="Sharpe ratio")
        for _, row in summary_df.nlargest(5, "total_return").iterrows():
            ax.annotate(row["ticker"], (row["avg_volatility"], row["total_return"]),
                        xytext=(6, 6), textcoords="offset points", fontsize=10, fontweight="bold")
        ax.axhline(0, color="#333333", lw=1)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set(xlabel="Average 20-day rolling volatility (annualized)",
               ylabel="Total return over the window")
        ax.set_title("Risk-Return Profile of Portfolio", fontsize=16, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def main() -> int:
    """Run every report, then summarize. Returns the process exit code."""
    run_date = date.today().isoformat()            # folder name = run date, not data window
    out_dir = REPORTS_ROOT / run_date
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("portfolio_summary.csv", "risk_return.png"):   # never show an old summary
        (out_dir / stale).unlink(missing_ok=True)
    # Log this script's messages to run.log. Papermill's own INFO chatter is not shown, and
    # per-ticker errors are echoed with tqdm.write so they don't break the progress bar.
    log.setLevel(logging.INFO)
    file_handler = logging.FileHandler(out_dir / "run.log", mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.handlers = [file_handler]

    tickers = load_tickers(DB_PATH)
    start_date, end_date, latest = analysis_window(DB_PATH)
    log.info("Latest cleaned observation %s -> window %s to %s for %d tickers",
             latest, start_date, end_date, len(tickers))

    succeeded: dict[str, Path] = {}
    failed: dict[str, str] = {}
    for ticker in tqdm(tickers, desc="Generating reports"):
        try:
            succeeded[ticker] = run_report(ticker, start_date, end_date, out_dir)
        except Exception as exc:                    # log and keep going with the next ticker
            reason = f"{type(exc).__name__}: {str(exc).strip().splitlines()[-1] if str(exc).strip() else ''}"
            failed[ticker] = reason
            log.error("%s failed: %s", ticker, reason)
            tqdm.write(f"{ticker} failed: {reason}")

    print(f"Successfully generated {len(succeeded)}/{len(tickers)} reports")
    (out_dir / "failed_stocks.txt").write_text(
        "".join(f"{t}\t{r}\n" for t, r in failed.items()))  # empty file = no failures
    if not succeeded:
        print("No reports succeeded, so there is no portfolio summary. See run.log for errors.")
        return 1

    summary_df = (collect_summary(succeeded)
                  .sort_values("sharpe_ratio", ascending=False, na_position="last")
                  .reset_index(drop=True))
    summary_df.to_csv(out_dir / "portfolio_summary.csv", index=False)   # fractions, not %

    # Coverage disclosure: tickers whose data ends before the requested window does
    # (compared with the fullest coverage, since the window may end on a non-trading day)
    full_end = summary_df["last_obs"].max()
    short = summary_df[summary_df["last_obs"] < full_end]
    print(f"\nRequested window {start_date} to {end_date}; fullest coverage ends {full_end}.")
    if len(short):
        print(f"{len(short)} ticker(s) have data ending earlier (no prices are extrapolated). Their "
              "metrics cover a shorter period, so compare them with the others with care:")
        print(short[["ticker", "first_obs", "last_obs", "n_days"]].to_string(index=False))

    ranked = summary_df[summary_df["sharpe_ratio"].notna()]   # undefined Sharpe is not ranked
    show = ["ticker", "sharpe_ratio", "total_return", "avg_volatility", "max_drawdown", "last_obs"]
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(f"\nTop 10 by Sharpe ratio ({len(ranked)} ranked):")
        print(ranked.head(10)[show].to_string(index=False))
        print("\nBottom 10 by Sharpe ratio:")
        print(ranked.tail(10)[show].to_string(index=False))
    if len(ranked) < 20:
        print("(Fewer than 20 ranked tickers, so the top and bottom lists overlap.)")

    plot_risk_return(summary_df, out_dir / "risk_return.png")
    print(f"\nSaved {out_dir / 'portfolio_summary.csv'} and {out_dir / 'risk_return.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
