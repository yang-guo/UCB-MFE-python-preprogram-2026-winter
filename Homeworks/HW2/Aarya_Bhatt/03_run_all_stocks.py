import sqlite3
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import papermill as pm
import scrapbook as sb
from tqdm import tqdm

DB_PATH = Path("data/data.db")
TEMPLATE_NB = Path("02_stock_report_template.ipynb")

def get_tickers_and_window(db_path: Path) -> tuple[list[str], str, str]:
    """Load unique tickers and compute the last 6 complete calendar months."""
    with sqlite3.connect(db_path) as conn:
        tickers = pd.read_sql_query("SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker", conn)["ticker"].tolist()
        max_ts_str = pd.read_sql_query("SELECT MAX(ts) AS max_ts FROM ohlc_hw_cleaned", conn)["max_ts"].iloc[0]
        
    latest_dt = pd.Timestamp(max_ts_str)
    first_of_latest_month = latest_dt.replace(day=1)
    end_dt = first_of_latest_month - pd.Timedelta(days=1)
    start_dt = (first_of_latest_month - pd.DateOffset(months=6)).replace(day=1)
    
    return tickers, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")

def main():
    tickers, start_date, end_date = get_tickers_and_window(DB_PATH)
    run_date_str = date.today().strftime("%Y-%m-%d")
    out_dir = Path("reports") / run_date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analysis Window: {start_date} to {end_date} | Output Dir: {out_dir}")
    
    records = []
    failed_tickers = []
    
    for tkr in tqdm(tickers, desc="Generating reports"):
        out_nb = out_dir / f"{tkr}_report.ipynb"
        try:
            pm.execute_notebook(
                input_path=str(TEMPLATE_NB),
                output_path=str(out_nb),
                parameters={
                    "ticker": tkr,
                    "start_date": start_date,
                    "end_date": end_date,
                    "db_path": str(DB_PATH.resolve())
                },
                progress_bar=False,
                kernel_name="python3"
            )
            # Read glued metrics from the executed notebook using scrapbook
            nb_glue = sb.read_notebook(str(out_nb))
            m = nb_glue.scraps["portfolio_metrics"].data
            records.append({
                "ticker": m["ticker"],
                "avg_daily_return": m["avg_daily_return"],
                "avg_volatility": m["avg_volatility"],
                "max_drawdown": m["max_drawdown"],
                "sharpe_ratio": np.nan if m["sharpe_ratio"] is None else float(m["sharpe_ratio"]),
                "total_return": m["total_return"],
                "report_path": str(out_nb)
            })
        except Exception as exc:
            failed_tickers.append(f"{tkr}: {exc}")
            print(f"\nError on {tkr}: {exc}")

    print(f"\nSuccessfully generated {len(records)}/{len(tickers)} reports")
    
    # Save failed tickers log if any
    failed_file = Path("failed_stocks.txt")
    failed_file.write_text("\n".join(failed_tickers) + ("\n" if failed_tickers else ""))
    
    if not records:
        print("All stock reports failed. Skipping summary dashboard.")
        return

    summary_df = pd.DataFrame(records)
    summary_df = summary_df.sort_values(by="sharpe_ratio", ascending=False, na_position="last").reset_index(drop=True)
    
    # Print Top 10 and Bottom 10
    valid_sharpe_df = summary_df.dropna(subset=["sharpe_ratio"])
    print("\n--- Top 10 Stocks by Sharpe Ratio ---")
    print(valid_sharpe_df.head(10).to_string(index=False))
    
    print("\n--- Bottom 10 Stocks by Sharpe Ratio ---")
    print(valid_sharpe_df.tail(10).to_string(index=False))
    
    # Save CSV summary
    csv_out = out_dir / "portfolio_summary.csv"
    summary_df.to_csv(csv_out, index=False)
    
    # Risk-Return Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        summary_df["avg_volatility"],
        summary_df["total_return"],
        c=summary_df["sharpe_ratio"],
        cmap="viridis",
        s=80,
        edgecolors="black",
        alpha=0.85
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Annualized Sharpe Ratio")
    
    top5_ret = summary_df.nlargest(5, "total_return")
    for _, row in top5_ret.iterrows():
        ax.annotate(
            row["ticker"],
            xy=(row["avg_volatility"], row["total_return"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontweight="bold",
            fontsize=9
        )
        
    ax.set_title("Risk-Return Profile of Portfolio", fontsize=14, fontweight="bold")
    ax.set_xlabel("Average Rolling Volatility (Fraction)")
    ax.set_ylabel("Total Return (Fraction)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "portfolio_risk_return.png", dpi=150)
    plt.close()
    print(f"\nSummary CSV and risk-return scatter plot successfully saved to {out_dir}/")

if __name__ == "__main__":
    main()