import sqlite3
import papermill as pm
from pathlib import Path
from datetime import date
import pandas as pd
from tqdm import tqdm
import scrapbook as sb
import matplotlib.pyplot as plt


# connect to the cleaned database
conn = sqlite3.connect("data/data.db")

# unique tickers
tickers = pd.read_sql_query(
    "SELECT DISTINCT ticker FROM ohlc_hw_cleaned ORDER BY ticker",
    conn)
tickers = tickers["ticker"].tolist()

# last observation date
latest_date = pd.read_sql_query(
    "SELECT MAX(ts) AS latest_date FROM ohlc_hw_cleaned",
    conn
)
latest_date = pd.to_datetime(latest_date.loc[0, "latest_date"])

conn.close()


# first day of the latest observation's month
current_month = latest_date.replace(day=1)

# six complete calendar months before that month
start_date = current_month - pd.DateOffset(months=6)
end_date = current_month - pd.Timedelta(days=1)

# convert dates to strings for Papermill
start_date = start_date.strftime("%Y-%m-%d")
end_date = end_date.strftime("%Y-%m-%d")



# create folder for reports
run_date = date.today().isoformat()

output_dir = Path("reports") / run_date
output_dir.mkdir(parents=True, exist_ok=True)



# tracking bar 
successful = 0
failed_stocks = []


results = []
for ticker in tqdm(tickers, desc="Generating reports"):

    output_path = output_dir / f"{ticker}_report.ipynb"

    try:
        pm.execute_notebook(
            "02_stock_report_template.ipynb",
            str(output_path),
            parameters={
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date})
        # read the completed notebook
        notebook = sb.read_notebook(str(output_path))

        # collect statistics for the portfolio summary
        results.append({
            "ticker": ticker,
            "avg_daily_return": notebook.scraps["avg_daily_return"].data,
            "avg_volatility": notebook.scraps["avg_volatility"].data,
            "max_drawdown": notebook.scraps["max_drawdown"].data,
            "sharpe_ratio": notebook.scraps["sharpe_ratio"].data,
            "total_return": notebook.scraps["total_return"].data,
            "report_path": str(output_path)})

        successful = successful+ 1

    except Exception as e:
        failed_stocks.append(ticker)
        print(f"Failed: {ticker} - {e}")


# how many stocks were processed successfully
print(f"Successfully generated {successful}/{len(tickers)} reports")


# save failed stock tickers
with open(output_dir / "failed_stocks.txt", "w") as file:
    for ticker in failed_stocks:
        file.write(ticker + "\n")

# Create the portfolio summary DataFrame
summary_df = pd.DataFrame(
    results,
    columns=[
        "ticker",
        "avg_daily_return",
        "avg_volatility",
        "max_drawdown",
        "sharpe_ratio",
        "total_return",
        "report_path" ])
summary_df.to_csv(output_dir / "portfolio_summary.csv", index=False)


# exclude undefined Sharpe ratios and sort from highest to lowest
ranked = summary_df.dropna(subset=["sharpe_ratio"])
ranked = ranked.sort_values("sharpe_ratio", ascending=False)

#  top 10 stocks
print("\nTop 10 Stocks by Sharpe Ratio")
print(ranked.head(10).to_string(index=False))

#  bottom 10 stocks
print("\nBottom 10 Stocks by Sharpe Ratio")
print(ranked.tail(10).sort_values("sharpe_ratio").to_string(index=False))



# plotting 
# Remove stocks with missing values
plot_df = summary_df.dropna(
    subset=["avg_volatility", "total_return", "sharpe_ratio"]
)

# Create scatter plot
plt.figure(figsize=(10, 6))

plt.scatter(
    plot_df["avg_volatility"],
    plot_df["total_return"],
    c=plot_df["sharpe_ratio"],
    cmap="RdYlGn"
)

plt.colorbar(label="Sharpe Ratio")

# Label the five highest-return stocks
for _, stock in plot_df.nlargest(5, "total_return").iterrows():
    plt.annotate(
        stock["ticker"],
        (stock["avg_volatility"], stock["total_return"])
    )

plt.title("Risk-Return Profile of Portfolio")
plt.xlabel("Average Rolling Volatility")
plt.ylabel("Total Return")
plt.grid(True)

plt.savefig(output_dir / "portfolio_risk_return.png")
plt.show()