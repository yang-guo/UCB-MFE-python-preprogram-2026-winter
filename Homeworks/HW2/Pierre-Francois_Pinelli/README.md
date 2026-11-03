HW2: Data cleaning and stock reports

Pierre-Francois Pinelli


How to run (from this folder)

Install the packages (Python 3.11 or later): python3 -m venv .venv, then source .venv/bin/activate, then pip install -r requirements.txt. The course environment (uv sync at the root of the repo) also works.

Create the folder data and put the course database in data/data.db (from bCourses, never committed).

Run 01_data_cleaning.ipynb (Run All in VS Code, or papermill 01_data_cleaning.ipynb 01_data_cleaning.ipynb, about 10 seconds), then python 03_run_all_stocks.py (about 1 minute). The results go in reports/{date of the run}/.


Design decisions

I use the reference table ohlc (same stocks and dates) to check ohlc_hw, as the assignment allows.

Adjusted close: 18,374 are missing and 10,417 are 4 to 24 times too high on single days. I rebuild them as close * factor, with the factor of ohlc on the same day. I never forward fill the factor because it changes at each split and dividend. This is the only choice that changes the 2024 reports.

Sunday rows (384, in 1974 and 1975): excluded, the NYSE is closed on weekends. The other days of these weeks are probably one day early too, but I cannot prove it, so I keep them.

open = 0 (5 rows, each the first day of a stock): excluded, there is no earlier price to repair from and I prefer not to invent one.

Impossible prices: Option 1 (missing, then forward fill in the stock), then I check the range of the day again. If high < low I only replace high and low, so a correct close is not lost. These cases are not in the data, so they are tested on a small fake table in 01.

Count: 397,711 raw rows = 29,460 duplicates + 389 excluded + 367,862 in ohlc_hw_cleaned.


Outliers

A large return is an error only with a proof: the close is more than 1% away from ohlc, or the volume is 0 (no trade, so the price cannot move). The check with ohlc finds nothing because the prices are the same as ohlc. The volume check finds 4 errors (TRV in 1976, 1978 and 1981, AXP in 1972), which I interpolate. I keep the other 5,182 flags, because deleting real crashes would hide risk.


Assumptions

ohlc is correct when the two tables disagree.

Window: the 6 complete months before the last date of the data (2024-12-06), so 2024-06-01 to 2024-11-30.

22 stocks have data only until 2024-10-18 (97 days instead of 126). Their metrics use only these dates, each report shows its real dates and the script prints a note. Their total return and maximum drawdown cover a shorter period, so they are not directly comparable with the 8 other stocks: 4 of the 5 best total returns on the chart (MMM, SHW, TRV, HD) are short histories. The average daily return, the volatility and the Sharpe ratio are per day or per year, so they stay comparable.


How to read the reports

The price is red when the 20-day volatility is above its mean, green below. A Sharpe ratio that cannot be computed is shown as N/A and not ranked. portfolio_summary.csv is sorted by Sharpe.


Sample reports

KO: the calm stock (lowest volatility, 13%). AMGN: a short history (97 days) with 13 repaired adjusted closes in the window. NKE: a losing stock (-16%, drawdown -27%) with 3 large returns kept as flags (they match ohlc, with high volume).
