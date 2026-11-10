# HW4 – Mila Marsot

## The idea

Every month, I buy a group of stocks and I keep each one for 90 days. I make four choices on 2012–2015 only: the score (3-month momentum, 12-month momentum or the course Ridge model), the number of stocks, the share of very volatile stocks I remove, and how I split the money. Each time, I keep the option with the best Sharpe ratio. Then I test the final strategy once, from April 2016 to September 2026, with my own backtester.

## How to run

As in Lecture 8: run `uv sync --locked`, put `equities_top2000.duckdb` in `Lectures/modeling/data/`, then open `hw4.ipynb` and click *Run All* with the `.venv` kernel.

## Results

My final strategy: 12-month momentum, 50 stocks per month, without the 10% most volatile stocks, the same amount for each stock.

| April 2016 – September 2026 | My strategy | S&P 500 |
|---|---:|---:|
| Final value | $2,634,120 | $4,402,747 |
| CAGR | 9.7% | 15.2% |
| Max drawdown | −38.8% | −33.8% |
| Sharpe ratio | 0.45 | 0.77 |

My strategy did not beat the S&P 500: it ends $1.77 million below. The *Findings* at the end of the notebook explain why, and show my checks, the data problems, and what I tried first.

## Sources and AI use

The data preparation and the Ridge model come from the course code (`Lectures/modeling/course_data.py`). The monthly rule (a third of the portfolio each month) comes from Lecture 8. The backtester is my own code; I only use the Lecture 8 backtester to check my results.

I used AI to help me write the notebook, understand the functions of the lectures and understand how to use them in my strategy. It also found a bug in how my backtester used cash (it used the money of a sale to buy on the same day), and it wrote the fix, the comparison cell, and the cash check. 