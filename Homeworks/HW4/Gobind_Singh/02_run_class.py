"""Mechanics check: trade the class extract with the saved Ridge scores.

The class database ends in 2015, and this model was fit on that same period,
so the result is not the homework test. It only checks that monthly top-10
orders, 90-day exits, and daily NAV run on real prices.
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

from backtest import run_backtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "Lectures" / "modeling"))
panel = pd.read_parquet(HERE / "artifacts" / "dev_panel_class.parquet")
saved = joblib.load(HERE / "artifacts" / "ridge_90d_class.joblib")

eligible = panel.loc[panel.eligible].copy()
eligible["score"] = saved["pipeline"].predict(eligible[saved["features"]])
prices = panel.pivot(index="date", columns="symbol", values="adj_close").sort_index()
result = run_backtest(prices, eligible[["date", "symbol", "score"]], "2014-01-02", "2015-12-31")

history = result["history"]
print(history.tail(3).to_string(index=False))
print(f"fills={len(result['fills'])} open_lots={len(result['open_lots'])} final_cash={result['cash']:.2f}")
history.to_csv(HERE / "artifacts" / "class_rehearsal_history.csv", index=False)
