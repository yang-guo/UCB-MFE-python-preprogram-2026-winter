"""Worked example: a function becomes a tool after we define its contract.

Run this script before opening the harness. All numbers here are synthetic.
The first calculation is small enough to check without trusting a model.
"""

import numpy as np
import pandas as pd

from research_harness.data import portfolio_exposure, risk_metrics
from research_harness.worker import ROOT


def main():
    # 100 -> 110 -> 99: +10%, -10%, and a -1% whole-period return.
    prices = pd.Series([100.0, 110.0, 99.0])
    print("Simple returns:", prices.pct_change(fill_method=None).tolist())
    print("Log returns:", np.log(prices / prices.shift(1)).tolist())
    print("Drawdowns:", (prices / prices.cummax() - 1).tolist())

    # A tool receives validated inputs and returns named, serializable results.
    # No model is needed to calculate a return or choose its denominator.
    fixture = pd.read_csv(ROOT / "data/prices.csv", parse_dates=["date"])
    metrics = {t: risk_metrics(g.sort_values("date")) for t, g in fixture.groupby("ticker")}
    print("\nAAPL risk metrics:", metrics["AAPL"])
    exposure = portfolio_exposure(ROOT / "data/holdings.csv", metrics, cash=5000)
    print("\nPortfolio exposure:")
    print(pd.DataFrame(exposure["positions"]).to_string(index=False))
    print("Cash weight:", exposure["cash_weight"])
    print("Weights including cash:", sum(p["weight"] for p in exposure["positions"]) + exposure["cash_weight"])


if __name__ == "__main__":
    main()
