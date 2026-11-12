"""Performance metrics shared by the strategy and the benchmark."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def max_drawdown(values):
    """Largest peak-to-trough loss of a value series (a negative number)."""
    return float((values / values.cummax() - 1).min())


def performance_metrics(values, risk_free=0.02):
    """Final value, CAGR, max drawdown and Sharpe from a daily value series.

    * CAGR uses actual elapsed calendar days / 365.25.
    * Sharpe uses daily returns in excess of the daily equivalent of a 2%
      annual rate, annualised with sqrt(252).
    """
    values = values.dropna()
    returns = values.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    daily_rf = (1 + risk_free) ** (1 / TRADING_DAYS) - 1
    excess = returns - daily_rf
    return pd.Series({
        "final_value": values.iloc[-1],
        "total_return": values.iloc[-1] / values.iloc[0] - 1,
        "cagr": (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1,
        "max_drawdown": max_drawdown(values),
        "sharpe": excess.mean() / excess.std() * np.sqrt(TRADING_DAYS),
        "annual_volatility": returns.std() * np.sqrt(TRADING_DAYS),
    })


def buy_and_hold(benchmark, first_close, end, initial_cash=1_000_000.0):
    """Value of ``initial_cash`` invested in the benchmark at ``first_close``.

    The series starts one session earlier at ``initial_cash`` (still in cash),
    exactly like the strategy history, so both share the same dates.
    """
    position = benchmark.index.get_loc(pd.Timestamp(first_close))
    series = benchmark.iloc[position - 1:].loc[:pd.Timestamp(end)]
    values = initial_cash * series / benchmark.iloc[position]
    values.iloc[0] = initial_cash
    return values.rename("benchmark")


def rank_ic(frame, prediction="prediction", target="target"):
    """Spearman correlation between prediction and outcome, date by date."""
    def one_date(group):
        if len(group) < 20:
            return np.nan
        return group[prediction].corr(group[target], method="spearman")

    usable = frame.dropna(subset=[prediction, target])
    return usable.groupby("date")[[prediction, target]].apply(one_date)
