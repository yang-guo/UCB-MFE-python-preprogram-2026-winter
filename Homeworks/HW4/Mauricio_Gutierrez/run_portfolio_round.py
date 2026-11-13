"""Development portfolio round: model D forecasts, four portfolio variants.

Compares 10 or 30 stocks per decision with the monthly schedule and the cash
schedule. The model is fixed: forecasts come from the suspect-price refit of
model D and are read from disk. Nothing after 2015-12-31 is read.

Run from the repository root:
    .venv/Scripts/python.exe Homeworks/HW4/Mauricio_Gutierrez/run_portfolio_round.py

Selection rule, fixed before the run: highest daily Sharpe; if another variant
is within 0.05 of it, choose the one with the smaller maximum drawdown.
"""
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HW_DIR))
from feature_engineering import load_development_inputs  # noqa: E402
from strategy_backtest import (  # noqa: E402
    BacktestConfig, Backtester, InverseVolatilityStrategy, detailed_performance,
    portfolio_chart_data, reconcile_account,
)

DB_PATH = HW_DIR / "outputs/data_cleaning/equities_top2000/equities_top2000_cleaned.duckdb"
FORECASTS = HW_DIR / "outputs/full_data_development/suspect_rule/forecasts.csv"
OUTPUT = HW_DIR / "outputs/full_data_development/portfolio_round"
START, END = "2013-01-02", "2015-12-31"
VOLATILITY_QUANTILE = 0.80      # Exclude the most volatile 20% of candidates at each signal.
SUSPECT_RATIO = (0.2, 5.0)      # One-day price ratio outside these bounds is treated as no quote.
MIN_CASH_WEIGHT = 0.05          # Cash schedule: decide only with at least 5% of NAV in cash.
MIN_SESSIONS_BETWEEN = 21       # Cash schedule: at least 21 sessions between decisions.
SHARPE_TIE = 0.05
BASE = BacktestConfig(initial_cash=1_000_000, top_n=10, budget_fraction=1 / 3, holding_days=90,
                      stock_limit=0.30, sector_limit=0.50, fee_rate=0.001, risk_free_rate=0.02)
# name: (stocks per decision, schedule, largest allowed universe_rank; 2000 keeps every member)
VARIANTS = {"D10-monthly": (10, "monthly", 2000), "D30-monthly": (30, "monthly", 2000),
            "D10-cash": (10, "cash", 2000), "D30-cash": (30, "cash", 2000),
            # Size round: same strategy, candidates limited to the largest companies.
            "D30-cash-top1000": (30, "cash", 1000), "D30-cash-top500": (30, "cash", 500),
            "D30-cash-top200": (30, "cash", 200)}
STAGE_1 = ["D10-monthly", "D30-monthly", "D10-cash", "D30-cash"]
STAGE_2 = ["D30-cash", "D30-cash-top1000", "D30-cash-top500", "D30-cash-top200"]


def choose(table):
    """Apply the fixed selection rule to the strategy rows of the metrics table."""
    best = table.daily_Sharpe.max()
    close = table.loc[table.daily_Sharpe >= best - SHARPE_TIE]
    return close.max_drawdown.idxmax()  # Drawdowns are negative: the largest is the smallest loss.


def main():
    started = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prices, calendar, membership, sectors, benchmark, _ = load_development_inputs(
        DB_PATH, end=END, lookback_start=START)
    forecasts = pd.read_csv(FORECASTS, usecols=["date", "symbol", "prediction", "volatility_63"],
                            parse_dates=["date"])
    assert forecasts.date.max() <= pd.Timestamp(END) and prices.date.max() <= pd.Timestamp(END)

    rows, curves_by_name, checks = [], {}, []
    for name, (top_n, schedule, max_rank) in VARIANTS.items():
        config = replace(BASE, top_n=top_n)
        # universe_rank is the quarterly size rank published with the membership list,
        # known at selection_date. Rank 1 is the largest capitalization proxy.
        members = membership.loc[membership.universe_rank.le(max_rank)]
        engine = Backtester(config, InverseVolatilityStrategy(config), allow_stale_quotes=True,
                            volatility_quantile=VOLATILITY_QUANTILE, schedule=schedule,
                            min_cash_weight=MIN_CASH_WEIGHT, min_sessions_between=MIN_SESSIONS_BETWEEN,
                            suspect_ratio=SUSPECT_RATIO)
        result = engine.run(prices, calendar, forecasts, sectors, members, START, END)
        checks.append({"variant": name, **reconcile_account(result, calendar, config)})
        curves, metrics = detailed_performance(result, benchmark, config)
        curves_by_name[name] = curves.Strategy
        buys = result.trades.side.eq("BUY")
        rows.append({"portfolio": name, **metrics.loc["Strategy"].to_dict(),
                     "max_universe_rank": max_rank,
                     "median_candidates": float(result.decisions.candidate_count.median()),
                     "decisions": len(result.decisions),
                     "decisions_with_purchases": int(result.decisions.reason.eq("planned").sum()),
                     "purchases": int(buys.sum()),
                     "no_quote_exits": int(result.trades.no_quote_exit.fillna(False).astype(bool).sum()),
                     "flagged_valuations": len(result.valuation_flags),
                     "suspect_quotes_removed": len(result.suspect_quotes),
                     "exposure_breach_sessions": int(result.daily.exposure_breach.sum())})
        run_dir = OUTPUT / name
        run_dir.mkdir(exist_ok=True)
        for field in ["daily", "trades", "orders", "decisions", "rejections", "open_lots", "valuation_flags"]:
            getattr(result, field).to_csv(run_dir / f"{field}.csv", index=False)
        (run_dir / "experiment.json").write_text(json.dumps({
            "variant": name, "start": START, "end": END, "forecasts": str(FORECASTS.relative_to(HW_DIR)),
            "schedule": schedule, "max_universe_rank": max_rank, "min_cash_weight": MIN_CASH_WEIGHT,
            "min_sessions_between": MIN_SESSIONS_BETWEEN, "suspect_ratio": SUSPECT_RATIO,
            "volatility_quantile": VOLATILITY_QUANTILE, "config": asdict(config)}, indent=2), encoding="utf-8")
        print(f"{name}: done, reconciliation {checks[-1]}")
    curves_by_name["S&P 500 TR"] = curves["S&P 500 TR"]
    rows.append({"portfolio": "S&P 500 TR", **metrics.loc["S&P 500 TR"].to_dict()})

    table = pd.DataFrame(rows).set_index("portfolio")
    strategies = table.drop(index="S&P 500 TR")
    chosen = choose(strategies.loc[STAGE_1])
    chosen_size = choose(strategies.loc[STAGE_2])
    table.to_csv(OUTPUT / "performance_comparison.csv")
    pd.DataFrame(checks).to_csv(OUTPUT / "reconciliations.csv", index=False)

    nav = pd.DataFrame(curves_by_name)
    charts = portfolio_chart_data(nav)
    charts["monthly_nav"].to_csv(OUTPUT / "monthly_nav.csv")
    figure, axis = plt.subplots(figsize=(11, 5.5))
    charts["monthly_nav"].plot(ax=axis, linewidth=1.4)
    axis.set_title("Model D portfolio variants: month-end account value, 2013-2015 (development)")
    axis.set_ylabel("Account value ($)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT / "monthly_nav.png", dpi=130)
    plt.close(figure)

    summary = (f"Stage 1 (stocks and schedule): {chosen}. Stage 2 (size filter): {chosen_size}. Rule: highest daily Sharpe; within {SHARPE_TIE} of it, "
               f"the smaller maximum drawdown. Development data only, {START} to {END}. "
               f"Runtime {perf_counter() - started:.0f} seconds.")
    (OUTPUT / "analysis.md").write_text(summary + "\n", encoding="utf-8")
    pd.set_option("display.width", 250)
    print(table.round(4).to_string())
    print(summary)


if __name__ == "__main__":
    main()
