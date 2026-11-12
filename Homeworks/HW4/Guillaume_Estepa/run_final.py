"""Step 2 - final test (2016-04-01 to 2026-09-30), run once.

Uses the configuration frozen by ``run_development.py``. The model is first
fitted at the 2016-03-31 close on outcomes that had already ended, then
refitted at the first signal date of each calendar year with the outcomes
completed by then.

Run from the repository root:
    uv run python Homeworks/HW4/Guillaume_Estepa/run_final.py
"""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from backtester import Backtester
from data import CLASS_DB, FULL_DB, load_market_data
from features import FEATURES, build_panel, weekly_grid
from metrics import buy_and_hold, performance_metrics, rank_ic
from model import refit_dates, walk_forward_predict
from strategy import TopNTrancheStrategy, build_schedule

RESULTS = Path(__file__).parent / "results"
TEST_START, TEST_END = "2016-04-01", "2026-09-30"


def plot_cumulative_returns(strategy_values, benchmark_values, sensitivity_values, path):
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
    for values, label, color in [(strategy_values, "Strategy (top-50 ridge, 3 tranches)", "#1f77b4"),
                                 (benchmark_values, "S&P 500 total return (^SP500TR)", "#555555")]:
        top.plot(values.index, (values / values.iloc[0] - 1) * 100, label=label, color=color, lw=1.4)
        bottom.plot(values.index, (values / values.cummax() - 1) * 100, color=color, lw=1.0)
    top.plot(sensitivity_values.index, (sensitivity_values / sensitivity_values.iloc[0] - 1) * 100,
             label="Sensitivity: acquired holdings cashed out at last quote", color="#1f77b4",
             lw=1.0, ls="--", alpha=0.7)
    top.set_ylabel("Cumulative return (%)")
    top.set_title("$1,000,000 invested at the 2016-04-01 close, through 2026-09-30")
    top.legend(frameon=False, loc="upper left")
    bottom.set_ylabel("Drawdown (%)")
    for axis in (top, bottom):
        axis.grid(alpha=0.3)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(database=FULL_DB):
    config = json.loads((RESULTS / "chosen_config.json").read_text())
    md = load_market_data(database, end=TEST_END)
    sessions = md.sessions
    first_entry = sessions[sessions.searchsorted(pd.Timestamp(TEST_START))]
    first_signal = sessions[sessions.get_loc(first_entry) - 1]

    schedule = build_schedule(sessions, first_signal, TEST_END, config["n_tranches"])
    panel = build_panel(md, weekly_grid(sessions).union(pd.DatetimeIndex(list(schedule))))
    fits = refit_dates(panel.date, first_signal)
    panel["prediction"], fit_log, models = walk_forward_predict(panel, config["model"], fits)
    # Look-ahead guard: every fit only saw outcomes that ended before it.
    assert (fit_log.last_label_end < fit_log.fit_date).all()
    print("Walk-forward fits\n", fit_log.to_string(index=False))

    strategy = TopNTrancheStrategy(panel, schedule, config["n_stocks"], config["n_tranches"],
                                   config["max_liquidity_rank"], config["sector_cap"])
    result = Backtester(md.adj_close, first_entry, TEST_END, cost_rate=config["cost_rate"]).run(strategy)
    values = result.daily.total_value.rename("strategy")
    benchmark = buy_and_hold(md.benchmark, first_entry, TEST_END)
    assert values.index.equals(benchmark.index)

    # ---- required comparison
    table = pd.DataFrame({"strategy": performance_metrics(values),
                          "benchmark": performance_metrics(benchmark)})
    table["difference"] = table.strategy - table.benchmark
    print("\nPerformance, first test close to 2026-09-30\n", table.round(4).to_string())
    table.to_csv(RESULTS / "performance.csv")

    # ---- sensitivity (NOT the headline): the price history of acquired
    # companies simply ends, so their lots can never be sold and that capital
    # sits idle. Here such lots are assumed to pay out their last observed
    # price once matured and unquoted for 10 sessions. Same model and picks.
    strategy_s = TopNTrancheStrategy(panel, schedule, config["n_stocks"], config["n_tranches"],
                                     config["max_liquidity_rank"], config["sector_cap"])
    result_s = Backtester(md.adj_close, first_entry, TEST_END, cost_rate=config["cost_rate"],
                          assume_cash_out_after=10).run(strategy_s)
    sensitivity = pd.DataFrame({
        "strategy_as_reported": performance_metrics(values),
        "strategy_assumed_cash_out": performance_metrics(result_s.daily.total_value),
        "benchmark": performance_metrics(benchmark)})
    sensitivity.to_csv(RESULTS / "sensitivity.csv")
    print(f"\nSensitivity ({result_s.audit['n_assumed_cash_outs']} assumed cash-outs)\n",
          sensitivity.round(4).to_string())
    plot_cumulative_returns(values, benchmark, result_s.daily.total_value,
                            RESULTS / "cumulative_returns.png")

    yearly = pd.DataFrame({"strategy": values, "benchmark": benchmark}).groupby(values.index.year).last()
    yearly = yearly.div(yearly.shift().fillna(1_000_000.0)) - 1
    yearly["excess"] = yearly.strategy - yearly.benchmark
    yearly.to_csv(RESULTS / "yearly_returns.csv")
    print("\nCalendar-year returns\n", yearly.round(4).to_string())

    # ---- accounting and timing evidence
    failed = [name for name, ok in result.audit.items() if isinstance(ok, bool) and not ok]
    print("\nAudit\n", json.dumps(result.audit, indent=2, default=str))
    assert not failed, f"Audit failed: {failed}"
    (RESULTS / "audit.json").write_text(json.dumps(result.audit, indent=2, default=str) + "\n")

    buys = result.trades[result.trades.side == "BUY"].set_index("lot_id")
    sells = result.trades[result.trades.side == "SELL"].set_index("lot_id")
    lots = buys[["symbol", "signal_date", "date", "price", "units", "cash_flow", "due_date"]].rename(
        columns={"date": "entry_date", "price": "entry_price", "cash_flow": "cash_paid"})
    lots["exit_date"], lots["exit_price"] = sells.date, sells.price
    lots["cash_received"] = sells.cash_flow
    lots["net_return"] = lots.cash_received / -lots.cash_paid - 1
    lots.to_csv(RESULTS / "lots.csv", float_format="%.6f")
    result.daily.to_csv(RESULTS / "daily_values.csv", float_format="%.2f")
    benchmark.to_csv(RESULTS / "benchmark_values.csv", float_format="%.2f")
    result.rejected.to_csv(RESULTS / "rejected_orders.csv", index=False)
    print("\nFirst three lots (hand-check these against the prices table)\n",
          lots.head(3).T.to_string())

    # ---- diagnostics for the write-up
    closed = lots.dropna(subset=["net_return"])
    extreme = closed[(closed.net_return > 2.0) | (closed.net_return < -0.8)]
    pnl = closed.cash_received + closed.cash_paid
    diagnostics = {
        "closed_lots": len(closed), "win_rate": float((closed.net_return > 0).mean()),
        "mean_lot_return": float(closed.net_return.mean()),
        "median_lot_return": float(closed.net_return.median()),
        "distinct_stocks_bought": int(lots.symbol.nunique()),
        "average_cash_share": float((result.daily.cash / result.daily.total_value).iloc[1:].mean()),
        "lots_with_extreme_return": len(extreme),
        "pnl_from_extreme_lots": float(pnl[extreme.index].sum()),
        "total_pnl": float(values.iloc[-1] - values.iloc[0]),
        "open_lots_value_at_end": float(result.daily.positions.iloc[-1]),
        "cash_at_end": float(result.daily.cash.iloc[-1]),
        "overdue_unsellable_lots_at_end": int((lots.exit_date.isna() & (lots.due_date <= values.index[-1])).sum()),
        "overdue_unsellable_value_at_end": float(result.daily.overdue_value.iloc[-1]),
    }
    (RESULTS / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    print("\nDiagnostics\n", json.dumps(diagnostics, indent=2))
    if len(extreme):
        print("\nLots with extreme returns (possible data problems)\n", extreme.to_string())

    ic = rank_ic(panel[panel.date >= first_signal])
    ic_by_year = ic.groupby(ic.index.year).agg(["mean", lambda s: (s > 0).mean(), "count"])
    ic_by_year.columns = ["mean_rank_ic", "hit_rate", "dates"]
    ic_by_year.to_csv(RESULTS / "test_rank_ic.csv")
    print("\nOut-of-sample rank IC by year\n", ic_by_year.round(4).to_string(),
          f"\nOverall mean rank IC: {ic.mean():.4f}")

    if hasattr(next(iter(models.values())), "coef_"):
        coefficients = pd.DataFrame({date.date(): model.coef_ for date, model in models.items()},
                                    index=FEATURES)
        coefficients.to_csv(RESULTS / "ridge_coefficients.csv", float_format="%.5f")
        print("\nRidge coefficients (first and last fit)\n",
              coefficients.iloc[:, [0, -1]].sort_values(coefficients.columns[-1]).round(4).to_string())
    fit_log.to_csv(RESULTS / "fit_log.csv", index=False)


if __name__ == "__main__":
    main(CLASS_DB if "--class-db" in sys.argv else FULL_DB)
