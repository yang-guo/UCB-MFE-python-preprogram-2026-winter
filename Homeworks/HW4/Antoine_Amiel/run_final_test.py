"""Step 2 – final test, Apr 1, 2016 .. Sep 30, 2026, with the frozen configuration.

Reads results/chosen_config.json (written after development, before this
script was ever run) and writes results/final/*.

    python run_final_test.py                    # ~6 minutes
    python run_final_test.py --fee-sensitivity  # also reruns at 0 and 25 bps (~15 minutes)
"""

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hw4.backtester import Backtester, accounting_checks, performance  # noqa: E402
from hw4.data import load_market_data  # noqa: E402
from hw4.features import FeatureStore  # noqa: E402
from hw4.model import WalkForwardScorer, build_panel, glitch_flags, training_dates  # noqa: E402
from hw4.strategy import SleeveTopN  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
SETTLE = "--settle-stale" in sys.argv   # sensitivity: close lots of stocks that stopped quoting
OUT = HERE / ("results/final_settle_stale_sensitivity" if SETTLE else "results/final")
TEST_START, TEST_END = pd.Timestamp("2016-04-01"), pd.Timestamp("2026-09-30")


def quarterly_fit_dates(calendar, start, end):
    """Fit after the close of the last session before the first entry, then
    at the first session of every later quarter."""
    first_signal = calendar[calendar.get_loc(start) - 1]
    later = [calendar[calendar >= q][0] for q in pd.date_range(start, end, freq="QS") if q > start]
    return [first_signal] + [d for d in later if d <= end]


def run(config, store, panel, glitch, fee_bps):
    fit_dates = quarterly_fit_dates(store.calendar, TEST_START, TEST_END)
    scorer = WalkForwardScorer(store, panel, config["model"], fit_dates, config["max_rank"], glitch)
    strategy = SleeveTopN(scorer, store.calendar, TEST_START, n_names=config["names_per_sleeve"],
                          n_sleeves=config["sleeves"], sleeve_gap=config["sleeve_gap_sessions"],
                          name_cap=config["name_cap"], weighting=config["weighting"])
    bt = Backtester(store.md.adj.loc[:TEST_END], strategy, initial_cash=config["initial_cash"], fee_bps=fee_bps,
                    settle_after_sessions=10 if SETTLE else None)
    return bt.run(TEST_START, TEST_END), scorer, strategy


def main():
    config = json.loads((HERE / "results/chosen_config.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    md = load_market_data(end=str(TEST_END.date()))
    store = FeatureStore(md)
    glitch = glitch_flags(store)
    panel = build_panel(store, training_dates(store.calendar, "2010-01-01", TEST_END))
    print(f"data + features + panel: {time.time() - t0:.0f}s, panel rows {len(panel):,}", flush=True)

    res, scorer, strategy = run(config, store, panel, glitch, config["fee_bps"])
    h = res.history
    bench = md.benchmark.loc[TEST_START:TEST_END]
    h["benchmark_nav"] = float(config["initial_cash"])   # cash before the first test close
    h.loc[bench.index, "benchmark_nav"] = config["initial_cash"] * bench / bench.iloc[0]

    perf = pd.DataFrame({"strategy": performance(h.nav), "S&P 500 TR": performance(h.benchmark_nav)})
    perf.loc["CAGR_difference", "strategy"] = perf.loc["CAGR", "strategy"] - perf.loc["CAGR", "S&P 500 TR"]
    perf.to_csv(OUT / "performance.csv")
    print(perf.round(4).to_string(), flush=True)

    checks = accounting_checks(res, store.md.adj.loc[:TEST_END], config["initial_cash"])
    checks.to_csv(OUT / "accounting_checks.csv", header=["value"])
    print(checks.to_string(), flush=True)

    # Diagnostics, sensitivities and the full audit trail.
    years = h.nav.resample("YE").last()
    annual = pd.DataFrame({"strategy": years.pct_change(), "S&P 500 TR": h.benchmark_nav.resample("YE").last().pct_change()})
    annual.iloc[0] = [years.iloc[0] / config["initial_cash"] - 1,
                      h.benchmark_nav.resample("YE").last().iloc[0] / config["initial_cash"] - 1]
    annual.index = annual.index.year
    annual.to_csv(OUT / "annual_returns.csv")
    h.to_csv(OUT / "daily_nav.csv")
    res.fills.to_csv(OUT / "fills.csv", index=False)
    res.open_lots.to_csv(OUT / "open_lots_at_end.csv", index=False)
    # One row per (lot or order, event type) instead of one row per day.
    res.events.assign(event=res.events.event.str.replace(r"_\d+_sessions", "", regex=True)) \
        .groupby(["symbol", "event"]).date.agg(["min", "max", "count"]).reset_index() \
        .to_csv(OUT / "events_summary.csv", index=False)
    res.incidents.to_csv(OUT / "price_incidents.csv", index=False)
    pd.DataFrame(scorer.audit).to_csv(OUT / "fit_audit.csv", index=False)
    pd.DataFrame(strategy.decisions).to_csv(OUT / "decisions.csv", index=False)

    last_quote = store.md.adj.loc[:TEST_END].apply(lambda c: c.last_valid_index())
    open_lots = res.open_lots.assign(last_quote=lambda x: x.symbol.map(last_quote),
                                     value=lambda x: x.units * x.symbol.map(res.last_marks))
    stale_value = open_lots.loc[open_lots.last_quote < TEST_END, "value"].sum()
    sensitivity = {
        "final_value": h.nav.iloc[-1],
        "final_value_if_stale_open_lots_worth_zero": h.nav.iloc[-1] - stale_value,
        "price_incidents": len(res.incidents),
        "final_value_without_incident_jumps": h.nav.iloc[-1] - res.incidents.pnl_impact.sum(),
        "total_fees": res.fills.fee.sum(),
        "average_cash_fraction_after_ramp_up": (h.cash / h.nav).iloc[63:].mean(),
        "average_open_lots": h.open_lots.iloc[63:].mean(),
        "turnover_bought_over_avg_nav_per_year": h.bought.sum() / h.nav.mean() / ((TEST_END - TEST_START).days / 365.25),
        "buy_orders_rejected_no_quote": int((res.events.event == "buy_rejected_no_quote").sum()),
        "exits_delayed_no_quote": int((res.events.event == "exit_delayed_no_quote").sum()),
    }
    for bps in ([0, 25] if "--fee-sensitivity" in sys.argv else []):
        alt, _, _ = run(config, store, panel, glitch, bps)
        sensitivity[f"final_value_fee_{bps}bps"] = alt.history.nav.iloc[-1]
        sensitivity[f"CAGR_fee_{bps}bps"] = performance(alt.history.nav).CAGR
    pd.Series(sensitivity).to_csv(OUT / "sensitivity.csv", header=["value"])
    print(pd.Series(sensitivity).to_string(), flush=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    (h.nav / config["initial_cash"] - 1).plot(ax=ax, label="Strategy")
    (h.benchmark_nav / config["initial_cash"] - 1).plot(ax=ax, label="S&P 500 total return", color="black", ls="--")
    ax.set(title="Cumulative return, Apr 2016 – Sep 2026 ($1,000,000 start)", ylabel="Cumulative return", xlabel="")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cumulative_returns.png", dpi=150)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
