"""Compare candidate signals on development data, or run the chosen strategy on the final test."""

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import sklearn

import data
import model
import report
from backtest import FEE_RATE, HOLDING_DAYS, INITIAL_CAPITAL, TopNStrategy, run_backtest


# Development choices, recorded here before the final test is run.
CHOSEN = "Equal-weight composite"
TOP_N = 20
SLEEVE_FRACTION = 1 / 3
# Used only for the development comparison.
VALIDATION_YEARS = (2013, 2014, 2015)
RIDGE = "Ridge-weighted composite"
ALPHA = 100.0
SMALLER_SLEEVE = 0.25
BENCHMARK = "S&P 500 Total Return"


def year_end_signals(calendar, years):
    """Return the signal before each year's first session, when that year's weights are fitted."""
    signals = data.monthly_signals(calendar)
    return [signals[signals < f"{year}-01-01"][-1] for year in years]


def simulate(scores, quotes, first_signal, sleeve_fraction=SLEEVE_FRACTION):
    """Run one score table through the shared trading rules."""
    strategy = TopNStrategy(model.to_rankings(scores), TOP_N, sleeve_fraction)
    return run_backtest(quotes, strategy, first_signal)


def benchmark_nav(market, sessions):
    """Invest the starting capital in the benchmark at the first close."""
    return INITIAL_CAPITAL * market.loc[sessions] / market.loc[sessions[0]]


def settings():
    """List the frozen choices and package versions behind a run."""
    components = ", ".join(label for label, _ in model.COMPONENTS.values())
    return pd.Series({
        "Signal": f"{CHOSEN} of the ranks of {components}",
        "Portfolio": f"Top {TOP_N} names monthly, sleeve of {SLEEVE_FRACTION:.1%} of NAV, equal budgets",
        "Holding period and fee": f"{HOLDING_DAYS} calendar days, {FEE_RATE * 1e4:.0f} bp per side",
        "Liquidity screen": f"63-session median dollar volume of at least ${data.MIN_DOLLAR_VOLUME:,}",
        "Price-break rule": f"Moves beyond {data.MAX_SESSION_MOVE:g}x between observed closes are checked against "
                            "price_events.csv. Unexplained ones end that stock's quotes",
        "Packages": f"pandas {pd.__version__}, numpy {np.__version__}, scikit-learn {sklearn.__version__}, "
                    f"duckdb {duckdb.__version__}",
    }).to_frame("Value")


def development_candidates(ranked, fit_dates):
    """Score the single components, their equal-weight composite, and the Ridge-weighted version."""
    window = ranked.loc[ranked.date.ge(fit_dates[0])]
    candidates = {label: model.component_scores(window, name) for name, (label, _) in model.COMPONENTS.items()}
    candidates[CHOSEN] = model.composite_scores(window)
    candidates[RIDGE], audit = model.walk_forward_scores(ranked, ALPHA, fit_dates)
    return candidates, audit


def run_development(database, output):
    """Validate every candidate on 2013-2015 with identical trading rules."""
    start = f"{VALIDATION_YEARS[0]}-01-01"
    panel, market, quotes, _ = data.load_research_data(database, data.DEV_END, start)
    fit_dates = year_end_signals(market.index, VALIDATION_YEARS)
    candidates, audit = development_candidates(model.add_ranks(panel), fit_dates)
    navs, cash = {}, {}
    for name, scores in candidates.items():
        daily = simulate(scores, quotes, fit_dates[0]).daily
        navs[name], cash[name] = daily.nav, (daily.cash / daily.nav).mean()
    smaller = simulate(candidates[CHOSEN], quotes, fit_dates[0], SMALLER_SLEEVE).daily
    label = f"{CHOSEN}, {SMALLER_SLEEVE:.0%} sleeve"
    navs[label], cash[label] = smaller.nav, (smaller.cash / smaller.nav).mean()
    navs = pd.DataFrame(navs).assign(**{BENCHMARK: benchmark_nav(market, quotes.index)})
    rank_ic = pd.DataFrame({name: model.rank_ic_by_year(scores) for name, scores in candidates.items()}).T
    rank_ic["Mean"] = rank_ic.mean(axis=1)
    _, table = report.metrics_table(navs)
    table["Average cash"] = pd.Series(cash).map("{:.1%}".format).reindex(table.index, fill_value="")
    audit = audit.apply(lambda column: column.dt.strftime("%Y-%m-%d") if column.dtype.kind == "M"
                        else column.map("{:,}".format) if column.dtype.kind == "i" else column.map("{:.0%}".format))
    report.plot_cumulative(navs[[CHOSEN, RIDGE, BENCHMARK]], output / "development_returns.png",
                           "Development backtests, 2013-2015")
    report.write_report(output / "development.md", "HW4 development comparison", [
        (None, [f"Signals from {fit_dates[0]:%Y-%m-%d} to {panel.date.max():%Y-%m-%d}, data through {data.DEV_END} "
                f"only. Every portfolio buys the top {TOP_N} names monthly with a {SLEEVE_FRACTION:.1%} sleeve unless "
                "stated. The Ridge weights are refitted before each validation year on outcomes that had ended.", "",
                report.label_coverage(panel, data.DEV_END), ""]),
        ("Mean monthly rank IC by entry year", report.markdown_table(rank_ic.map("{:+.4f}".format), "Signal")),
        ("Portfolio results", [*report.markdown_table(table, "Portfolio"),
                               "![Development cumulative returns](development_returns.png)", ""]),
        ("Ridge weights learned at each fit", report.markdown_table(audit.set_index("fit_date"), "fit_date")),
    ])
    print(rank_ic.round(4).to_string(), table.to_string(), sep="\n\n")


def run_test(database, output, start=data.TEST_START, end=data.TEST_END):
    """Run the chosen strategy once on April 2016 to September 2026."""
    panel, market, quotes, breaks = data.load_research_data(database, end, start)
    first_signal = market.index[market.index.get_loc(quotes.index[0]) - 1]
    scores = model.composite_scores(model.add_ranks(panel.loc[panel.date.ge(first_signal)]))
    result = simulate(scores, quotes, first_signal)
    benchmark = benchmark_nav(market, quotes.index)
    navs = pd.DataFrame({"Strategy": result.daily.nav, BENCHMARK: benchmark})
    metrics, table = report.metrics_table(navs)
    gap = metrics.loc["cagr", "Strategy"] - metrics.loc["cagr", BENCHMARK]
    value_gap = metrics.loc["final_value", "Strategy"] - metrics.loc["final_value", BENCHMARK]
    verdict = (f"The strategy finished {'ahead of' if gap > 0 else 'behind'} the benchmark by "
               f"{abs(gap) * 100:.2f} percentage points of CAGR and ${abs(value_gap):,.0f} of final value.")
    report.plot_cumulative(navs, output / "cumulative_returns.png", "HW4 final test, cumulative returns")
    report.write_report(output / "results.md", "HW4 final test results", [
        ("Settings", report.markdown_table(settings(), "Item")),
        ("Performance", [*report.markdown_table(table.T, "Metric"), verdict, "",
                         "![Cumulative returns](cumulative_returns.png)", ""]),
        ("Run diagnostics", [*report.markdown_table(report.run_diagnostics(result).to_frame("Strategy"), "Item"),
                             report.label_coverage(panel.loc[panel.date.ge(first_signal)], end), ""]),
        ("Holdings carried at an older price", report.frozen_holdings(result, quotes.index, breaks)),
        ("Checks on this run", report.markdown_table(report.reconciliation(result, benchmark), "Check")),
    ])
    details = output / "outputs"
    details.mkdir(exist_ok=True)
    result.daily.to_csv(details / "test_daily.csv")
    result.fills.to_csv(details / "test_fills.csv", index=False)
    result.events.to_csv(details / "test_events.csv", index=False)
    print(table.T.to_string(), verdict, sep="\n\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=data.DATABASE)
    parser.add_argument("--final-test", action="store_true",
                        help="Run the chosen strategy on the final test instead of the development comparison.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if not args.database.is_file():
        parser.error(f"Download the database from bCourses to {args.database}, or pass its path with --database.")
    args.output.mkdir(parents=True, exist_ok=True)
    (run_test if args.final_test else run_development)(args.database, args.output)


if __name__ == "__main__":
    main()
