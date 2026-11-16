"""Development-only top-500 entry comparison; never launches a final test.

Compare the original entry universe with one uniform rank cutoff, using exactly
the same fitted forecasts in each paired experiment. Annual folds recreate the
original model fits. A separate continuous 2013-2015 account holds one initial
fit throughout and preserves lots/cash across year-end, matching the fixed-fit
design of the original final strategy. Compare variants WITHIN each mode.

This revision was proposed after an original final-test coverage failure. It
is transparently a research revision, not an untouched original test selection.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from entry_universe import annotate_entry_ranks, select_entry_universe
from hw4.data import build_data, checksum
from hw4.engine import comparison, performance, run_backtest
from hw4.modeling import ModelSpec, forecast_metrics, predict_period
from hw4.settings import DATABASE, PROJECT, TradingRules
from run_experiments import save_run, versions, write_json

LOG = logging.getLogger(__name__)
RESULTS = PROJECT / "results/revision_top500_development"
LOCAL = PROJECT / "outputs/revision_top500_development"
CUTOFF = "2015-12-31"


def check_parent(parent):
    """Protect the source data and already-frozen original implementation."""
    configuration = parent["configuration"]
    if checksum(DATABASE) != configuration["source_sha256"]:
        raise ValueError("Source database changed")
    for name, digest in configuration["code_sha256"].items():
        if checksum(PROJECT / name) != digest:
            raise ValueError(f"Frozen original code changed: {name}")


def freeze_protocol(parent, spec, rules):
    """Write the scope and fixed comparison before calculating its results."""
    configuration = {
        "source_sha256": parent["configuration"]["source_sha256"],
        "parent_frozen_config_sha256": parent["sha256"],
        "model": parent["configuration"]["model"],
        "rules": rules.as_dict(),
        "data_end": CUTOFF,
        "annual_validation_years": [2013, 2014, 2015],
        "continuous_account": "2013-01-01 to 2015-12-31, one fit before 2013, no account resets",
        "only_change": "After generating common forecasts, restrict NEW entries to supplied universe_rank <=500 in the next-entry quarter.",
        "variants": {"original": 2000, "top500": 500},
        "training_population": "unchanged; do not refit on only the top 500",
        "unchanged": "features, predictor, training-label rules, risk screen, costs, allocation, caps, maturity and missing-price policy",
        "history_retention": "All quote histories remain available to mark/exit existing holdings after rank changes.",
        "no_selection_on": "current active status, future coverage, future targets, specific ticker exceptions, or final-period returns",
        "prior_test_exposure": "Original April 2016 account and May 2 HPY gap were observed; full benchmark history is known. This is a disclosed revision.",
        "selection_rule": "Review annual consistency, continuous-account return/drawdown, cash and data incidents together; do not choose another cutoff after a final-test gap.",
        "final_test": "Not run by this script. Instructor clarification on a revised final experiment remains outstanding.",
        "versions": versions(),
        "revision_code_sha256": {
            name: checksum(PROJECT / name)
            for name in ["entry_universe.py", "run_universe_comparison.py"]
        },
    }
    digest = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()
    path = RESULTS / "protocol.json"
    if path.exists():
        if json.loads(path.read_text())["sha256"] != digest:
            raise ValueError(
                "Revision protocol changed; preserve the existing experiment before revising it"
            )
    else:
        write_json(
            path,
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": digest,
                "configuration": configuration,
            },
        )


def independent_account_check(result, quotes, capital):
    """Rebuild units, cash and NAV from actual fills, without the simulator loop."""
    fills, history = result.fills, result.history
    if fills.empty:
        np.testing.assert_allclose(history.nav, capital)
        return {
            "nav_rows": len(history),
            "fills": 0,
            "max_nav_error": 0.0,
            "cross_year_open_lots": {},
        }
    signed = fills.units.where(fills.side.eq("BUY"), -fills.units)
    changes = (
        fills.assign(signed_units=signed)
        .groupby(["date", "symbol"])
        .signed_units.sum()
        .unstack(fill_value=0)
    )
    units = changes.reindex(history.index, fill_value=0).cumsum()
    assert (units.to_numpy() >= -1e-7).all()
    marks = quotes[units.columns].ffill(limit=5).reindex(history.index)
    assert not ((units > 1e-7) & marks.isna()).any().any()
    holdings = (units * marks).sum(axis=1)
    change = np.where(
        fills.side.eq("BUY"),
        -(fills.units * fills.price + fills.fee),
        fills.units * fills.price - fills.fee,
    )
    cash_changes = pd.Series(change, index=fills.date).groupby(level=0).sum()
    cash = capital + cash_changes.reindex(history.index, fill_value=0).cumsum()
    np.testing.assert_allclose(cash, history.cash, atol=1e-6)
    np.testing.assert_allclose(cash + holdings, history.nav, atol=1e-6)
    # Require every trade to use an observed source quote on its actual date.
    rows = quotes.index.get_indexer(fills.date)
    cols = quotes.columns.get_indexer(fills.symbol)
    assert (rows >= 0).all() and (cols >= 0).all()
    np.testing.assert_allclose(fills.price, quotes.to_numpy()[rows, cols], rtol=1e-12)
    buys = fills.loc[fills.side.eq("BUY")]
    sales = fills.loc[fills.side.eq("SELL")].merge(
        buys[["lot_id", "date"]], on="lot_id", suffixes=("", "_entry")
    )
    for sale in sales.itertuples():
        assert sale.maturity == sale.date_entry + pd.Timedelta(days=90)
        assert sale.date == quotes.loc[sale.maturity :, sale.symbol].dropna().index[0]
    return {
        "nav_rows": len(history),
        "fills": len(fills),
        "sales_checked": len(sales),
        "max_nav_error": float(abs(cash + holdings - history.nav).max()),
        "minimum_cash": float(cash.min()),
        "cross_year_open_lots": {
            date: int(history.loc[date, "open_lots"])
            for date in ["2013-12-31", "2014-12-31"]
            if pd.Timestamp(date) in history.index
        },
    }


def benchmark_account(data, start, end, capital):
    dates = data.benchmark.loc[start:end].index
    initial = data.benchmark.index[data.benchmark.index.get_loc(dates[0]) - 1]
    nav = capital * data.benchmark.loc[dates] / data.benchmark.loc[dates[0]]
    return pd.concat([pd.Series([capital], index=[initial]), nav])


def run():
    parent = json.loads((PROJECT / "results/final/frozen_config.json").read_text())
    check_parent(parent)
    spec = ModelSpec(**parent["configuration"]["model"])
    assert spec.name == "long_momentum" and not spec.quarterly
    rules = TradingRules(**parent["configuration"]["rules"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    freeze_protocol(parent, spec, rules)
    data = build_data(DATABASE, end=CUTOFF)
    assert data.quotes.index.max() <= pd.Timestamp(CUTOFF)
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        membership = con.execute(
            """SELECT symbol,quarter_start,selection_date,effective_start,universe_rank
            FROM universe_membership WHERE quarter_start<=? ORDER BY quarter_start,symbol""",
            [CUTOFF],
        ).fetchdf()
    old_results = pd.read_csv(PROJECT / "results/development_momentum/experiments.csv")
    periods = [(str(y), f"{y}-01-01", f"{y}-12-31") for y in [2013, 2014, 2015]]
    periods.append(("continuous_2013_2015", "2013-01-01", CUTOFF))
    rows, audits, candidate_counts, checks, incidents = [], [], [], {}, []
    continuous = {}
    for period, start, end in periods:
        LOG.info("Paired development comparison: %s", period)
        scores, audit, models = predict_period(
            data.panel, data.quotes.index, spec, start, end
        )
        assert scores.date.max() <= pd.Timestamp(CUTOFF)
        assert audit.last_training_label.lt(audit.fit_date).all()
        assert len(models) == 1  # Fixed fit within each independent period.
        audits.append(audit.assign(period=period))
        annotated = annotate_entry_ranks(scores, membership, data.quotes.index)
        assert len(annotated) == len(scores)
        shared_path = LOCAL / period
        shared_path.mkdir(parents=True, exist_ok=True)
        annotated.to_csv(shared_path / "shared_forecasts_with_ranks.csv", index=False)
        daily_counts = annotated.groupby("date").agg(
            original_candidates=("symbol", "size"),
            top500_candidates=("universe_rank", lambda x: x.le(500).sum()),
            original_censored_targets=("target", lambda x: x.isna().sum()),
        )
        daily_counts["period"] = period
        candidate_counts.append(daily_counts.reset_index())
        benchmark_nav = benchmark_account(data, start, end, rules.initial_cash)
        benchmark_stats = performance(benchmark_nav, rules.annual_rf)
        if period.startswith("continuous"):
            continuous["benchmark"] = benchmark_nav
        for variant, cutoff in [("original", 2000), ("top500", 500)]:
            filtered = select_entry_universe(annotated, cutoff)
            original_scores = annotated.set_index(["date", "symbol"]).score
            selected_keys = pd.MultiIndex.from_frame(filtered[["date", "symbol"]])
            np.testing.assert_array_equal(
                filtered.score, original_scores.reindex(selected_keys)
            )
            if variant == "original":
                assert len(filtered) == len(annotated)
            result = run_backtest(
                data.quotes, data.benchmark, filtered, start, end, rules
            )
            save_run(result, shared_path / variant)
            verification = independent_account_check(
                result, data.quotes, rules.initial_cash
            )
            # Recheck actual buys against their dated ranks; only new entries are screened.
            buys = result.fills.loc[result.fills.side.eq("BUY")]
            keys = pd.MultiIndex.from_frame(
                buys[["signal_date", "symbol"]].rename(columns={"signal_date": "date"})
            )
            entry_info = annotated.set_index(["date", "symbol"]).reindex(keys)
            assert (
                entry_info.universe_rank.notna().all()
                and entry_info.universe_rank.le(cutoff).all()
            )
            verification["entry_ranks_verified"] = len(buys)
            checks[f"{period}/{variant}"] = verification
            row = {
                "period": period,
                "variant": variant,
                "status": result.status,
                "last_valued_date": result.history.index[-1],
                "predictions": len(filtered),
                "predictions_with_unavailable_target": int(
                    filtered.target.isna().sum()
                ),
                "benchmark_return": benchmark_stats["total_return"],
                "benchmark_drawdown": benchmark_stats["max_drawdown"],
                **forecast_metrics(filtered, pd.Timestamp(end)),
            }
            if result.status == "complete":
                metrics = comparison(result, rules)
                row.update(metrics.loc["strategy"].to_dict())
                row["excess_return"] = row["total_return"] - row["benchmark_return"]
                row["average_cash_fraction"] = (
                    (result.history.cash / result.history.nav).iloc[1:].mean()
                )
                row["fees"] = result.history.fees.sum()
            if variant == "original" and period.isdigit():
                prior = old_results.loc[old_results.year.eq(int(period))].iloc[0]
                assert prior.status == result.status
                for field in [
                    "final_value",
                    "total_return",
                    "max_drawdown",
                    "Sharpe",
                    "fees",
                ]:
                    np.testing.assert_allclose(
                        row[field], prior[field], rtol=1e-10, atol=1e-8
                    )
                verification["original_annual_result_reproduced"] = True
            if not result.incidents.empty:
                incidents.append(
                    result.incidents.assign(period=period, variant=variant)
                )
            if period.startswith("continuous"):
                continuous[variant] = result.history.nav
                result.history.to_csv(RESULTS / f"continuous_{variant}_nav.csv")
            rows.append(row)
            pd.DataFrame(rows).to_csv(RESULTS / "experiments.csv", index=False)
            LOG.info(
                "%s / %s: %s through %s",
                period,
                variant,
                result.status,
                result.history.index[-1].date(),
            )
    pd.concat(audits, ignore_index=True).to_csv(RESULTS / "fit_audit.csv", index=False)
    pd.concat(candidate_counts, ignore_index=True).to_csv(
        RESULTS / "candidate_counts.csv", index=False
    )
    incident_frame = (
        pd.concat(incidents, ignore_index=True)
        if incidents
        else pd.DataFrame(columns=["date", "symbol", "event", "period", "variant"])
    )
    incident_frame.to_csv(RESULTS / "incidents.csv", index=False)
    write_json(
        RESULTS / "verification.json",
        {
            "accounts": checks,
            "source_and_original_code_unchanged": True,
            "new_final_test_executed": False,
            "stock_price_cutoff": str(data.quotes.index.max().date()),
            "identical_common_forecasts": True,
        },
    )
    table = pd.DataFrame(rows)
    figure, axes = plt.subplots(2, 1, figsize=(11, 8))
    annual = table.loc[table.period.str.isdigit()]
    bars = annual.pivot(index="period", columns="variant", values="total_return")
    bars["benchmark"] = annual.groupby("period").benchmark_return.first()
    (100 * bars).plot.bar(ax=axes[0], rot=0)
    axes[0].set(
        title="Annual development accounts — only the entry rank cutoff changes",
        ylabel="Total return (%)",
        xlabel="Year",
    )
    axes[0].axhline(0, color="black", linewidth=0.8)
    for name, nav in continuous.items():
        axes[1].plot(nav.index, 100 * (nav / rules.initial_cash - 1), label=name)
    axes[1].set(
        title="Continuous development: one initial fit, holdings carried across year-end",
        ylabel="Cumulative return (%)",
        xlabel="Date",
    )
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    figure.tight_layout()
    figure.savefig(RESULTS / "development_comparison.png", dpi=150)
    plt.close(figure)
    check_parent(parent)
    print(
        table[
            [
                "period",
                "variant",
                "status",
                "total_return",
                "benchmark_return",
                "max_drawdown",
                "Sharpe",
                "average_cash_fraction",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
