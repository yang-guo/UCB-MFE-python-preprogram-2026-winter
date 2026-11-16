"""Paired stock-only versus stock-plus-sector development allocation experiment.

Shared model fits and forecasts isolate allocation. Every comparison uses the
same reviewed development input layer; annual baseline NAV must reproduce the
saved baseline before its sector variant is accepted. No post-2015 run occurs.
"""

import json

import duckdb
import numpy as np
import pandas as pd

from development_revision import corrected_data, remove_fluidigm_alias
from hw4.data import build_data, checksum
from hw4.engine import comparison
from hw4.modeling import MOMENTUM_SPECS, predict_period
from hw4.settings import DATABASE, PROJECT, TradingRules
from review_sector_risk import holdings_exposure
from run_development_revision import verify_account
from run_experiments import save_run, write_json
from run_universe_comparison import check_parent
from sector_engine import run_sector_backtest

OUT = PROJECT / "results/sector_caps"
LOCAL = PROJECT / "outputs/sector_caps"
VARIANTS = [
    {
        "name": "A_stock_only",
        "sector_cap": None,
        "parent": "development_revision/D2_alias_removed",
        "change": "Replay the reviewed baseline with the original 5% aggregate stock entry cap and no sector cap.",
    },
    {
        "name": "B_sector20",
        "sector_cap": 0.20,
        "parent": "sector_caps/A_stock_only",
        "change": "Keep the 5% stock cap and add a 20% aggregate sector entry cap. Proportionally reduce constrained-sector orders; leave blocked budget in cash.",
    },
]


def verify_entry_caps(result, quotes, sectors, rules, sector_cap):
    """Independently check actual purchase notionals against prior-close exposure.

    Do not subtract today's scheduled sales: the original planner conservatively
    sizes new entries using holdings and cash already observed yesterday.
    """
    f = result.fills.copy()
    f["signed_units"] = np.where(f.side.eq("BUY"), f.units, -f.units)
    units = f.groupby(["date", "symbol"]).signed_units.sum().unstack(fill_value=0)
    units = units.reindex(result.history.index, fill_value=0).cumsum()
    marks = quotes[units.columns].ffill(limit=5).reindex(result.history.index)
    values = (units * marks).fillna(0)
    maximum_planned_sector_weight = 0.0
    for day, buys in f.loc[f.side.eq("BUY")].groupby("date"):
        before = result.history.index[result.history.index.get_loc(day) - 1]
        nav, cash = result.history.loc[before, ["nav", "cash"]]
        assert -buys.cash_change.sum() <= cash + 1e-6
        held = values.loc[before]
        incoming = buys.groupby("symbol").notional.sum()
        for symbol, amount in incoming.items():
            assert (
                held.get(symbol, 0.0) + amount
                <= max(rules.name_cap * nav, held.get(symbol, 0.0)) + 1e-6
            )
        if sector_cap is not None:
            old_sectors = held.groupby(sectors).sum()
            new_sectors = incoming.groupby(sectors).sum()
            for sector, amount in new_sectors.items():
                planned = old_sectors.get(sector, 0.0) + amount
                assert (
                    planned
                    <= max(sector_cap * nav, old_sectors.get(sector, 0.0)) + 1e-6
                )
                maximum_planned_sector_weight = max(
                    maximum_planned_sector_weight, planned / nav
                )
    return maximum_planned_sector_weight


def run():
    parent = json.loads((PROJECT / "results/final/frozen_config.json").read_text())
    check_parent(parent)
    original_model_hash = checksum(PROJECT / "artifacts/model.joblib")
    original = build_data(DATABASE, end="2015-12-31")
    data, _, _ = corrected_data(original)
    data, _ = remove_fluidigm_alias(data)
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        sectors = (
            con.execute("SELECT symbol,sector FROM universe ORDER BY symbol")
            .df()
            .set_index("symbol")
            .sector
        )
    assert sectors.notna().all() and not sectors.index.has_duplicates
    rules = TradingRules()
    protocol = {
        "data_end": "2015-12-31",
        "source_sha256": checksum(DATABASE),
        "variants": VARIANTS,
        "validation_years": [2013, 2014, 2015],
        "continuous_account": "2013-2015, one pre-2013 fit, no year-end reset",
        "model": "Unchanged one-feature long_momentum Ridge; common fits/forecasts across allocation variants.",
        "data_layer": "development_revision D2: supported price overlays and scoped LAB alias removal",
        "rules": rules.as_dict(),
        "sector_labels": "Static vendor snapshot, not certified historical sector classifications.",
        "cap_timing": "Prior-close NAV and aggregate open holdings plus planned new notional. Price moves/fees can put observed weights over the entry cap. No early sales.",
        "sector_budget_method": "Scale proposed orders proportionally within constrained sectors; no replacement picks or reallocation of unused budget.",
        "third_comparison": "Pending user choice; first two comparisons proceed independently.",
        "code_sha256": {
            name: checksum(PROJECT / name)
            for name in [
                "sector_allocation.py",
                "sector_engine.py",
                "run_sector_comparison.py",
            ]
        },
    }
    write_json(OUT / "protocol.json", protocol)
    rows, checks, audits = [], [], []
    periods = [(str(y), f"{y}-01-01", f"{y}-12-31") for y in [2013, 2014, 2015]]
    periods.append(("continuous_2013_2015", "2013-01-01", "2015-12-31"))
    for period, start, end in periods:
        scores, audit, _ = predict_period(
            data.panel, data.quotes.index, MOMENTUM_SPECS[0], start, end
        )
        (LOCAL / period).mkdir(parents=True, exist_ok=True)
        scores.to_csv(LOCAL / period / "shared_forecasts.csv", index=False)
        audits.append(audit.assign(period=period))
        for variant in VARIANTS:
            name = variant["name"]
            result = run_sector_backtest(
                data.quotes,
                data.benchmark,
                scores,
                start,
                end,
                rules,
                sectors=sectors.to_dict(),
                sector_cap=variant["sector_cap"],
            )
            save_run(result, LOCAL / period / name)
            row = {
                "variant": name,
                "period": period,
                "status": result.status,
                "last_valued_date": result.history.index[-1],
                "parent": variant["parent"],
                "change": variant["change"],
            }
            if result.status == "complete":
                nav_error = verify_account(result, data.quotes, rules.initial_cash)
                planned_cap = verify_entry_caps(
                    result, data.quotes, sectors, rules, variant["sector_cap"]
                )
                if name == "A_stock_only":
                    old = pd.read_csv(
                        PROJECT
                        / f"results/development_revision/{period}_D2_alias_removed_nav.csv",
                        index_col="date",
                        parse_dates=True,
                    )
                    np.testing.assert_allclose(
                        result.history.nav, old.nav, rtol=0, atol=1e-6
                    )
                metric = comparison(result, rules)
                weights = holdings_exposure(
                    result.fills, result.history, data.quotes, sectors
                )
                weights.to_csv(OUT / f"{period}_{name}_sector_weights.csv")
                result.history.to_csv(OUT / f"{period}_{name}_nav.csv")
                row.update(metric.loc["strategy"].to_dict())
                row.update(
                    benchmark_return=metric.loc["benchmark", "total_return"],
                    average_cash_fraction=(result.history.cash / result.history.nav)
                    .iloc[1:]
                    .mean(),
                    fees=result.fills.fee.sum(),
                    buys=int(result.fills.side.eq("BUY").sum()),
                    peak_sector_weight=weights.iloc[1:].max().max(),
                    peak_energy_weight=weights.Energy.iloc[1:].max(),
                    mean_energy_weight=weights.Energy.iloc[1:].mean(),
                    planned_sector_weight_max=planned_cap,
                )
                # A reconciled price ledger is not automatically verified total return.
                incidents = result.incidents
                unresolved = (
                    not incidents.empty
                    and (
                        (incidents.get("symbol", pd.Series(dtype=str)).eq("GYRE"))
                        & (pd.to_datetime(incidents.date).eq("2015-08-20"))
                    ).any()
                )
                row["specific_open_data_issue"] = (
                    "GYRE/Targacept distribution requires reconciliation"
                    if unresolved
                    else ""
                )
                checks.append(
                    {
                        "period": period,
                        "variant": name,
                        "max_nav_error": nav_error,
                        "entry_cap_checks_passed": True,
                        "maturity_checks_passed": True,
                    }
                )
            else:
                row["incident"] = json.dumps(
                    result.incidents.tail(1).to_dict("records"), default=str
                )
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT / "experiments.csv", index=False)
            print(
                period,
                name,
                result.status,
                f"return={row.get('total_return', float('nan')):.2%}",
                flush=True,
            )
    pd.concat(audits, ignore_index=True).to_csv(OUT / "fit_audit.csv", index=False)
    write_json(
        OUT / "verification.json",
        {
            "accounts": checks,
            "source_unchanged": checksum(DATABASE) == protocol["source_sha256"],
            "original_model_unchanged": checksum(PROJECT / "artifacts/model.joblib")
            == original_model_hash,
            "new_post2015_run": False,
        },
    )
    check_parent(parent)


if __name__ == "__main__":
    run()
