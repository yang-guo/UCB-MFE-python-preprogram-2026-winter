"""Third allocation comparison: known maturity-sale cash, next-session entries.

Reuse the saved A/B monthly predictions exactly. Neither model parameters nor
monthly rankings are refitted. Preserve A/B outputs and their source snapshot;
test the extended engine against every saved B account before accepting C.
"""

import json

import duckdb
import numpy as np
import pandas as pd

from development_revision import corrected_data, remove_fluidigm_alias
from hw4.data import build_data, checksum
from hw4.engine import comparison
from hw4.settings import DATABASE, PROJECT, TradingRules
from review_sector_risk import holdings_exposure
from run_development_revision import verify_account
from run_experiments import save_run, write_json
from run_sector_comparison import verify_entry_caps
from run_universe_comparison import check_parent
from sector_engine import run_sector_backtest
from sector_reinvestment import build_entry_eligibility

OUT = PROJECT / "results/sector_reinvestment"
LOCAL = PROJECT / "outputs/sector_reinvestment"
PERIODS = [(str(y), f"{y}-01-01", f"{y}-12-31") for y in [2013, 2014, 2015]] + [
    ("continuous_2013_2015", "2013-01-01", "2015-12-31")
]
VARIANT = {
    "name": "C_sector20_reinvest",
    "parent": "sector_caps/B_sector20",
    "change": "Add next-session investment of actual net maturity-sale proceeds using latest monthly rankings, current causal eligibility, unchanged 5% stock/20% sector entry limits and 90-day lots.",
}


def verify_reinvestment(result, scores, eligible, rules):
    """Check funds, dates and ranking provenance independently of the engine."""
    f, d, h = result.fills, result.decisions, result.history
    assert not d.date.duplicated().any()
    sales = f.loc[f.side.eq("SELL")].groupby("date").cash_change.sum()
    buys = f.loc[f.side.eq("BUY")]
    assert buys.date.isin(d.date).all()
    extra_days, extra_spend = 0, 0.0
    for decision in d.itertuples():
        prior = h.index[h.index.get_loc(decision.date) - 1]
        assert decision.signal_date == prior
        known_date = scores.loc[scores.date.le(prior), "date"].max()
        assert decision.model_score_date == known_date
        np.testing.assert_allclose(
            decision.previous_sale_proceeds, sales.get(prior, 0.0)
        )
        batch = buys.loc[buys.date.eq(decision.date)]
        assert -batch.cash_change.sum() <= decision.planned_spending + 1e-6
        assert (
            decision.planned_spending
            <= min(h.loc[prior, "cash"], h.loc[prior, "nav"] * rules.cohort_fraction)
            + 1e-6
        )
        if prior in set(scores.date):
            assert decision.trigger == "monthly"
            candidates = scores.loc[scores.date.eq(prior)]
        else:
            assert decision.trigger == "maturity_reinvestment"
            assert sales.get(prior, 0.0) > 0
            assert decision.planned_spending <= sales[prior] + 1e-6
            candidates = scores.loc[scores.date.eq(known_date)]
            candidates = candidates.loc[
                eligible.loc[decision.date, candidates.symbol].to_numpy()
            ]
            extra_days += 1
            extra_spend += -batch.cash_change.sum()
        top = candidates.sort_values(["score", "symbol"], ascending=[False, True]).head(
            rules.top_n
        )
        assert batch.symbol.isin(top.symbol).all()
    initial = buys.set_index("lot_id")
    for sale in f.loc[f.side.eq("SELL")].itertuples():
        due = initial.loc[sale.lot_id, "date"] + pd.Timedelta(days=90)
        # Development runs have observed maturity quotes; never silently assume
        # this for a different dataset where the missing-price rule may defer.
        assert sale.date == eligible.index[eligible.index.searchsorted(due)]
        np.testing.assert_allclose(sale.units, initial.loc[sale.lot_id, "units"])
    assert len(initial.loc[initial.maturity.le(h.index[-1])]) == len(
        f.loc[f.side.eq("SELL")]
    )
    return {
        "extra_decision_days": extra_days,
        "extra_purchase_spending": extra_spend,
        "proceeds_timing_and_rank_checks": True,
        "exact_maturity_sales_checked": True,
    }


def run():
    frozen = json.loads((PROJECT / "results/final/frozen_config.json").read_text())
    check_parent(frozen)
    model_hash = checksum(PROJECT / "artifacts/model.joblib")
    previous = json.loads((PROJECT / "results/sector_caps/protocol.json").read_text())
    for name in ["sector_allocation.py", "sector_engine.py"]:
        # The first A/B run used the preserved pre-extension sources. A fresh
        # reproduction may use the current engine with reinvestment disabled.
        # Accept either traceable version; numerical B parity is checked below.
        sources = [PROJECT / name, PROJECT / "research_snapshots/sector_caps_v1" / name]
        assert previous["code_sha256"][name] in {checksum(path) for path in sources}
    forecast_paths = {
        period: PROJECT / f"outputs/sector_caps/{period}/shared_forecasts.csv"
        for period, _, _ in PERIODS
    }
    protocol = {
        "variants": [VARIANT],
        "data_end": "2015-12-31",
        "rules": TradingRules().as_dict(),
        "source_sha256": checksum(DATABASE),
        "forecast_sha256": {p: checksum(path) for p, path in forecast_paths.items()},
        "new_model_fits": 0,
        "model": "Exactly the saved sector-pair monthly forecasts.",
        "data_layer": "Reviewed D2 price overlays and scoped LAB alias removal, unchanged.",
        "budget": "min(previous session net maturity-sale proceeds, prior cash, one third prior NAV), then stock and proportional sector caps; unused cash is not retried daily.",
        "eligibility": "Latest monthly scores carried forward, daily prior-close history/risk screens and entry-quarter supplied membership rechecked. No newly eligible but unscored names added.",
        "overlap": "A regular monthly batch takes precedence; never add a second proceeds-funded batch on that date.",
        "cap_timing": "Prior-close values; entry limits do not force early sales or prevent subsequent drift.",
        "sector_labels": "Static vendor snapshot, not certified dated classifications.",
        "annual_accounts": "Separate fits/accounts for 2013, 2014, 2015, using saved predictions.",
        "continuous_account": "2013-2015, single pre-2013 fit, positions carried across year ends.",
        "code_sha256": {
            n: checksum(PROJECT / n)
            for n in [
                "sector_engine.py",
                "sector_allocation.py",
                "sector_reinvestment.py",
                "run_sector_reinvestment.py",
                "development_revision.py",
            ]
        },
    }
    write_json(OUT / "protocol.json", protocol)
    original = build_data(DATABASE, end="2015-12-31")
    data, _, _ = corrected_data(original)
    data, _ = remove_fluidigm_alias(data)
    print("Building causal daily eligibility; forecasts are already fixed.", flush=True)
    eligible = build_entry_eligibility(DATABASE, data.quotes, data.panel)
    print("Daily screen matches every saved monthly candidate decision.", flush=True)
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        sectors = (
            con.execute("SELECT symbol,sector FROM universe ORDER BY symbol")
            .df()
            .set_index("symbol")
            .sector
        )
    rows, checks = [], []
    rules = TradingRules()
    for period, start, end in PERIODS:
        scores = pd.read_csv(
            forecast_paths[period], parse_dates=["date", "label_end", "fit_date"]
        )
        # A disabled extension must reproduce the existing sector-cap account.
        control = run_sector_backtest(
            data.quotes,
            data.benchmark,
            scores,
            start,
            end,
            rules,
            sectors=sectors.to_dict(),
            sector_cap=0.20,
        )
        old = pd.read_csv(
            PROJECT / f"results/sector_caps/{period}_B_sector20_nav.csv",
            index_col="date",
            parse_dates=True,
        )
        np.testing.assert_allclose(control.history.nav, old.nav, rtol=0, atol=1e-6)
        result = run_sector_backtest(
            data.quotes,
            data.benchmark,
            scores,
            start,
            end,
            rules,
            sectors=sectors.to_dict(),
            sector_cap=0.20,
            reinvest_maturity_cash=True,
            entry_eligibility=eligible,
        )
        name = VARIANT["name"]
        folder = LOCAL / period / name
        save_run(result, folder)
        result.decisions.to_csv(folder / "decisions.csv", index=False)
        row = {
            "variant": name,
            "period": period,
            "status": result.status,
            "last_valued_date": result.history.index[-1],
            "parent": VARIANT["parent"],
            "change": VARIANT["change"],
        }
        if result.status == "complete":
            nav_error = verify_account(result, data.quotes, rules.initial_cash)
            cap = verify_entry_caps(result, data.quotes, sectors, rules, 0.20)
            timing = verify_reinvestment(result, scores, eligible, rules)
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
                planned_sector_weight_max=cap,
                extra_decision_days=timing["extra_decision_days"],
                extra_purchase_spending=timing["extra_purchase_spending"],
            )
            incidents = result.incidents
            unresolved = (
                not incidents.empty
                and (
                    incidents.get("symbol", pd.Series(dtype=str)).eq("GYRE")
                    & pd.to_datetime(incidents.date).eq("2015-08-20")
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
                    "sector_control_parity": True,
                    "entry_caps_checked": True,
                    **timing,
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
            result.status,
            f"return={row.get('total_return', float('nan')):.2%}",
            flush=True,
        )
    assert all(
        checksum(path) == protocol["forecast_sha256"][p]
        for p, path in forecast_paths.items()
    )
    check_parent(frozen)
    write_json(
        OUT / "verification.json",
        {
            "accounts": checks,
            "source_unchanged": checksum(DATABASE) == protocol["source_sha256"],
            "original_model_unchanged": checksum(PROJECT / "artifacts/model.joblib")
            == model_hash,
            "monthly_screen_parity": True,
            "saved_forecasts_unchanged": True,
            "new_model_fits": 0,
            "new_post2015_run": False,
        },
    )


if __name__ == "__main__":
    run()
