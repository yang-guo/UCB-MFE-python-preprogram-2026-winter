"""Declared 2015-only allocation grid; winners are in-sample, not validation.

Both schedules reuse the exact saved pre-2015 fit and monthly forecasts. First
cross stock and common-sector caps. Then change ONE sector at a time around
each schedule's best common-cap setting. Finally test the combined individually
preferred caps at every stock cap. These are finite-grid maxima, not a proof of
a global optimum. No outcome from 2016 onward is loaded or evaluated.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import time

import duckdb
import numpy as np
import pandas as pd

from cap_sensitivity_engine import run_sector_backtest
from development_revision import corrected_data, remove_fluidigm_alias
from hw4.data import build_data, checksum
from hw4.engine import comparison
from hw4.settings import DATABASE, PROJECT, TradingRules
from run_development_revision import verify_account
from run_experiments import save_run, write_json
from run_sector_reinvestment import verify_reinvestment
from run_universe_comparison import check_parent
from sector_reinvestment import build_entry_eligibility

OUT = PROJECT / "results/cap_sensitivity_2015"
LOCAL = PROJECT / "outputs/cap_sensitivity_2015"
STOCK_CAPS = [n / 200 for n in range(2, 11)]
COMMON_CAPS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.0]
SECTOR_CAPS = [0.0] + COMMON_CAPS
SCHEDULES = ["monthly", "reinvestment"]
CONTEXT = None


def init_worker(path):
    global CONTEXT
    # Only this script's locally generated, source-keyed cache is loaded.
    CONTEXT = pd.read_pickle(path)


def independently_check(result, quotes, sectors, rules, caps):
    """Rebuild accounts, enforce causal entry limits and independently calculate risk."""
    f, h = result.fills, result.history
    error = verify_account(result, quotes, rules.initial_cash) if not f.empty else 0.0
    if f.empty:
        assert (
            h.nav.eq(rules.initial_cash).all() and h.cash.eq(rules.initial_cash).all()
        )
    else:
        signed = f.assign(signed=np.where(f.side.eq("BUY"), f.units, -f.units))
        units = signed.groupby(["date", "symbol"]).signed.sum().unstack(fill_value=0)
        units = units.reindex(h.index, fill_value=0).cumsum()
        value = (units * quotes[units.columns].ffill(limit=5).reindex(h.index)).fillna(
            0.0
        )
        for day, buys in f.loc[f.side.eq("BUY")].groupby("date"):
            prior = h.index[h.index.get_loc(day) - 1]
            nav, cash = h.loc[prior, ["nav", "cash"]]
            assert (
                -buys.cash_change.sum() <= min(cash, nav * rules.cohort_fraction) + 1e-6
            )
            held, incoming = value.loc[prior], buys.groupby("symbol").notional.sum()
            for symbol, amount in incoming.items():
                assert (
                    held.get(symbol, 0.0) + amount
                    <= max(held.get(symbol, 0.0), nav * rules.name_cap) + 1e-6
                )
            old_sector, new_sector = (
                held.groupby(sectors).sum(),
                incoming.groupby(sectors).sum(),
            )
            for sector, amount in new_sector.items():
                assert (
                    old_sector.get(sector, 0.0) + amount
                    <= max(old_sector.get(sector, 0.0), caps[sector] * nav) + 1e-6
                )
        buys = f.loc[f.side.eq("BUY")].set_index("lot_id")
        assert buys.signal_date.lt(buys.date).all()
        for sale in f.loc[f.side.eq("SELL")].itertuples():
            due = buys.loc[sale.lot_id, "date"] + pd.Timedelta(days=90)
            assert sale.date == quotes.index[quotes.index.searchsorted(due)]
            np.testing.assert_allclose(sale.units, buys.loc[sale.lot_id, "units"])
        assert len(buys.loc[buys.maturity.le(h.index[-1])]) == f.side.eq("SELL").sum()
    wealth = h.nav.to_numpy()
    daily = wealth[1:] / wealth[:-1] - 1
    sd = np.std(daily, ddof=1)
    metrics = comparison(result, rules).loc["strategy"]
    independent = {
        "final_value": wealth[-1],
        "total_return": wealth[-1] / wealth[0] - 1,
        "max_drawdown": np.min(wealth / np.maximum.accumulate(wealth) - 1),
        "annualized_volatility": sd * np.sqrt(252),
        "CAGR": (wealth[-1] / wealth[0]) ** (365.25 / (h.index[-1] - h.index[1]).days)
        - 1,
        "Sharpe": np.sqrt(252) * (np.mean(daily) - (1.02 ** (1 / 252) - 1)) / sd
        if sd > 0
        else np.nan,
    }
    for key, observed in independent.items():
        np.testing.assert_allclose(
            observed, metrics[key], rtol=1e-10, atol=1e-8, equal_nan=True
        )
    return error


def evaluate(spec):
    c = CONTEXT
    rules = replace(TradingRules(), name_cap=spec["stock_cap"])
    caps = {sector: spec["common_cap"] for sector in c["sectors"].unique()}
    caps.update(spec.get("overrides", {}))
    result = run_sector_backtest(
        c["quotes"],
        c["benchmark"],
        c["scores"],
        "2015-01-01",
        "2015-12-31",
        rules,
        sectors=c["sectors"].to_dict(),
        sector_cap=caps,
        reinvest_maturity_cash=spec["schedule"] == "reinvestment",
        entry_eligibility=c["eligible"],
    )
    folder = LOCAL / spec["variant"]
    save_run(result, folder)
    if hasattr(result, "decisions"):
        result.decisions.to_csv(folder / "decisions.csv", index=False)
    row = {
        **spec,
        "overrides": json.dumps(spec.get("overrides", {}), sort_keys=True),
        "period": "2015",
        "status": result.status,
        "last_valued_date": str(result.history.index[-1]),
        "specific_open_data_issue": "",
    }
    if result.status == "complete":
        row["max_nav_error"] = independently_check(
            result, c["quotes"], c["sectors"], rules, caps
        )
        if spec["schedule"] == "reinvestment":
            row.update(verify_reinvestment(result, c["scores"], c["eligible"], rules))
        metrics = comparison(result, rules)
        row.update(metrics.loc["strategy"].to_dict())
        row.update(
            benchmark_return=metrics.loc["benchmark", "total_return"],
            average_cash_fraction=(result.history.cash / result.history.nav)
            .iloc[1:]
            .mean(),
            fees=result.fills.fee.sum(),
            buys=int(result.fills.side.eq("BUY").sum()),
            incident_rows=len(result.incidents),
            checks_passed=True,
        )
        row["excess_return_pp"] = 100 * (row["total_return"] - row["benchmark_return"])
        # Preserve and flag any held event; do not rank questionable observations
        # as verified economic gains just because the cash ledger reconciles.
        if not result.incidents.empty:
            row["specific_open_data_issue"] = (
                "Incident ledger requires review before selection"
            )
    else:
        row["specific_open_data_issue"] = (
            "Incomplete account; no full-year return assigned"
        )
    return row


def eligible_rows(rows):
    df = pd.DataFrame(rows)
    return df.loc[df.status.eq("complete") & df.specific_open_data_issue.eq("")].copy()


def best(rows):
    """Return maximizer; exact ties prefer less restrictive caps, not a fake optimum."""
    df = eligible_rows(rows)
    assert len(df), "No completed, incident-free candidate to select."
    maximum = df.total_return.max()
    ties = df.loc[np.isclose(df.total_return, maximum, rtol=0, atol=1e-12)]
    return (
        ties.sort_values(
            ["common_cap", "stock_cap", "variant"], ascending=[False, False, True]
        )
        .iloc[0]
        .to_dict()
    )


def run():
    started = time.perf_counter()
    LOCAL.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((PROJECT / "results/final/frozen_config.json").read_text())
    check_parent(frozen)
    source_hash, model_hash = (
        checksum(DATABASE),
        checksum(PROJECT / "artifacts/model.joblib"),
    )
    forecasts = PROJECT / "outputs/sector_caps/2015/shared_forecasts.csv"
    immutable = {
        str(p.relative_to(PROJECT)): checksum(p)
        for p in [
            forecasts,
            PROJECT / "sector_engine.py",
            PROJECT / "sector_allocation.py",
            PROJECT / "artifacts/model.joblib",
        ]
    }
    code_files = [
        "cap_sensitivity_engine.py",
        "cap_sensitivity_allocation.py",
        "run_cap_sensitivity.py",
        "sector_reinvestment.py",
        "development_revision.py",
    ]
    protocol = {
        "scope": "2015-only hindsight sensitivity; not out-of-sample selection evidence",
        "stock_caps": STOCK_CAPS,
        "common_caps": COMMON_CAPS,
        "individual_sector_caps": SECTOR_CAPS,
        "schedules": SCHEDULES,
        "selection_objective": "Maximum 2015 net total return among tested settings",
        "sequence": [
            "Cross all stock/common-sector limits",
            "Vary each sector alone around each schedule best common-cap setting",
            "Combine each sector preferred cap and retest every stock cap",
        ],
        "tie_policy": "Return tolerance 1e-12; common grid ties prefer larger caps. Sector ties retain reference cap when possible, otherwise larger cap.",
        "selection_exclusions": "Incomplete accounts or unreviewed incident logs remain reported but cannot win.",
        "source_sha256": source_hash,
        "immutable_sha256": immutable,
        "code_sha256": {n: checksum(PROJECT / n) for n in code_files},
        "new_fits": 0,
        "later_years_evaluated": False,
        "rules": TradingRules().as_dict(),
        "sector_labels": "Static vendor snapshot; not certified dated classifications.",
        "blocked_budget": "Cash; no replacement picks, no early liquidation.",
        "variants": [],
    }
    write_json(OUT / "protocol.json", protocol)
    identity = hashlib.sha256(
        json.dumps(
            {
                "source": source_hash,
                "forecast": checksum(forecasts),
                "data": checksum(PROJECT / "hw4/data.py"),
                "eligibility": checksum(PROJECT / "sector_reinvestment.py"),
                "revision": checksum(PROJECT / "development_revision.py"),
                "price_registry": checksum(
                    PROJECT / "data_review/development_price_repairs.json"
                ),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    context_path = LOCAL / f"context_{identity}.pkl"
    if not context_path.exists():
        original = build_data(DATABASE, end="2015-12-31")
        data, _, _ = corrected_data(original)
        data, _ = remove_fluidigm_alias(data)
        print(
            "Preparing unchanged forecasts and causal eligibility through 2015.",
            flush=True,
        )
        eligible = build_entry_eligibility(DATABASE, data.quotes, data.panel)
        print("Daily eligibility matches the original monthly decisions.", flush=True)
        scores = pd.read_csv(forecasts, parse_dates=["date", "fit_date", "label_end"])
        with duckdb.connect(str(DATABASE), read_only=True) as con:
            sectors = (
                con.execute("SELECT symbol,sector FROM universe ORDER BY symbol")
                .df()
                .set_index("symbol")
                .sector
            )
        # Only names that have a saved forecast can ever be bought. Dropping
        # unreachable columns speeds marking without changing an entry decision.
        symbols = sorted(scores.symbol.unique())
        context = {
            "quotes": data.quotes.loc["2014-12-31":, symbols],
            "benchmark": data.benchmark.loc["2014-12-31":],
            "scores": scores,
            "eligible": eligible.loc["2014-12-31":, symbols],
            "sectors": sectors,
            "monthly_screen_parity": True,
        }
        pd.to_pickle(context, context_path)
    init_worker(context_path)
    assert CONTEXT["quotes"].index.max() == pd.Timestamp("2015-12-31")
    sectors = sorted(CONTEXT["sectors"].unique())
    protocol["sectors"] = sectors
    rows = []

    def execute_phase(pool, specs, phase):
        protocol["variants"].extend(specs)
        write_json(
            OUT / "protocol.json", protocol
        )  # Declare each stage before its results.
        tasks = {pool.submit(evaluate, spec): spec for spec in specs}
        for i, future in enumerate(as_completed(tasks), 1):
            rows.append(future.result())
            if i % 20 == 0 or i == len(specs):
                pd.DataFrame(rows).sort_values("variant").to_csv(
                    OUT / "experiments.csv", index=False
                )
                print(f"{phase}: {i}/{len(specs)} accounts checked", flush=True)

    with ProcessPoolExecutor(
        max_workers=2, initializer=init_worker, initargs=(context_path,)
    ) as pool:
        grid = [
            {
                "variant": f"G_{schedule[0]}_n{int(round(n * 10000)):03d}_s{int(s * 100):03d}",
                "phase": "common_grid",
                "schedule": schedule,
                "stock_cap": n,
                "common_cap": s,
                "overrides": {},
                "parent": "sector_caps/B_sector20"
                if schedule == "monthly"
                else "sector_reinvestment/C_sector20_reinvest",
                "change": f"2015 sensitivity: stock cap {n:.1%}, common sector cap {s:.0%}; {schedule} schedule.",
            }
            for schedule in SCHEDULES
            for n in STOCK_CAPS
            for s in COMMON_CAPS
        ]
        execute_phase(pool, grid, "Common-cap grid")
        # Same settings must reproduce the saved baselines, despite narrower columns.
        for schedule, family, variant in [
            ("monthly", "sector_caps", "B_sector20"),
            ("reinvestment", "sector_reinvestment", "C_sector20_reinvest"),
        ]:
            name = f"G_{schedule[0]}_n500_s020"
            current = pd.read_csv(
                LOCAL / name / "daily_nav.csv", index_col="date", parse_dates=True
            )
            old = pd.read_csv(
                PROJECT / f"results/{family}/2015_{variant}_nav.csv",
                index_col="date",
                parse_dates=True,
            )
            np.testing.assert_allclose(current.nav, old.nav, rtol=0, atol=1e-6)
        parents = {
            schedule: best([r for r in rows if r["schedule"] == schedule])
            for schedule in SCHEDULES
        }
        specs = []
        for schedule, parent in parents.items():
            for ix, sector in enumerate(sectors):
                for cap in SECTOR_CAPS:
                    specs.append(
                        {
                            "variant": f"P_{schedule[0]}_{ix:02d}_s{int(cap * 100):03d}",
                            "phase": "individual_sector",
                            "schedule": schedule,
                            "stock_cap": parent["stock_cap"],
                            "common_cap": parent["common_cap"],
                            "overrides": {sector: cap},
                            "sector": sector,
                            "tested_sector_cap": cap,
                            "parent": f"cap_sensitivity_2015/{parent['variant']}",
                            "change": f"Change only {sector} cap to {cap:.0%}; other sectors fixed at {parent['common_cap']:.0%}.",
                        }
                    )
        execute_phase(pool, specs, "Individual sectors")
        sector_choices, combinations = [], {}
        for schedule, parent in parents.items():
            combinations[schedule] = {}
            for sector in sectors:
                frame = eligible_rows(
                    [
                        r
                        for r in rows
                        if r["phase"] == "individual_sector"
                        and r["schedule"] == schedule
                        and r["sector"] == sector
                    ]
                )
                maximum = frame.total_return.max()
                ties = frame.loc[
                    np.isclose(frame.total_return, maximum, rtol=0, atol=1e-12)
                ]
                matching = ties.loc[ties.tested_sector_cap.eq(parent["common_cap"])]
                chosen = (
                    matching
                    if len(matching)
                    else ties.sort_values("tested_sector_cap", ascending=False)
                ).iloc[0]
                cap = float(chosen.tested_sector_cap)
                combinations[schedule][sector] = cap
                sector_choices.append(
                    {
                        "schedule": schedule,
                        "sector": sector,
                        "stock_cap": parent["stock_cap"],
                        "other_sector_cap": parent["common_cap"],
                        "best_cap": cap,
                        "tied_best_caps": ",".join(
                            f"{x:.0%}" for x in sorted(ties.tested_sector_cap)
                        ),
                        "best_return": maximum,
                        "change_vs_common_pp": 100 * (maximum - parent["total_return"]),
                        "best_max_drawdown": chosen.max_drawdown,
                        "best_variant": chosen.variant,
                    }
                )
        pd.DataFrame(sector_choices).to_csv(OUT / "sector_best.csv", index=False)
        write_json(OUT / "selected_common_caps.json", parents)
        specs = [
            {
                "variant": f"C_{schedule[0]}_n{int(round(n * 10000)):03d}",
                "phase": "combined_sectors",
                "schedule": schedule,
                "stock_cap": n,
                "common_cap": parents[schedule]["common_cap"],
                "overrides": combinations[schedule],
                "parent": f"cap_sensitivity_2015/{parents[schedule]['variant']}",
                "change": f"Combine individually preferred sector caps; test stock cap {n:.1%}. Interactions require an actual account run.",
            }
            for schedule in SCHEDULES
            for n in STOCK_CAPS
        ]
        execute_phase(pool, specs, "Combined sector choices")
    bests = {s: best([r for r in rows if r["schedule"] == s]) for s in SCHEDULES}
    write_json(OUT / "best_tested.json", bests)
    assert all(checksum(PROJECT / name) == digest for name, digest in immutable.items())
    check_parent(frozen)
    proof = {
        "accounts": len(rows),
        "completed": sum(r["status"] == "complete" for r in rows),
        "unreviewed_accounts": sum(bool(r["specific_open_data_issue"]) for r in rows),
        "max_nav_error": max(r.get("max_nav_error", 0.0) for r in rows),
        "monthly_and_reinvestment_baseline_parity": True,
        "monthly_screen_parity": True,
        "source_unchanged": checksum(DATABASE) == source_hash,
        "original_model_unchanged": checksum(PROJECT / "artifacts/model.joblib")
        == model_hash,
        "new_fits": 0,
        "post2015_runs": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(OUT / "verification.json", proof)
    print(json.dumps(proof, indent=2), flush=True)
    for schedule, winner in bests.items():
        print(
            schedule,
            winner["variant"],
            f"best tested return {winner['total_return']:.2%}",
            flush=True,
        )


if __name__ == "__main__":
    run()
