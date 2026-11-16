"""Run predeclared development comparisons with a hard 2015 price cutoff.

First isolate price repairs, then alias removal. Compare each entry/risk change
against that common revised baseline using identical fitted forecasts. Annual
accounts refit at each year's start; the separate continuous account fits once
before 2013. No saved original experiment or fitted submission model is replaced.
"""

from dataclasses import replace
import json
import logging

import numpy as np
import pandas as pd

from development_revision import corrected_data, remove_fluidigm_alias, select_entries
from hw4.data import build_data, checksum
from hw4.engine import comparison, run_backtest
from hw4.modeling import MOMENTUM_SPECS, predict_period, training_rows
from hw4.settings import DATABASE, PROJECT, TradingRules
from run_experiments import save_run, write_json
from run_universe_comparison import check_parent

OUT = PROJECT / "results/development_revision"
LOCAL = PROJECT / "outputs/development_revision"
VARIANTS = [
    {
        "name": "D1_price_repairs",
        "data": "prices",
        "entry_rule": "unchanged",
        "cap": 0.05,
        "parent": "original_long_momentum",
        "change": "Four evidence-backed price-unit corrections; recompute affected features, targets and risk screen.",
    },
    {
        "name": "D2_alias_removed",
        "data": "deduplicated",
        "entry_rule": "unchanged",
        "cap": 0.05,
        "parent": "D1_price_repairs",
        "change": "Remove documented LAB alias from fitting and new entries; retain historical FLDM source.",
    },
    {
        "name": "S1_positive_trend",
        "data": "deduplicated",
        "entry_rule": "positive_63d_trend",
        "cap": 0.05,
        "parent": "D2_alias_removed",
        "change": "Require strictly positive trailing 63-session return for new entries; same forecasts and model fit.",
    },
    {
        "name": "S2_volatility_40pct",
        "data": "deduplicated",
        "entry_rule": "volatility_40pct",
        "cap": 0.05,
        "parent": "D2_alias_removed",
        "change": "Require trailing 63-return annualized volatility <=40% for new entries; training still uses original 80% screen.",
    },
    {
        "name": "S3_name_cap_2_5pct",
        "data": "deduplicated",
        "entry_rule": "unchanged",
        "cap": 0.025,
        "parent": "D2_alias_removed",
        "change": "Halve aggregate recorded-symbol entry cap from 5% to 2.5%; keep scores, cohort budget and maturity unchanged.",
    },
]


def verify_account(result, quotes, capital):
    """Independently reconstruct daily cash and units from fills, including fees."""
    f = result.fills.copy()
    f["date"] = pd.to_datetime(f.date)
    f["signed_units"] = np.where(f.side.eq("BUY"), f.units, -f.units)
    index = result.history.index
    flows = f.groupby("date").cash_change.sum().reindex(index, fill_value=0)
    cash = capital + flows.cumsum()
    units = f.groupby(["date", "symbol"]).signed_units.sum().unstack(fill_value=0)
    units = units.reindex(index, fill_value=0).cumsum()
    marks = quotes[units.columns].ffill(limit=5).reindex(index)
    assert not ((units.abs() > 1e-7) & marks.isna()).any().any()
    nav = cash + (units * marks).fillna(0).sum(axis=1)
    np.testing.assert_allclose(nav, result.history.nav, atol=1e-6, rtol=0)
    np.testing.assert_allclose(cash, result.history.cash, atol=1e-6, rtol=0)
    assert result.history.cash.min() >= -1e-6
    sells = f.loc[f.side.eq("SELL")]
    buys = f.loc[f.side.eq("BUY")].set_index("lot_id")
    for row in sells.itertuples():
        assert row.date >= pd.Timestamp(buys.loc[row.lot_id, "date"]) + pd.Timedelta(
            days=90
        )
    return float(np.abs(nav - result.history.nav).max())


def run():
    parent = json.loads((PROJECT / "results/final/frozen_config.json").read_text())
    check_parent(parent)
    original_model_hash = checksum(PROJECT / "artifacts/model.joblib")
    original = build_data(DATABASE, end="2015-12-31")
    assert original.quotes.index.max() <= pd.Timestamp("2015-12-31")
    prices, changes, anchors = corrected_data(original)
    deduplicated, aliases_removed = remove_fluidigm_alias(prices)
    data_variants = {"prices": prices, "deduplicated": deduplicated}
    configuration = {
        "data_end": "2015-12-31",
        "source_sha256": checksum(DATABASE),
        "price_repairs_sha256": checksum(
            PROJECT / "data_review/development_price_repairs.json"
        ),
        "variants": VARIANTS,
        "validation_years": [2013, 2014, 2015],
        "continuous_account": "2013-2015, single fit before 2013; no annual reset",
        "model": "Unchanged long_momentum specification; refit on each declared revised development panel.",
        "training_unchanged_for_S_variants": True,
        "choice_basis": "Review worst-year return, drawdown and cross-year/continuous consistency alongside return; no outcome is promised.",
        "financial_features": "Not used. The partial financial repair sample is not extrapolated to the full panel.",
        "source_limitations": "These targeted price/identity corrections do not certify every vendor record or ranking input.",
    }
    write_json(OUT / "protocol.json", configuration)
    pd.DataFrame(changes).to_csv(OUT / "price_boundary_checks.csv", index=False)
    pd.DataFrame(anchors).to_csv(OUT / "lci_anchor_checks.csv", index=False)
    all_audits, rows, verification, training_audit = [], [], [], []
    periods = [(str(y), f"{y}-01-01", f"{y}-12-31") for y in [2013, 2014, 2015]]
    periods.append(("continuous_2013_2015", "2013-01-01", "2015-12-31"))
    for period, start, end in periods:
        forecasts = {}
        for label, data in data_variants.items():
            scores, audit, models = predict_period(
                data.panel, data.quotes.index, MOMENTUM_SPECS[0], start, end
            )
            forecasts[label] = scores
            all_audits.append(audit.assign(period=period, data_variant=label))
            # Log fitted direction; a feature called momentum can learn contrarian ranking.
            fit_date, fitted = models[0]
            regression = fitted["regression"].estimator_
            eligible_panel = data.panel.copy()
            eligible_panel["eligible"] &= eligible_panel.risk_screen_pass
            train = training_rows(eligible_panel, fit_date)
            training_audit.append(
                {
                    "period": period,
                    "data_variant": label,
                    "training_rows": len(train),
                    "unclipped_min_target": train.target.min(),
                    "unclipped_max_target": train.target.max(),
                    "targets_above_500pct": int(train.target.gt(5).sum()),
                    "standardized_feature_coefficient": float(regression.coef_[0]),
                }
            )
        for variant in VARIANTS:
            data = data_variants[variant["data"]]
            scores = select_entries(
                forecasts[variant["data"]], data.panel, variant["entry_rule"]
            )
            rules = replace(TradingRules(), name_cap=variant["cap"])
            result = run_backtest(
                data.quotes, data.benchmark, scores, start, end, rules
            )
            name = variant["name"]
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
                error = verify_account(result, data.quotes, rules.initial_cash)
                metrics = comparison(result, rules)
                row.update(metrics.loc["strategy"].to_dict())
                row.update(
                    benchmark_return=metrics.loc["benchmark", "total_return"],
                    average_cash_fraction=(result.history.cash / result.history.nav)
                    .iloc[1:]
                    .mean(),
                    fees=result.fills.fee.sum(),
                    buys=int(result.fills.side.eq("BUY").sum()),
                )
                result.history.to_csv(OUT / f"{period}_{name}_nav.csv")
                verification.append(
                    {"period": period, "variant": name, "max_nav_error": error}
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
    pd.concat(all_audits).to_csv(OUT / "fit_audit.csv", index=False)
    pd.DataFrame(training_audit).to_csv(OUT / "training_label_checks.csv", index=False)
    write_json(
        OUT / "verification.json",
        {
            "accounts": verification,
            "aliases_removed_from_screened_panel": aliases_removed,
            "source_unchanged": checksum(DATABASE) == configuration["source_sha256"],
            "original_model_unchanged": checksum(PROJECT / "artifacts/model.joblib")
            == original_model_hash,
            "new_post2015_run": False,
        },
    )
    check_parent(parent)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run()
