"""Evaluate the selected pre-2013 model without refitting or changing its rules.

Run from this folder with .venv/Scripts/python.exe run_fixed_anchor_final.py.
The saved model, forecasts, data sources and accounting evidence are hashed.
Each distinct accounting protocol receives its own output directory. Only a
complete independently reconciled account is marked verified.
"""

import hashlib
import json

import duckdb
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fixed_anchor_inputs import (
    corrected_feature_panel,
    predict_selected,
    redirected_attribute,
    add_accounting_events,
)
from hw4.data import checksum
from hw4.engine import performance
from hw4.settings import DATABASE, PROJECT, TradingRules
from rank_accounting_prices import REPAIRS
from rank_event_accounting import run_event_account
from rank_final_inputs import load_final_context
from research_session import save_json, verify_saved_sources
import run_rank_final
import verify_rank_final


OUT = PROJECT / "results/fixed_anchor_final"
LOCAL = PROJECT / "outputs/fixed_anchor_final"
SELECTION = PROJECT / "results/fixed_anchor_selection"
ARTIFACT = PROJECT / "artifacts/fixed_anchor_2012.joblib"


def selected_policy():
    """Reject changed selection or artifacts before computing any forecasts."""
    frozen = json.loads((SELECTION / "frozen_config.json").read_text())
    receipt = json.loads((SELECTION / "verification.json").read_text())
    digest = hashlib.sha256(
        json.dumps(frozen["configuration"], sort_keys=True).encode()
    ).hexdigest()
    assert digest == frozen["sha256"] == receipt["policy_sha256"]
    assert checksum(ARTIFACT) == receipt["model_sha256"]
    reference = PROJECT / "artifacts/fixed_anchor_2012_reference.csv"
    assert checksum(reference) == receipt["reference_sha256"]
    model = joblib.load(ARTIFACT)  # Trusted local artifact verified above.
    sample = pd.read_csv(reference)
    np.testing.assert_allclose(
        model.predict(sample[frozen["configuration"]["features"]]),
        sample["expected_prediction"],
        atol=1e-12,
        rtol=0,
    )
    return frozen, receipt, model


def prepare_scores(context, frozen, receipt, model):
    """Reuse identical saved forecasts, or create them once using predict only."""
    files = [
        "fixed_anchor_inputs.py",
        "rank_final_inputs.py",
        "rank_accounting_prices.py",
        "research_models.py",
        "hw4/data.py",
        "development_revision.py",
    ]
    files += [
        f"research_events/price_unit_evidence/{s['symbol']}.json" for s in REPAIRS
    ]
    files += [
        "research_events/fixed_anchor_price_repairs.json",
        "research_events/fixed_anchor_CVNA_evidence.json",
    ]
    identity = dict(
        policy_sha256=frozen["sha256"],
        model_sha256=receipt["model_sha256"],
        database_sha256=checksum(DATABASE),
        files={name: checksum(PROJECT / name) for name in files},
    )
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = LOCAL / "forecast_versions" / f"{version}.csv"
    audit_path = OUT / "forecast_versions" / f"{version}.json"
    if path.exists() or audit_path.exists():
        assert path.exists() and audit_path.exists(), (
            "Incomplete forecast artifact pair."
        )
        audit = json.loads(audit_path.read_text())
        assert identity == audit["identity"], (
            "Inputs changed; preserve forecasts and review separately."
        )
        assert checksum(path) == audit["predictions_sha256"]
        print("Reusing verified saved forecasts; no retraining.", flush=True)
    else:
        print(
            "Preparing corrected inputs and predicting with the saved model.",
            flush=True,
        )
        panel, repairs = corrected_feature_panel(
            context, frozen["configuration"]["features"]
        )
        scores = predict_selected(model, panel, frozen["configuration"])
        path.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(path, index=False)
        audit = dict(
            identity=identity,
            predictions_sha256=checksum(path),
            predictions_path=str(path.relative_to(PROJECT)),
            candidate_predictions=len(scores),
            unknown_future_targets_retained=int(scores.target.isna().sum()),
            new_fits=0,
            model_or_trading_rule_changes=0,
            first_signal=str(scores.date.min().date()),
            last_signal=str(scores.date.max().date()),
            fit_date=frozen["configuration"]["fitted_asof"],
            pre_seam_features_unchanged=True,
            repairs=repairs,
        )
        save_json(audit_path, audit)
    scores = pd.read_csv(
        path,
        parse_dates=["date", "entry_date", "scheduled_exit", "label_end", "fit_date"],
    )
    save_json(OUT / "current_prediction_audit.json", audit)
    return scores, audit


def run():
    frozen, selection, model = selected_policy()
    config = frozen["configuration"]
    context, protected = load_final_context()
    scores, prediction_audit = prepare_scores(context, frozen, selection, model)
    # Legacy helpers write evidence; restore their original locations even if
    # a check fails. The archived accounts and their files remain untouched.
    with redirected_attribute(run_rank_final, "OUT", OUT):
        quotes, raw, events, sectors, supplements, proxies = (
            run_rank_final.accounting_inputs(context)
        )
    quotes, raw, events, added_repairs = add_accounting_events(
        quotes, raw, events, sectors
    )
    save_json(OUT / "additional_price_unit_audit.json", added_repairs)
    rules = TradingRules(**config["trading_rules"])
    files = [
        "run_fixed_anchor_final.py",
        "fixed_anchor_inputs.py",
        "rank_event_accounting.py",
        "rank_final_inputs.py",
        "rank_accounting_prices.py",
        "run_rank_final.py",
        "verify_rank_final.py",
        "research_events/events.json",
        "research_events/rank_model_events.json",
        "research_events/supplemental_quotes.csv",
        "research_events/fixed_anchor_events.json",
        "research_events/fixed_anchor_price_repairs.json",
        "research_events/fixed_anchor_supplemental_quotes.csv",
        "research_events/fixed_anchor_CVSA_evidence.json",
        "research_events/fixed_anchor_CVNA_evidence.json",
    ]
    optional = PROJECT / "research_events/rank_supplemental_quotes.csv"
    if optional.exists():
        files.append(str(optional.relative_to(PROJECT)))
    files += [
        f"research_events/price_unit_evidence/{s['symbol']}.json" for s in REPAIRS
    ]
    protocol = dict(
        policy_sha256=frozen["sha256"],
        model_sha256=selection["model_sha256"],
        predictions_sha256=prediction_audit["predictions_sha256"],
        source_sha256=checksum(DATABASE),
        files={name: checksum(PROJECT / name) for name in files},
        sector_proxies=proxies,
        quote_supplements=supplements,
        additional_sector_proxy="CVSA inherits ATGE sector for inherited holdings only",
        additional_quote_scope="CVSA: February 23 through April 2, 2026; original maturity retained",
        settlement_sessions=2,
        new_fits=0,
        model_or_trading_rule_changes=0,
        evaluation_scope="Retrospective evaluation of the selected fixed strategy; not a pristine independent holdout.",
    )
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    save_json(OUT / "protocols" / f"{digest}.json", protocol)
    print("Running the required April 2016–September 2026 account.", flush=True)
    result = run_event_account(
        quotes,
        context["benchmark"],
        scores,
        raw,
        events,
        "2016-04-01",
        "2026-09-30",
        rules,
        2,
        sectors=sectors,
        sector_cap=config["sector_entry_cap"],
    )
    local = LOCAL / "runs" / digest
    local.mkdir(parents=True, exist_ok=True)
    for name, frame in result.items():
        if isinstance(frame, pd.DataFrame):
            frame.to_csv(local / f"{name}.csv", index=name == "history")
    status = dict(
        status=result["status"],
        last_valued_date=str(result["history"].index[-1].date()),
        protocol_sha256=digest,
        local_output=str(local.relative_to(PROJECT)),
        policy_sha256=frozen["sha256"],
        incident_rows=len(result["incidents"]),
        fills=len(result["fills"]),
        event_lots=len(result["events"]),
        verified=False,
    )
    save_json(OUT / "attempts" / f"{digest}.json", status)
    save_json(OUT / "latest_status.json", status)
    print(json.dumps(status, indent=2), flush=True)
    verify_saved_sources(protected)
    assert checksum(ARTIFACT) == selection["model_sha256"]
    if result["status"] != "complete":
        print(result["incidents"].tail(15).to_string(index=False), flush=True)
        raise RuntimeError("Account preserved; factual data investigation is required.")
    metrics = pd.DataFrame(
        {
            "strategy": performance(result["history"].nav, rules.annual_rf),
            "benchmark": performance(result["history"].benchmark_nav, rules.annual_rf),
        }
    ).T
    metrics.to_csv(OUT / "performance.csv")
    result["history"].to_csv(OUT / "daily_nav.csv")
    result["incidents"].to_csv(OUT / "incidents.csv", index=False)
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        membership = con.execute(
            "SELECT symbol,quarter_start,selection_date,effective_start FROM universe_membership"
        ).df()
    print(
        "Independently reconstructing the ledger and checking timing/caps.", flush=True
    )
    with redirected_attribute(verify_rank_final, "RESULTS", OUT):
        checks = verify_rank_final.verify(
            2,
            quotes,
            raw,
            {"events": events},
            membership,
            scores,
            sectors,
            context["benchmark"],
        )
    checks.update(
        protocol_sha256=digest,
        model_sha256=selection["model_sha256"],
        selected_model_unchanged=True,
        new_fits=0,
        predictions_sha256=prediction_audit["predictions_sha256"],
        no_parameter_selection_in_this_evaluation=True,
        unit_repairs_consistent_in_features_and_accounting=True,
    )
    assert (
        checksum(PROJECT / prediction_audit["predictions_path"])
        == prediction_audit["predictions_sha256"]
    )
    save_json(OUT / "verification.json", checks)
    status["verified"] = True
    save_json(OUT / "latest_status.json", status)
    save_json(OUT / "attempts" / f"{digest}.json", status)
    print(metrics.to_string(), flush=True)
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        run()
