"""Preserve the exact pre-2013 fixed model selected after report22.

This is preparation, not final-period evaluation. Reconstruct the already tested
fit using development data only, reproduce every saved forecast, and export a
separate artifact. Never call the older final-preparation script: that script
fits on the full development sample and would change the selected estimator.
"""

from datetime import datetime, timezone
import hashlib
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from hw4.data import checksum
from hw4.settings import PROJECT, TradingRules
from dataclasses import replace
from research_models import eligible_panel, predict_research
from research_session import load_context, save_json, verify_saved_sources
from run_rank_model_stability import FIXED


OUT = PROJECT / "results/fixed_anchor_selection"
ARTIFACT = PROJECT / "artifacts/fixed_anchor_2012.joblib"
REFERENCE = PROJECT / "artifacts/fixed_anchor_2012_reference.csv"


def run():
    sources = [
        PROJECT / "outputs/fitting_policy/F00_fixed/forecasts.csv",
        PROJECT / "results/fitting_policy/experiments.csv",
        PROJECT / "results/fitting_policy/verification.json",
        PROJECT / "artifacts/rank_monthly_model.joblib",
    ]
    original_hashes = {str(p): checksum(p) for p in sources}
    prior_results = pd.read_csv(sources[1])
    selected_metrics = prior_results.loc[
        prior_results.variant.eq("F00_fixed")
        & prior_results.period.eq("continuous_2013_2015")
    ].iloc[0]
    assert selected_metrics.status == "complete" and selected_metrics.incident_rows == 0
    context, protected = load_context()
    with threadpool_limits(limits=2):
        scores, audit, models = predict_research(
            context["panel"], context["quotes"].index, FIXED, 1000.0
        )
    assert len(models) == len(audit) == 1
    fit = audit.iloc[0]
    assert fit.fit_date == pd.Timestamp("2012-12-31")
    assert fit.training_rows == 51633
    assert fit.first_training_signal >= pd.Timestamp("2010-01-01")
    assert fit.last_training_label < fit.fit_date
    saved = pd.read_csv(sources[0], parse_dates=["date", "label_end", "fit_date"])
    pd.testing.assert_frame_equal(
        scores.drop(columns="score"), saved.drop(columns="score"), check_dtype=False
    )
    np.testing.assert_allclose(scores.score, saved.score, rtol=0, atol=1e-12)
    model = models[0][1]
    p = eligible_panel(context["panel"])
    candidates = p.loc[
        p.eligible
        & p.date.ge(fit.fit_date)
        & p.entry_date.between("2013-01-01", "2015-12-31")
    ].copy()
    pd.testing.assert_frame_equal(
        candidates[["date", "symbol"]].reset_index(drop=True),
        scores[["date", "symbol"]].reset_index(drop=True),
    )
    x = candidates[list(FIXED.features)]
    with threadpool_limits(limits=2):
        expected = model.predict(x)
    np.testing.assert_allclose(expected, saved.score, atol=1e-12, rtol=0)
    rules = replace(TradingRules(), cohort_fraction=1.0)
    configuration = dict(
        selection="Sterling: 'Okay, let's proceed with the fixed model.' Interpreted as report22 F00_fixed with ordinary filing timing.",
        selected_candidate="fitting_policy/F00_fixed",
        model="Ridge",
        alpha=1000.0,
        features=list(FIXED.features),
        fitted_asof="2012-12-31",
        training_start="2010-01-01",
        actual_first_training_signal=str(fit.first_training_signal.date()),
        actual_last_training_signal=str(fit.last_training_signal.date()),
        actual_last_training_label=str(fit.last_training_label.date()),
        training_rows=int(fit.training_rows),
        update_policy="Retain the exact pre-2013 coefficients and preprocessing. Refresh inputs and forecasts monthly; no full-development refit, annual replacement or averaging.",
        preprocessing="Training median with missing indicators, training standardization; no clipping of rank inputs; training targets clipped at training1st/99th percentiles only.",
        training_target_bounds=[float(fit.target_lower), float(fit.target_upper)],
        financial_policy="Valid quarterly filing+1 calendar day; no extra delay; fiscal age over550 days makes ratios missing. No future backfill.",
        trading_rules=rules.as_dict(),
        sector_entry_cap=0.2,
        entry_policy="Monthly top20, next-session close; use prior-close cash/exposures. Maturity proceeds await next monthly entry.",
        holding_policy="Each lot90 calendar days; first market session on/after maturity. Open lots valued at endpoint.",
        cap_scope="5% recorded-symbol and20% supplied-sector limits at entry; market drift allowed. Static sectors and incomplete issuer mapping remain limitations.",
        intended_evaluation_start="2016-04-01",
        intended_evaluation_end="2026-09-30",
        benchmark="Supplied ^SP500TR; $1m from first evaluation close to the same endpoint.",
        evaluated_in_this_preparation="2013–2015 saved-forecast reproduction only; no post-2015 prediction or backtest.",
        evaluation_accounting="Reuse previously documented price/event accounting without overwriting archives; any newly encountered data gap must be investigated, never given an invented exit.",
        historical_context="Earlier different models were evaluated over the final period; preserve that history. This selected model has not received a new final-period account in this preparation.",
        source_sha256=protected,
        archived_forecasts_sha256=original_hashes[str(sources[0])],
    )
    digest = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()
    config_path = OUT / "frozen_config.json"
    if config_path.exists():
        prior = json.loads(config_path.read_text())
        if prior["sha256"] != digest:
            raise RuntimeError(
                "Existing selected policy differs; preserve it and review a new version explicitly."
            )
    elif ARTIFACT.exists() or REFERENCE.exists():
        raise RuntimeError(
            "An artifact already exists without this policy; refusing to overwrite it."
        )
    else:
        save_json(
            config_path,
            dict(
                configuration=configuration,
                sha256=digest,
                selected_at_utc=datetime.now(timezone.utc).isoformat(),
            ),
        )
    if not ARTIFACT.exists():
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, ARTIFACT)
    loaded = joblib.load(ARTIFACT)  # Only our own newly verified local artifact.
    with threadpool_limits(limits=2):
        actual = loaded.predict(x)
    np.testing.assert_array_equal(actual, expected)
    # Keep varied dates and missing-input examples for a fresh-process reload check.
    sample = candidates.sample(n=200, random_state=731).sort_values(["date", "symbol"])
    missing = candidates.loc[candidates[list(FIXED.features)].isna().any(axis=1)].head(
        10
    )
    sample = (
        pd.concat([sample, missing])
        .drop_duplicates(["date", "symbol"])
        .sort_values(["date", "symbol"])
    )
    reference = sample[["date", "symbol"] + list(FIXED.features)].copy()
    with threadpool_limits(limits=2):
        reference["expected_prediction"] = loaded.predict(sample[list(FIXED.features)])
    if REFERENCE.exists():
        old = pd.read_csv(REFERENCE, parse_dates=["date"])
        pd.testing.assert_frame_equal(
            reference.reset_index(drop=True), old, check_dtype=False
        )
    else:
        reference.to_csv(REFERENCE, index=False)
    verification = dict(
        selected_candidate="fitting_policy/F00_fixed",
        policy_sha256=digest,
        model_sha256=checksum(ARTIFACT),
        reference_sha256=checksum(REFERENCE),
        reproduced_forecasts=len(scores),
        maximum_forecast_difference=float(np.max(np.abs(scores.score - saved.score))),
        loaded_predictions_exact=True,
        reference_rows=len(reference),
        training_rows=int(fit.training_rows),
        fitted_asof=str(fit.fit_date.date()),
        development_return=float(selected_metrics.total_return),
        development_max_drawdown=float(selected_metrics.max_drawdown),
        evaluation_runner_ready=False,
        next_step="Prepare a separate final runner that loads this artifact without calling fit; verify existing accounting interfaces, then run the required comparison on explicit instruction.",
        post2015_predictions=0,
        post2015_accounts=0,
        preserved_comparators=original_hashes,
        preparation_code_sha256=checksum(PROJECT / "prepare_fixed_anchor.py"),
        python_version=platform.python_version(),
        sklearn_version=sklearn.__version__,
    )
    verify_saved_sources(protected)
    assert all(checksum(p) == h for p, h in original_hashes.items())
    save_json(OUT / "verification.json", verification)
    print(
        json.dumps(
            {
                k: verification[k]
                for k in [
                    "selected_candidate",
                    "reproduced_forecasts",
                    "maximum_forecast_difference",
                    "loaded_predictions_exact",
                    "reference_rows",
                    "fitted_asof",
                    "post2015_predictions",
                    "post2015_accounts",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run()
