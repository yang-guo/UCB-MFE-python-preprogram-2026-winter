"""Focused stability study of fixed rank-Ridge and available-cash deployment.

Declared after the preceding development comparisons, not independent testing.
No seed/feature grid expansion is inferred from the results of this script.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from hw4.data import checksum
from hw4.settings import PROJECT, TradingRules
from quality_robustness import replace_financials, attribute_lots
from research_models import choose_alpha, predict_research
from research_session import (
    load_context,
    evaluate_account,
    verify_saved_sources,
    save_json,
)
from run_quality_robustness import (
    BASE,
    RANKS,
    load_financials,
    extended_account,
    load_daily_eligibility,
)

FAMILY = "rank_model_stability"
OUT = PROJECT / "results" / FAMILY
PARENT = "quality_robustness/U01_fixed"
FIXED = replace(BASE, name="V00_fixed_replay", schedule="fixed")
SPECS = [
    FIXED,
    replace(FIXED, name="V01_delay30"),
    replace(FIXED, name="V02_alpha100"),
    replace(FIXED, name="V03_alpha10000"),
    replace(
        FIXED,
        name="V04_without_six_month",
        features=tuple(f for f in FIXED.features if f != "rank_momentum_126_21"),
    ),
    replace(FIXED, name="V05_roa_only", features=RANKS + ("rank_operating_roa",)),
    replace(
        FIXED,
        name="V06_smaller_roa",
        features=tuple(f for f in RANKS if f != "rank_momentum_126_21")
        + ("rank_operating_roa",),
    ),
    replace(FIXED, name="V07_excess", target="excess"),
]
CHANGES = {
    "V00_fixed_replay": "Replay fixed pre-2013 rank-Ridge account.",
    "V01_delay30": "Additional 30-day financial-event delay in fixed model.",
    "V02_alpha100": "Fixed-model penalty sensitivity alpha=100.",
    "V03_alpha10000": "Fixed-model penalty sensitivity alpha=10000.",
    "V04_without_six_month": "Fixed model without six-to-one-month momentum rank.",
    "V05_roa_only": "Fixed model without cash-flow rank.",
    "V06_smaller_roa": "From V05, also remove six-to-one-month momentum rank.",
    "V07_excess": "Fixed rank model fitted to benchmark-relative outcome plus training-only market offset.",
}
# Each allocator uses the exact saved fixed-model forecasts, not a new fit.
ALLOC = [
    ("W00_cost10", "V00_fixed_replay", dict(fee_bps=10.0)),
    ("W01_cost25", "V00_fixed_replay", dict(fee_bps=25.0)),
    ("W02_top30", "V00_fixed_replay", dict(top_n=30)),
    ("W03_budget_half", "V00_fixed_replay", dict(cohort_fraction=0.5)),
    ("W04_budget_full", "V00_fixed_replay", dict(cohort_fraction=1.0)),
    ("W05_full_cost25", "V00_fixed_replay", dict(cohort_fraction=1.0, fee_bps=25.0)),
    ("W06_delay_full", "V01_delay30", dict(cohort_fraction=1.0)),
    ("W07_delay_full_cost25", "V01_delay30", dict(cohort_fraction=1.0, fee_bps=25.0)),
    ("W08_sector20", "V00_fixed_replay", dict(sector_cap=0.20)),
    ("W09_reinvest", "V00_fixed_replay", dict(reinvest=True)),
    (
        "W10_full_sector20",
        "V00_fixed_replay",
        dict(cohort_fraction=1.0, sector_cap=0.20),
    ),
]


def run():
    variants = [
        dict(
            s.as_dict(),
            parent=(
                f"{FAMILY}/V05_roa_only" if s.name == "V06_smaller_roa" else PARENT
            ),
            change=CHANGES[s.name],
        )
        for s in SPECS
    ]
    variants += [
        dict(
            name=n,
            parent=f"{FAMILY}/{source}",
            change=f"Use {source} forecasts with {settings}; all lots retain 90-day maturity.",
            settings=settings,
        )
        for n, source, settings in ALLOC
    ]
    protocol = dict(
        scope="Fixed rank-model stability and cash deployment, retrospectively motivated development only.",
        variants=variants,
        alpha="Pre-2013 small grid for changed feature/target sets; original 1000 otherwise, explicit sensitivity at100/10000.",
        rules=TradingRules().as_dict(),
        post2015_runs=0,
        selection="No automatic final selection; report all attempts and data incidents. No future-year eligibility or outcomes used.",
        code_sha256={
            n: checksum(PROJECT / n)
            for n in [
                "research_models.py",
                "research_session.py",
                "quality_robustness.py",
                "run_quality_robustness.py",
                "run_rank_model_stability.py",
            ]
        },
    )
    save_json(OUT / "protocol.json", protocol)
    context, protected = load_context()
    protocol["source_sha256"] = protected
    save_json(OUT / "protocol.json", protocol)
    delayed = replace_financials(
        context["panel"], load_financials(), context["sectors"], 30
    )
    all_scores, rows, cvs, audits = {}, [], [], []

    def record(row, result):
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "experiments.csv", index=False)
        if result.status == "complete" and not len(result.incidents):
            lots = attribute_lots(
                result.fills,
                result.open_lots,
                context["quotes"],
                result.history.nav.iloc[-1],
                1e6,
            )
            lots["sector"] = lots.symbol.map(context["sectors"])
            lots.to_csv(OUT / f"{row['variant']}_lot_attribution.csv", index=False)
        print(
            row["variant"],
            result.status,
            f"return={row.get('total_return', np.nan):.2%}",
            f"incidents={len(result.incidents)}",
            flush=True,
        )

    keys0 = None
    for spec in SPECS:
        panel = delayed if spec.name == "V01_delay30" else context["panel"]
        alpha = (
            100.0
            if spec.name == "V02_alpha100"
            else 10000.0
            if spec.name == "V03_alpha10000"
            else 1000.0
        )
        with threadpool_limits(limits=2):
            if spec.features != FIXED.features or spec.target != "return":
                alpha, cv = choose_alpha(panel, context["quotes"].index, spec)
                cvs.append(cv.assign(variant=spec.name))
                pd.concat(cvs, ignore_index=True).to_csv(
                    OUT / "alpha_validation.csv", index=False
                )
            scores, audit, _ = predict_research(
                panel, context["quotes"].index, spec, alpha
            )
        keys = (
            scores[["date", "symbol"]]
            .sort_values(["date", "symbol"])
            .reset_index(drop=True)
        )
        if keys0 is None:
            keys0 = keys
        else:
            pd.testing.assert_frame_equal(keys, keys0)
        all_scores[spec.name] = scores
        audits.append(audit)
        pd.concat(audits, ignore_index=True).to_csv(OUT / "fit_audit.csv", index=False)
        row, result = evaluate_account(context, scores, FAMILY, spec.name)
        row.update(
            alpha=alpha,
            schedule="fixed",
            group="model",
            predicts_return=True,
            parent=next(v["parent"] for v in variants if v["name"] == spec.name),
            change=CHANGES[spec.name],
        )
        if spec.name == "V00_fixed_replay":
            prior = pd.read_csv(
                PROJECT / "results/quality_robustness/U01_fixed_nav.csv"
            )
            np.testing.assert_allclose(result.history.nav, prior.nav, atol=1e-6, rtol=0)
        record(row, result)
    eligible = None
    for name, source, settings in ALLOC:
        rules = replace(
            TradingRules(),
            **{
                k: v for k, v in settings.items() if k not in ["sector_cap", "reinvest"]
            },
        )
        if "sector_cap" in settings or "reinvest" in settings:
            if settings.get("reinvest") and eligible is None:
                eligible = load_daily_eligibility(context, protected)
            row, result = extended_account(
                context,
                all_scores[source],
                name,
                settings.get("sector_cap", 1.0),
                settings.get("reinvest", False),
                eligible,
                family=FAMILY,
                rules=rules,
            )
        else:
            row, result = evaluate_account(
                context, all_scores[source], FAMILY, name, rules
            )
        row.update(
            alpha=1000.0,
            schedule="fixed",
            group="allocation",
            predicts_return=True,
            parent=f"{FAMILY}/{source}",
            change=f"Use {source} forecasts with {settings}.",
        )
        record(row, result)
    verify_saved_sources(protected)
    save_json(
        OUT / "verification.json",
        dict(
            accounts=len(rows),
            complete=sum(r["status"] == "complete" for r in rows),
            clean_complete=sum(
                r["status"] == "complete" and not r["specific_open_data_issue"]
                for r in rows
            ),
            replay=True,
            candidate_keys_identical=True,
            source_and_original_artifact_unchanged=True,
            post2015_runs=0,
        ),
    )


if __name__ == "__main__":
    run()
