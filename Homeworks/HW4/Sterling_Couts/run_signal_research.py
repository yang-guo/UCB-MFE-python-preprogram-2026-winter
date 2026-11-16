"""Predeclared lecture-first signal study; no portfolio tuning or later prices.

All variants use one continuous 2013–2015 account and the original allocation.
Rule scores are diagnostic comparators, not calibrated return forecasts or an
automatic submission-model choice. Every failure and incident is preserved.
"""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from hw4.data import checksum
from hw4.settings import PROJECT, TradingRules
from research_models import (
    ResearchSpec,
    choose_alpha,
    predict_research,
    score_diagnostics,
)
from research_session import (
    load_context,
    evaluate_account,
    verify_saved_sources,
    save_json,
)
from signal_extensions import add_regime_features

FAMILY = "signal_research"
OUT = PROJECT / "results" / FAMILY
THREE = ("momentum_63", "momentum_252_21", "volatility_63")
FIVE = (
    "momentum_21",
    "momentum_63",
    "momentum_252_21",
    "volatility_63",
    "drawdown_252",
)
QUALITY = ("operating_roa", "fcf_to_assets")
FULL_FIN = (
    "net_margin",
    "operating_roa",
    "debt_to_assets",
    "cash_to_assets",
    "fcf_to_assets",
    "accruals_to_assets",
    "filing_age_days",
)
RANKS = tuple("rank_" + f for f in (*FIVE, "momentum_126_21"))
RELATIVE = ("sector_relative_momentum_63", "sector_relative_momentum_252_21")
REGIME = (
    "market_momentum_63",
    "market_volatility_63",
    "market_trend_200",
    "market_beta_126",
    "residual_momentum_126_21",
)
SPECS = [
    ResearchSpec("S00_one_quarterly", ("momentum_252_21",), schedule="quarterly"),
    ResearchSpec("S01_three_fixed", THREE),
    ResearchSpec("S02_three_quarterly", THREE, schedule="quarterly"),
    ResearchSpec("S03_lecture_price", FIVE, schedule="quarterly"),
    ResearchSpec("S04_price_unclipped", FIVE, schedule="quarterly", feature_clip=False),
    ResearchSpec("S05_six_month", FIVE + ("momentum_126_21",), schedule="quarterly"),
    ResearchSpec(
        "S06_downside_risk",
        FIVE + ("downside_volatility_63", "volatility_acceleration"),
        schedule="quarterly",
    ),
    ResearchSpec("S07_profitability", FIVE + ("operating_roa",), schedule="quarterly"),
    ResearchSpec("S08_cash_flow", FIVE + QUALITY, schedule="quarterly"),
    ResearchSpec(
        "S09_full_lecture", FIVE + FULL_FIN, schedule="quarterly", include_sector=True
    ),
    ResearchSpec("S10_sector_relative", FIVE + RELATIVE, schedule="quarterly"),
    ResearchSpec(
        "S11_relative_quality",
        FIVE + QUALITY + ("sector_relative_operating_roa",),
        schedule="quarterly",
    ),
    ResearchSpec("S12_price_ranks", RANKS, schedule="quarterly", feature_clip=False),
    ResearchSpec(
        "S13_quality_ranks",
        RANKS + ("rank_operating_roa", "rank_fcf_to_assets"),
        schedule="quarterly",
        feature_clip=False,
    ),
    ResearchSpec("S14_excess_target", FIVE, schedule="quarterly", target="excess"),
    ResearchSpec("S15_tree_price", FIVE, schedule="quarterly", estimator="hist"),
    ResearchSpec(
        "S16_tree_quality", FIVE + QUALITY, schedule="quarterly", estimator="hist"
    ),
    ResearchSpec(
        "S17_tree_relative", FIVE + RELATIVE, schedule="quarterly", estimator="hist"
    ),
    ResearchSpec(
        "S18_tree_rank_quality",
        RANKS + ("rank_operating_roa", "rank_fcf_to_assets"),
        schedule="quarterly",
        feature_clip=False,
        estimator="hist",
    ),
    ResearchSpec("S19_tree_fixed", FIVE, estimator="hist"),
    ResearchSpec("S20_market_state", FIVE + REGIME, schedule="quarterly"),
    ResearchSpec(
        "S21_tree_market_state", FIVE + REGIME, schedule="quarterly", estimator="hist"
    ),
    ResearchSpec("R01_low_volatility", ("volatility_63",), rule="low_volatility"),
    ResearchSpec("R02_six_month_trend", ("momentum_126_21",), rule="six_month_trend"),
    ResearchSpec("R03_near_high", ("drawdown_252",), rule="near_high"),
    ResearchSpec(
        "R04_quality_trend_blend",
        ("rank_momentum_126_21", "rank_volatility_63", "rank_operating_roa"),
        rule="quality_trend_blend",
    ),
]
PARENTS = {
    "S00_one_quarterly": "predictor_foundations/F2_quarterly",
    "S01_three_fixed": "predictor_foundations/F1_alpha_selected",
    "S02_three_quarterly": "signal_research/S01_three_fixed",
    "S03_lecture_price": "signal_research/S02_three_quarterly",
    **{
        f"S{i:02}_{name}": "signal_research/S03_lecture_price"
        for i, name in [
            (4, "price_unclipped"),
            (5, "six_month"),
            (6, "downside_risk"),
            (7, "profitability"),
            (10, "sector_relative"),
            (14, "excess_target"),
            (15, "tree_price"),
            (20, "market_state"),
        ]
    },
    "S08_cash_flow": "signal_research/S07_profitability",
    "S09_full_lecture": "signal_research/S08_cash_flow",
    "S11_relative_quality": "signal_research/S08_cash_flow",
    "S12_price_ranks": "signal_research/S05_six_month",
    "S13_quality_ranks": "signal_research/S12_price_ranks",
    "S16_tree_quality": "signal_research/S08_cash_flow",
    "S17_tree_relative": "signal_research/S10_sector_relative",
    "S18_tree_rank_quality": "signal_research/S13_quality_ranks",
    "S19_tree_fixed": "signal_research/S15_tree_price",
    "S21_tree_market_state": "signal_research/S20_market_state",
}


def rule_scores(panel, spec):
    p = panel.loc[
        panel.eligible
        & panel.risk_screen_pass
        & panel.entry_date.between("2013-01-01", "2015-12-31")
    ].copy()
    if spec.rule == "low_volatility":
        score = -p.volatility_63
    elif spec.rule == "six_month_trend":
        score = p.momentum_126_21
    elif spec.rule == "near_high":
        score = p.drawdown_252
    elif spec.rule == "quality_trend_blend":
        score = (
            p.rank_momentum_126_21.fillna(0.5)
            + 1
            - p.rank_volatility_63.fillna(0.5)
            + p.rank_operating_roa.fillna(0.5)
        ) / 3
    else:
        raise ValueError("Unknown diagnostic rule")
    assert score.notna().all()
    result = p[["date", "symbol", "target", "label_end"]].copy()
    result["score"] = score
    result["fit_date"] = pd.NaT
    return result


def run():
    protocol = dict(
        scope="Lecture-first and declared additional signals, development through 2015 only.",
        variants=[
            dict(
                s.as_dict(),
                parent=PARENTS.get(s.name, "predictor_foundations/F0_original"),
                change=str(s.as_dict()),
            )
            for s in SPECS
        ],
        alpha_grid=[1, 100, 1000],
        cv="2011/12 entry-year folds known before 2012-12-31; mean fold RMSE. Alpha fixed thereafter.",
        rules=TradingRules().as_dict(),
        sector_cap=1.0,
        tree="100 iterations, 7 leaves, depth 3, minimum leaf 100, learning rate .05, L2=10, seed42, no random early stopping.",
        sources="Lectures 7/8: price and financial features, six-month momentum, sector-relative transforms, small Ridge grid, fixed/quarterly fitting, shallow boosting.",
        extra_hypotheses="Downside volatility, relative risk, contemporaneous percentile ranks, market-state interactions, excess-return labels and equal-weight quality/trend/risk blend.",
        selection="Every trial retained; unresolved held-price incidents prevent treating headline wealth as verified. No final strategy selected automatically.",
        limitations="Static sector labels; restated vendor financials; dependent overlapping labels; retrospective development comparisons.",
        post2015_runs=0,
    )
    protocol["code_sha256"] = {
        n: checksum(PROJECT / n)
        for n in [
            "research_models.py",
            "research_session.py",
            "signal_extensions.py",
            "run_signal_research.py",
        ]
    }
    save_json(OUT / "protocol.json", protocol)
    context, protected = load_context()
    context["panel"] = add_regime_features(
        context["panel"], context["quotes"], context["benchmark"]
    )
    protocol["source_sha256"] = protected
    save_json(OUT / "protocol.json", protocol)
    cv_cache = {}
    rows, audits, cv_tables, ties = [], [], [], []
    candidate_reference = None
    for spec in SPECS:
        if spec.rule:
            alpha = np.nan
            scores = rule_scores(context["panel"], spec)
        else:
            key = (spec.features, spec.feature_clip, spec.include_sector, spec.target)
            with threadpool_limits(limits=2):
                if spec.estimator == "ridge":
                    if key not in cv_cache:
                        alpha, table = choose_alpha(
                            context["panel"], context["quotes"].index, spec
                        )
                        cv_cache[key] = alpha
                        cv_tables.append(table.assign(variant=spec.name))
                        pd.concat(cv_tables, ignore_index=True).to_csv(
                            OUT / "alpha_validation.csv", index=False
                        )
                    alpha = cv_cache[key]
                else:
                    alpha = 1000.0  # Ignored by the fixed histogram-tree estimator.
                scores, audit, _ = predict_research(
                    context["panel"], context["quotes"].index, spec, alpha
                )
            audits.append(audit)
            pd.concat(audits, ignore_index=True).to_csv(
                OUT / "fit_audit.csv", index=False
            )
        keys = (
            scores[["date", "symbol"]]
            .sort_values(["date", "symbol"])
            .reset_index(drop=True)
        )
        if candidate_reference is None:
            candidate_reference = keys
        else:
            pd.testing.assert_frame_equal(candidate_reference, keys)
        ties.append(score_diagnostics(scores).assign(variant=spec.name))
        row, result = evaluate_account(context, scores, FAMILY, spec.name)
        row.update(
            alpha=alpha,
            schedule=spec.schedule,
            feature_clip=spec.feature_clip,
            estimator=spec.estimator if not spec.rule else "rule_diagnostic",
            parent=PARENTS.get(spec.name, "predictor_foundations/F0_original"),
            change=str(spec.as_dict()),
            predicts_return=spec.rule is None,
        )
        if spec.rule:
            row["RMSE"] = row["MAE"] = (
                np.nan
            )  # Raw ranking scores are not return forecasts.
        if spec.name == "S00_one_quarterly":
            old = pd.read_csv(
                PROJECT / "results/predictor_foundations/F2_quarterly_nav.csv"
            )
            np.testing.assert_allclose(result.history.nav, old.nav, rtol=0, atol=1e-6)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT / "experiments.csv", index=False)
        pd.concat(ties, ignore_index=True).to_csv(OUT / "monthly_ties.csv", index=False)
        print(
            spec.name,
            result.status,
            f"return={row.get('total_return', np.nan):.2%}",
            f"incidents={len(result.incidents)}",
            flush=True,
        )
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
            quarterly_control_replay=True,
            candidate_keys_identical=True,
            source_and_original_artifact_unchanged=True,
            post2015_runs=0,
        ),
    )


if __name__ == "__main__":
    run()
