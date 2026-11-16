"""Declared follow-up to S13, with model and allocation changes kept separate.

This study is chosen after seeing development results: it is robustness work,
not an untouched validation set. All outcomes, incidents and failed ideas stay.
"""

from dataclasses import replace
import hashlib
import json

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cap_sensitivity_engine import run_sector_backtest
from full_period_cap_tools import calendar_year_returns
from hw4.data import checksum
from hw4.engine import comparison
from hw4.modeling import forecast_metrics
from hw4.settings import DATABASE, PROJECT, TradingRules
from quality_robustness import replace_financials, sector_quality_ranks, attribute_lots
from research_models import ResearchSpec, choose_alpha, predict_research
from research_session import (
    load_context,
    evaluate_account,
    verify_saved_sources,
    save_json,
)
from run_cap_sensitivity import independently_check
from run_development_revision import verify_account
from run_experiments import save_run
from run_sector_reinvestment import verify_reinvestment
from run_signal_research import RANKS
from sector_reinvestment import build_entry_eligibility

FAMILY = "quality_robustness"
OUT = PROJECT / "results" / FAMILY
PARENT = "signal_research/S13_quality_ranks"
FEATURES = RANKS + ("rank_operating_roa", "rank_fcf_to_assets")
BASE = ResearchSpec("U00_replay", FEATURES, schedule="quarterly", feature_clip=False)
# The last two entries distinguish new hypotheses from stability checks.
SPECS = [
    BASE,
    replace(BASE, name="U01_fixed", schedule="fixed"),
    replace(BASE, name="U02_annual", schedule="annual"),
    replace(BASE, name="U03_alpha100"),
    replace(BASE, name="U04_alpha10000"),
    replace(BASE, name="U05_roa_only", features=RANKS + ("rank_operating_roa",)),
    replace(BASE, name="U06_fcf_only", features=RANKS + ("rank_fcf_to_assets",)),
    replace(
        BASE,
        name="U07_quality_only",
        features=("rank_operating_roa", "rank_fcf_to_assets"),
    ),
    replace(
        BASE,
        name="U08_without_six_month",
        features=tuple(f for f in FEATURES if f != "rank_momentum_126_21"),
    ),
    replace(
        BASE,
        name="U09_without_volatility",
        features=tuple(f for f in FEATURES if f != "rank_volatility_63"),
    ),
    replace(
        BASE,
        name="U10_sector_quality",
        features=RANKS + ("sector_rank_operating_roa", "sector_rank_fcf_to_assets"),
    ),
    replace(BASE, name="U11_filing_delay30"),
    replace(BASE, name="U12_excess_target", target="excess"),
    replace(
        BASE,
        name="U13_without_short_momentum",
        features=tuple(f for f in FEATURES if f != "rank_momentum_21"),
    ),
]
CHANGES = {
    "U00_replay": "Exact S13 replay before follow-up comparisons.",
    "U01_fixed": "Fit once before 2013 instead of quarterly.",
    "U02_annual": "Refit annually instead of quarterly.",
    "U03_alpha100": "Penalty sensitivity: alpha=100, retaining all other S13 choices.",
    "U04_alpha10000": "Penalty sensitivity: alpha=10000, retaining all other S13 choices.",
    "U05_roa_only": "Remove cash-flow rank; keep price ranks and profitability rank.",
    "U06_fcf_only": "Remove profitability rank; keep price ranks and cash-flow rank.",
    "U07_quality_only": "Remove every price input; retain two financial percentile ranks.",
    "U08_without_six_month": "Remove six-to-one-month momentum rank to assess incremental information.",
    "U09_without_volatility": "Remove volatility rank to assess incremental information.",
    "U10_sector_quality": "Use within-sector rather than global financial percentiles; static-sector limitation.",
    "U11_filing_delay30": "Lecture exercise: delay every valid financial event an additional 30 calendar days.",
    "U12_excess_target": "Fit stock-minus-benchmark 90-day outcomes, adding training-average benchmark return back.",
    "U13_without_short_momentum": "Remove one-month momentum rank to test a smaller price specification.",
}
ALLOCATION = [
    ("A00_cost0", dict(fee_bps=0.0)),
    ("A01_cost10", dict(fee_bps=10.0)),
    ("A02_cost25", dict(fee_bps=25.0)),
    ("A03_top10", dict(top_n=10)),
    ("A04_top30", dict(top_n=30)),
    ("A05_top40", dict(top_n=40)),
    ("A06_stock3", dict(name_cap=0.03)),
    ("A07_stock6", dict(name_cap=0.06)),
    ("A08_budget_half", dict(cohort_fraction=0.5)),
    ("A09_budget_full", dict(cohort_fraction=1.0)),
    ("A10_sector20", dict(sector_cap=0.20)),
    ("A11_sector10", dict(sector_cap=0.10)),
    ("A12_reinvest", dict(reinvest=True)),
]


def load_financials():
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        f = con.execute("""SELECT symbol,fiscal_date,period,filing_date,reported_currency,
            revenue,net_income,operating_income,total_assets,total_debt,cash_and_equivalents,
            free_cash_flow,operating_cash_flow FROM fundamentals
            WHERE fiscal_date<='2015-12-31'
            AND (filing_date<='2015-12-31' OR filing_date IS NULL)""").df()
    for col in ["fiscal_date", "filing_date"]:
        f[col] = pd.to_datetime(f[col]).astype("datetime64[ns]")
    return f


def load_daily_eligibility(context, protected):
    key = hashlib.sha256(
        json.dumps(
            {
                "source": protected[str(DATABASE)],
                "data_code": checksum(PROJECT / "hw4/data.py"),
                "eligibility_code": checksum(PROJECT / "sector_reinvestment.py"),
                "revision_code": checksum(PROJECT / "development_revision.py"),
                "repairs": checksum(
                    PROJECT / "data_review/development_price_repairs.json"
                ),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    path = PROJECT / f"outputs/selected_caps_development/daily_eligibility_{key}.pkl"
    e = (
        pd.read_pickle(path)
        if path.exists()
        else build_entry_eligibility(DATABASE, context["quotes"], context["panel"])
    )
    assert e.index.equals(context["quotes"].index)
    assert e.columns.equals(context["quotes"].columns)
    return e


def extended_account(
    context, scores, name, caps_fraction, reinvest, eligible, family=FAMILY, rules=None
):
    """Use the same audited engine for sector limits or optional reinvestment."""
    rules = TradingRules() if rules is None else rules
    out = PROJECT / "results" / family
    out.mkdir(parents=True, exist_ok=True)
    local = PROJECT / "outputs" / family / name
    local.mkdir(parents=True, exist_ok=True)
    scores.to_csv(local / "forecasts.csv", index=False)
    symbols = sorted(scores.symbol.unique())
    q = context["quotes"].loc["2012-12-31":, symbols]
    sectors = context["sectors"]
    caps = {s: caps_fraction for s in sectors.unique()}
    e = eligible.reindex(index=q.index, columns=q.columns) if reinvest else None
    result = run_sector_backtest(
        q,
        context["benchmark"].loc[q.index],
        scores,
        "2013-01-01",
        "2015-12-31",
        rules,
        sectors=sectors.to_dict(),
        sector_cap=caps,
        reinvest_maturity_cash=reinvest,
        entry_eligibility=e,
    )
    save_run(result, local)
    if reinvest:
        result.decisions.to_csv(local / "decisions.csv", index=False)
    result.history.to_csv(out / f"{name}_nav.csv")
    result.incidents.to_csv(out / f"{name}_incidents.csv", index=False)
    row = dict(
        variant=name,
        period="continuous_2013_2015",
        status=result.status,
        last_valued_date=result.history.index[-1],
        incident_rows=len(result.incidents),
        specific_open_data_issue="",
        **forecast_metrics(scores, pd.Timestamp("2015-12-31")),
    )
    if result.status == "complete":
        error = independently_check(result, q, sectors, rules, caps)
        if reinvest:
            save_json(
                local / "reinvestment_checks.json",
                verify_reinvestment(result, scores, e, rules),
            )
        metrics = comparison(result, rules)
        metrics.to_csv(out / f"{name}_performance.csv")
        annual = calendar_year_returns(result.history)
        annual.to_csv(out / f"{name}_calendar_years.csv", index=False)
        row.update(metrics.loc["strategy"].to_dict())
        row.update(
            benchmark_return=metrics.loc["benchmark", "total_return"],
            excess_return_pp=100
            * (
                metrics.loc["strategy", "total_return"]
                - metrics.loc["benchmark", "total_return"]
            ),
            average_cash_fraction=(result.history.cash / result.history.nav)
            .iloc[1:]
            .mean(),
            fees=result.fills.fee.sum(),
            buys=result.fills.side.eq("BUY").sum(),
            max_nav_error=error,
        )
        row.update({f"return_{r.year}": r.strategy_return for r in annual.itertuples()})
        if len(result.incidents):
            row["specific_open_data_issue"] = (
                "Unresolved held-price incidents; recorded-price performance is provisional."
            )
    else:
        row["specific_open_data_issue"] = (
            "Incomplete account; no full-period performance assigned."
        )
        verify_account(result, q, rules.initial_cash)
    return row, result


def run():
    protocol = dict(
        scope="Follow-up after observing S13: development robustness, not independent confirmation.",
        variants=[
            dict(s.as_dict(), parent=PARENT, change=CHANGES[s.name]) for s in SPECS
        ]
        + [
            dict(
                name=n,
                parent=PARENT,
                change=f"Same S13 forecasts; allocation/cost change: {v}",
                settings=v,
            )
            for n, v in ALLOCATION
        ],
        alpha="Same pre-2013 grid [1,100,1000] for new feature/target sets; S13 alpha1000 held for schedule/delay changes. Explicit 100/10000 sensitivity is not reselection.",
        baseline_rules=TradingRules().as_dict(),
        data_end="2015-12-31",
        post2015_runs=0,
        selection="Retain every run. Missing/incident-marked accounts cannot be called verified benchmark wins.",
        code_sha256={
            n: checksum(PROJECT / n)
            for n in [
                "research_models.py",
                "research_session.py",
                "quality_robustness.py",
                "run_quality_robustness.py",
            ]
        },
    )
    save_json(OUT / "protocol.json", protocol)
    context, protected = load_context()
    protocol["source_sha256"] = protected
    save_json(OUT / "protocol.json", protocol)
    original = context["panel"]
    fundamentals = load_financials()
    parity = replace_financials(original, fundamentals, context["sectors"], 0)
    financial_cols = [
        "fiscal_date",
        "filing_date",
        "available_date",
        "operating_roa",
        "fcf_to_assets",
        "rank_operating_roa",
        "rank_fcf_to_assets",
    ]
    pd.testing.assert_frame_equal(
        original[financial_cols], parity[financial_cols], check_dtype=False
    )
    delayed = replace_financials(original, fundamentals, context["sectors"], 30)
    both = original.fiscal_date.notna() & delayed.fiscal_date.notna()
    assert (delayed.loc[both, "fiscal_date"] <= original.loc[both, "fiscal_date"]).all()
    sector = sector_quality_ranks(original)
    save_json(
        OUT / "financial_checks.json",
        dict(
            zero_delay_exact_parity=True,
            delayed_never_newer=True,
            changed_roa_rows=int(
                (
                    ~np.isclose(
                        original.operating_roa, delayed.operating_roa, equal_nan=True
                    )
                ).sum()
            ),
            original_known_roa=int(original.operating_roa.notna().sum()),
            delayed_known_roa=int(delayed.operating_roa.notna().sum()),
            limitation="Extra delay does not recover original vendor statement vintages.",
        ),
    )
    print(
        "Financial zero-delay parity and 30-day availability checks passed.", flush=True
    )
    rows, cvs, audits = [], [], []
    reference = None
    baseline_scores = None

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

    for spec in SPECS:
        panel = (
            delayed
            if spec.name == "U11_filing_delay30"
            else sector
            if spec.name == "U10_sector_quality"
            else original
        )
        alpha = (
            100.0
            if spec.name == "U03_alpha100"
            else 10000.0
            if spec.name == "U04_alpha10000"
            else 1000.0
        )
        with threadpool_limits(limits=2):
            if spec.features != FEATURES or spec.target != "return":
                alpha, cv = choose_alpha(panel, context["quotes"].index, spec)
                cvs.append(cv.assign(variant=spec.name))
                pd.concat(cvs, ignore_index=True).to_csv(
                    OUT / "alpha_validation.csv", index=False
                )
            scores, audit, _ = predict_research(
                panel, context["quotes"].index, spec, alpha
            )
        audits.append(audit)
        pd.concat(audits, ignore_index=True).to_csv(OUT / "fit_audit.csv", index=False)
        keys = (
            scores[["date", "symbol"]]
            .sort_values(["date", "symbol"])
            .reset_index(drop=True)
        )
        if reference is None:
            reference = keys
        else:
            pd.testing.assert_frame_equal(reference, keys)
        row, result = evaluate_account(context, scores, FAMILY, spec.name)
        row.update(
            alpha=alpha,
            schedule=spec.schedule,
            parent=PARENT,
            change=CHANGES[spec.name],
            group="model",
            predicts_return=True,
        )
        if spec.name == "U00_replay":
            saved = pd.read_csv(
                PROJECT / "results/signal_research/S13_quality_ranks_nav.csv"
            )
            np.testing.assert_allclose(result.history.nav, saved.nav, atol=1e-6, rtol=0)
            baseline_scores = scores
        record(row, result)
    eligibility = None
    for name, settings in ALLOCATION:
        if "sector_cap" in settings or "reinvest" in settings:
            if settings.get("reinvest") and eligibility is None:
                eligibility = load_daily_eligibility(context, protected)
            row, result = extended_account(
                context,
                baseline_scores,
                name,
                settings.get("sector_cap", 1.0),
                settings.get("reinvest", False),
                eligibility,
            )
        else:
            row, result = evaluate_account(
                context,
                baseline_scores,
                FAMILY,
                name,
                replace(TradingRules(), **settings),
            )
        row.update(
            alpha=1000.0,
            schedule="quarterly",
            parent=PARENT,
            change=f"Same S13 forecasts; {settings}",
            group="allocation",
            predicts_return=True,
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
            control_replay=True,
            candidate_keys_identical=True,
            financial_zero_delay_parity=True,
            source_and_original_artifact_unchanged=True,
            post2015_runs=0,
        ),
    )


if __name__ == "__main__":
    run()
