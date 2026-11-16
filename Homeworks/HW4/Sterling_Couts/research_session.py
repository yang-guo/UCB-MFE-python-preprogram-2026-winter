"""Shared source snapshots and independent account checks for new research only."""

import hashlib
import json

import duckdb
import pandas as pd

from cap_sensitivity_engine import run_sector_backtest
from development_revision import corrected_data, remove_fluidigm_alias
from full_period_cap_tools import calendar_year_returns
from hw4.data import build_data, checksum
from hw4.engine import comparison
from hw4.modeling import forecast_metrics
from hw4.settings import DATABASE, PROJECT, TradingRules
from research_models import add_market_features, score_diagnostics
from run_cap_sensitivity import independently_check
from run_development_revision import verify_account
from run_experiments import save_run


def source_snapshot():
    paths = [
        DATABASE,
        PROJECT / "artifacts/model.joblib",
        PROJECT / "hw4/modeling.py",
        PROJECT / "hw4/preprocessing.py",
        PROJECT / "hw4/engine.py",
        PROJECT / "hw4/data.py",
        PROJECT / "cap_sensitivity_engine.py",
        PROJECT / "cap_sensitivity_allocation.py",
        PROJECT / "development_revision.py",
        PROJECT / "data_review/development_price_repairs.json",
        PROJECT / "outputs/sector_caps/continuous_2013_2015/shared_forecasts.csv",
    ]
    return {str(path): checksum(path) for path in paths}


def load_context():
    protected = source_snapshot()
    key = hashlib.sha256(
        json.dumps(
            {**protected, "features_code": checksum(PROJECT / "research_models.py")},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    cache = PROJECT / f"outputs/research_session/context_{key}.pkl"
    if cache.exists():
        context = pd.read_pickle(cache)  # Only this research program's own cache.
    else:
        data = build_data(DATABASE, end="2015-12-31")
        data, _, _ = corrected_data(data)
        data, _ = remove_fluidigm_alias(data)
        with duckdb.connect(str(DATABASE), read_only=True) as con:
            sectors = (
                con.execute("SELECT symbol,sector FROM universe ORDER BY symbol")
                .df()
                .set_index("symbol")
                .sector
            )
        panel = add_market_features(data.panel, data.quotes, data.benchmark, sectors)
        context = dict(
            panel=panel, quotes=data.quotes, benchmark=data.benchmark, sectors=sectors
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(context, cache)
    assert context["quotes"].index.max() == pd.Timestamp("2015-12-31")
    return context, protected


def evaluate_account(context, scores, family, name, rules=None):
    """Run and audit one continuous account; preserve every failure or incident."""
    rules = TradingRules() if rules is None else rules
    out = PROJECT / "results" / family
    local = PROJECT / "outputs" / family / name
    out.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    scores.to_csv(local / "forecasts.csv", index=False)
    score_diagnostics(scores).to_csv(
        local / "monthly_score_diagnostics.csv", index=False
    )
    symbols = sorted(scores.symbol.unique())
    quotes = context["quotes"].loc["2012-12-31":, symbols]
    benchmark = context["benchmark"].loc[quotes.index]
    sectors = context["sectors"]
    caps = {s: 1.0 for s in sectors.unique()}
    result = run_sector_backtest(
        quotes,
        benchmark,
        scores,
        "2013-01-01",
        "2015-12-31",
        rules,
        sectors=sectors.to_dict(),
        sector_cap=caps,
    )
    save_run(result, local)
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
        error = independently_check(result, quotes, sectors, rules, caps)
        metrics = comparison(result, rules)
        metrics.to_csv(out / f"{name}_performance.csv")
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
            fees=float(result.fills.fee.sum()),
            buys=int(result.fills.side.eq("BUY").sum()),
            max_nav_error=error,
        )
        annual = calendar_year_returns(result.history)
        annual.to_csv(out / f"{name}_calendar_years.csv", index=False)
        for r in annual.itertuples():
            row[f"return_{r.year}"] = r.strategy_return
        if len(result.incidents):
            row["specific_open_data_issue"] = (
                "Unresolved held-price incidents; recorded-price performance is provisional."
            )
    else:
        row["specific_open_data_issue"] = (
            "Incomplete account; no full-period performance assigned."
        )
        if len(result.fills):
            verify_account(result, quotes, rules.initial_cash)
    assert result.history.cash.min() >= -1e-6
    return row, result


def verify_saved_sources(protected):
    assert all(checksum(path) == digest for path, digest in protected.items())


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
