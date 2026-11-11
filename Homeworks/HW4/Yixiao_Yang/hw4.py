"""HW4 model fitting, report generation and reproducible experiment entry point.

Run `python hw4.py --database PATH` to regenerate the complete experiment,
or add `--replay` to use the preserved forecasts. The financial definitions,
validation choices, portfolio rules and recorded outcomes are unchanged.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import shutil
import time
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import (
    EXPECTED_SHA256,
    FEATURE_COLUMNS,
    action_frames,
    audit_data_issues,
    load_registry,
    overlay_prices,
    prepare_data,
)
from backtester import (
    DataQualityError,
    benchmark_history,
    performance_metrics,
    run_backtest,
)

HERE = Path(__file__).resolve().parent
REPOSITORY = "https://github.com/yang-guo/UCB-MFE-python-preprogram-2026-winter"

# 1. Fixed research protocol and chronological model fitting

DEV_START = pd.Timestamp("2010-01-01")
DEV_END = pd.Timestamp("2015-12-31")
TEST_START = pd.Timestamp("2016-04-01")
TEST_END = pd.Timestamp("2026-09-30")
ALPHAS = (10.0, 100.0, 1000.0)
VALIDATION_YEARS = (2013, 2014, 2015)
SOURCE_COMMIT = "5212932d7d2e9de49785ee2c7e0395579b42d694"


class TrainingQuantileClipper(TransformerMixin, BaseEstimator):
    """Learn 1%/99% feature limits on each training fold only.

    Adapted from course_data.QuantileClipper. All-missing columns stay
    unbounded, then SimpleImputer handles them without dropping columns.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or not 0 <= self.lower < self.upper <= 1:
            raise ValueError("Expected a matrix and 0 <= lower < upper <= 1")
        self.n_features_in_ = values.shape[1]
        self.lower_bounds_ = np.full(values.shape[1], -np.inf)
        self.upper_bounds_ = np.full(values.shape[1], np.inf)
        observed = np.isfinite(values).any(axis=0)
        self.lower_bounds_[observed] = np.nanquantile(
            values[:, observed], self.lower, axis=0
        )
        self.upper_bounds_[observed] = np.nanquantile(
            values[:, observed], self.upper, axis=0
        )
        return self

    def transform(self, X):
        if not hasattr(self, "lower_bounds_"):
            raise ValueError("Fit the clipper before transforming features")
        values = np.asarray(X, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_:
            raise ValueError("Feature shape changed after fitting")
        return np.clip(values, self.lower_bounds_, self.upper_bounds_)


def make_model(alpha: float = 100.0) -> Pipeline:
    """A small, deterministic numeric model; every learned step is refitted."""
    return Pipeline([
        ("clip", TrainingQuantileClipper()),
        ("impute", SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        )),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=float(alpha))),
    ])


def feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Explicit allowlist: outcomes and security snapshot fields never enter X."""
    return frame.loc[:, FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


def mature_training_rows(panel: pd.DataFrame, asof) -> pd.DataFrame:
    """Only eligible historical entries with outcomes ended BEFORE this fit."""
    asof = pd.Timestamp(asof)
    mask = (
        panel.eligible.fillna(False)
        & panel.entry_date.ge(DEV_START)
        & panel.signal_date.lt(asof)
        & panel.label_end.lt(asof)
        & np.isfinite(panel.target)
    )
    train = panel.loc[mask].sort_values(["signal_date", "symbol"]).copy()
    if not train.empty:
        assert train.label_end.max() < asof
        assert train.signal_date.max() < asof
    return train


def _rmse(actual, predicted) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def _rank_ic(frame: pd.DataFrame, predictions) -> float:
    values = frame[["signal_date", "target"]].copy()
    values["prediction"] = np.asarray(predictions)
    correlations = []
    for _, group in values.groupby("signal_date", sort=True):
        if len(group) >= 5 and group.target.nunique() > 1 and group.prediction.nunique() > 1:
            correlations.append(group.target.corr(group.prediction, method="spearman"))
    return float(np.mean(correlations)) if correlations else float("nan")


def select_alpha(
    panel: pd.DataFrame,
    alphas=ALPHAS,
    validation_years=VALIDATION_YEARS,
) -> tuple[float, pd.DataFrame]:
    """Choose by mean annual validation RMSE; the final test is never consulted.

    Fold membership uses entry year, so a January entry may have a December
    signal. Validation targets must finish within development. Censored rows
    remain in the panel and are counted, not used as an investment screen.
    """
    if not alphas or not validation_years:
        raise ValueError("Need at least one penalty and validation year")
    rows = []
    for year in validation_years:
        candidates = panel.loc[
            panel.eligible.fillna(False) & panel.entry_date.dt.year.eq(year)
        ]
        if candidates.empty:
            raise ValueError(f"No eligible monthly entry candidates for {year}")
        asof = candidates.signal_date.min()
        train = mature_training_rows(panel, asof)
        valid = candidates.loc[
            np.isfinite(candidates.target) & candidates.label_end.le(DEV_END)
        ]
        if train.empty or valid.empty:
            raise ValueError(f"Insufficient matured training/validation data for {year}")
        X_train, X_valid = feature_matrix(train), feature_matrix(valid)
        dummy = np.full(len(valid), float(train.target.mean()))
        shared = {
            "validation_year": int(year),
            "fit_date": asof,
            "training_rows": len(train),
            "last_training_label_end": train.label_end.max(),
            "eligible_validation_rows": len(candidates),
            "evaluated_validation_rows": len(valid),
            "unevaluated_validation_rows": len(candidates) - len(valid),
            "last_validation_label_end": valid.label_end.max(),
        }
        rows.append({**shared, "model": "training_mean", "alpha": np.nan,
                     "rmse": _rmse(valid.target, dummy), "mean_monthly_rank_ic": np.nan})
        for alpha in alphas:
            model = make_model(float(alpha))
            model.fit(X_train, train.target)
            prediction = model.predict(X_valid)
            rows.append({**shared, "model": "ridge", "alpha": float(alpha),
                         "rmse": _rmse(valid.target, prediction),
                         "mean_monthly_rank_ic": _rank_ic(valid, prediction)})
    validation = pd.DataFrame(rows)
    ranking = (
        validation.loc[validation.model.eq("ridge")]
        .groupby("alpha", as_index=False).rmse.mean()
        .sort_values(["rmse", "alpha"], kind="stable")
    )
    return float(ranking.iloc[0].alpha), validation


def generate_walk_forward_scores(
    panel: pd.DataFrame, alpha: float, start_date, end_date
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quarterly expanding-window refits, with fixed hyperparameters and rules."""
    candidates = panel.loc[
        panel.eligible.fillna(False)
        & panel.entry_date.between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ].sort_values(["entry_date", "symbol"]).copy()
    if candidates.empty:
        raise ValueError("No eligible entry candidates in the requested period")
    candidates["entry_quarter"] = candidates.entry_date.dt.to_period("Q")
    score_parts, audits = [], []
    for quarter, group in candidates.groupby("entry_quarter", sort=True):
        asof = group.signal_date.min()
        train = mature_training_rows(panel, asof)
        if train.empty:
            raise ValueError(f"No mature training targets before {asof.date()}")
        model = make_model(alpha)
        model.fit(feature_matrix(train), train.target)
        scored = group[["signal_date", "entry_date", "symbol"]].copy()
        scored["score"] = model.predict(feature_matrix(group))
        scored["fit_date"] = asof
        if not np.isfinite(scored.score).all():
            raise ValueError("Non-finite prediction")
        assert scored.fit_date.le(scored.signal_date).all()
        score_parts.append(scored)
        earlier = panel.loc[
            panel.eligible.fillna(False)
            & panel.entry_date.ge(DEV_START)
            & panel.signal_date.lt(asof)
        ]
        audits.append({
            "entry_quarter": str(quarter), "fit_date": asof, "alpha": float(alpha),
            "training_rows": len(train),
            "first_training_signal": train.signal_date.min(),
            "last_training_signal": train.signal_date.max(),
            "last_training_label_end": train.label_end.max(),
            "earlier_rows_not_yet_usable": len(earlier) - len(train),
            "scored_rows": len(scored),
        })
    return pd.concat(score_parts, ignore_index=True), pd.DataFrame(audits)


def protocol() -> dict[str, Any]:
    """Design choices that do not depend on final-test performance."""
    return {
        "author": "Yixiao Yang", "course_source_commit": SOURCE_COMMIT,
        "development_entries": [str(DEV_START.date()), str(DEV_END.date())],
        "buffer": ["2016-01-01", "2016-03-31"],
        "final_entries": [str(TEST_START.date()), str(TEST_END.date())],
        "features": list(FEATURE_COLUMNS), "model": "Ridge",
        "alphas": list(ALPHAS), "validation_years": list(VALIDATION_YEARS),
        "selection_metric": "mean annual validation RMSE; tie -> smaller alpha",
        "validation_outcome_cutoff": str(DEV_END.date()),
        "training": "expanding, entry >=2010-01-01, label_end strictly before fit",
        "refit": "first signal in each entry quarter",
        "entry_schedule": "first market session each month, prior-close signal",
        "top_n": 20, "cohort_fraction": 1 / 3, "holding_calendar_days": 90,
        "initial_cash": 1_000_000.0, "fee_bps_per_side": 10.0,
        "cash_interest": 0.0, "annual_sharpe_risk_free": 0.02,
        "units": "fractional adjusted-price total-return units",
        "max_stale_sessions": 5,
        "missing_entry": "cancel fixed order, do not rerank",
        "missing_maturity_quote": "stop and report unresolved data gap",
        "quarterly_membership": "entry-quarter members; existing lots stay held",
        "test_end": "mark open lots; no forced liquidation or later quote",
    }


def _json_default(value):
    if isinstance(value, (pd.Timestamp, pd.Period, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.to_dict()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False),
                    encoding="utf-8")


def _save_result(result, output_dir: Path, prefix: str = "") -> None:
    result.history.to_csv(output_dir / f"{prefix}daily_nav.csv", index_label="date")
    result.trades.to_csv(output_dir / f"{prefix}trades.csv", index=False)
    result.orders.to_csv(output_dir / f"{prefix}orders.csv", index=False)
    result.open_positions.to_csv(output_dir / f"{prefix}open_positions.csv", index=False)
    save_json(output_dir / f"{prefix}backtest_diagnostics.json", result.diagnostics)


def make_chart(comparison: pd.DataFrame, path: Path) -> None:
    """Produce the required chart only from a completed, audited full run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(10, 5.6), layout="constrained")
    initial = 1_000_000.0
    ax.plot(comparison.index, comparison.strategy / initial - 1,
            label="Quarterly refitted Ridge strategy (net of costs)", color="#126b72", lw=1.8)
    ax.plot(comparison.index, comparison.benchmark / initial - 1,
            label="S&P 500 total return (^SP500TR)", color="#cf8135", lw=1.8)
    ax.set(title="HW4 | Cumulative total return", ylabel="Cumulative return", xlabel="Date")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(output_dir: Path, metrics: pd.DataFrame, alpha: float,
                 validation: pd.DataFrame, result, data_audit: dict) -> None:
    """Use the same completed-results report for regeneration and frozen replay."""

    write_full_report(output_dir, metrics, alpha, validation, result, data_audit)


def run_experiment(database, output_dir="outputs", batch_size: int = 128) -> dict:
    """Complete pipeline; publish final results only if all strict checks pass."""
    database, output_dir = Path(database).resolve(), Path(output_dir).resolve()
    if not database.is_file():
        raise FileNotFoundError(
            f"Missing {database}. Download equities_top2000.duckdb from bCourses. "
            "The class extract cannot replace the full homework database."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_names = [
        "REPORT.md", "metrics.csv", "partial_period_metrics.csv", "cumulative_returns.png", "comparison.csv",
        "daily_nav.csv", "trades.csv", "orders.csv", "open_positions.csv",
        "backtest_diagnostics.json", "data_audit.json", "validation.csv",
        "fit_audit.csv", "INCOMPLETE.txt", "RUN_STATUS.json", "run_config.json",
        "data_issues.json", "extreme_development_labels.csv",
        "price_overlay_lineage.csv", "corporate_action_evidence.json",
        "annual_returns.csv", "CORPORATE_ACTIONS.md", "FULL_ACCOUNTING_AUDIT.json",
    ]
    generated_names += ["INCOMPLETE_" + name for name in [
        "daily_nav.csv", "trades.csv", "orders.csv", "open_positions.csv",
        "backtest_diagnostics.json",
    ]]
    for name in generated_names:
        (output_dir / name).unlink(missing_ok=True)
    (output_dir / "logs" / "predictions.csv").unlink(missing_ok=True)
    config = protocol()
    config["protocol_sha256"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    config["python"] = platform.python_version()
    config["packages"] = {
        name: importlib.metadata.version(name)
        for name in ["numpy", "pandas", "scikit-learn", "matplotlib", "duckdb"]
    }
    save_json(output_dir / "run_config.json", config)
    save_json(output_dir / "RUN_STATUS.json", {"status": "running", "complete": False})
    started = time.perf_counter()
    stage = "data preparation"
    try:
        print("1/4 Reading and auditing the full database...", flush=True)
        data = prepare_data(database, batch_size=batch_size)
        save_json(output_dir / "data_audit.json", data.audit)
        if not data.audit.get("matches_published_sha256", False):
            raise ValueError(
                "This input does not match the published equities_top2000.duckdb SHA-256. "
                "The final experiment requires the full official bCourses file. "
                "If the instructor replaced it, verify the new manifest before updating the expected hash."
            )
        stage = "development validation"
        print("2/4 Selecting Ridge alpha using development years only...", flush=True)
        alpha, validation = select_alpha(data.panel)
        validation.to_csv(output_dir / "validation.csv", index=False)
        config["selected_alpha"] = alpha
        config["selection_frozen_before_final_scoring"] = True
        save_json(output_dir / "run_config.json", config)
        stage = "walk-forward prediction"
        print(f"3/4 Scoring final period with fixed alpha={alpha:g}...", flush=True)
        scores, fit_audit = generate_walk_forward_scores(
            data.panel, alpha, TEST_START, TEST_END
        )
        fit_audit.to_csv(output_dir / "fit_audit.csv", index=False)
        log_dir = output_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        scores.to_csv(log_dir / "predictions.csv", index=False)
        with gzip.GzipFile(filename=str(output_dir / "generated_predictions.csv.gz"),
                           mode="wb", mtime=0) as archive:
            archive.write((log_dir / "predictions.csv").read_bytes())
        stage = "portfolio simulation"
        print("4/4 Running independent cash-and-lot backtest...", flush=True)

        registry = load_registry()
        accounting_prices, lineage = overlay_prices(
            data.prices, registry, Path(__file__).parent / "supplemental_prices.csv"
        )
        cash_actions, security_actions = action_frames(registry)
        lineage.to_csv(output_dir / "price_overlay_lineage.csv", index=False)
        save_json(output_dir / "corporate_action_evidence.json", registry)
        config["accounting_overlay_sha256"] = registry["file_sha256"]
        config["generated_predictions_sha256"] = hashlib.sha256(
            (log_dir / "predictions.csv").read_bytes()
        ).hexdigest()
        config["accounting_repair_only"] = True
        save_json(output_dir / "run_config.json", config)
        result = run_backtest(
            accounting_prices, scores, TEST_START, TEST_END,
            initial_cash=config["initial_cash"], top_n=config["top_n"],
            cohort_fraction=config["cohort_fraction"], fee_bps=config["fee_bps_per_side"],
            holding_days=config["holding_calendar_days"],
            max_stale_sessions=config["max_stale_sessions"],
            cash_actions=cash_actions, security_actions=security_actions,
        )
        benchmark = benchmark_history(
            data.benchmark, TEST_START, TEST_END, config["initial_cash"]
        )
        comparison = pd.DataFrame({"strategy": result.history.nav, "benchmark": benchmark})
        if comparison.isna().any().any():
            raise ValueError("Strategy and benchmark do not cover the same sessions")
        metrics = pd.DataFrame({
            name: performance_metrics(
                comparison[name], config["initial_cash"], TEST_START, TEST_END,
                risk_free=config["annual_sharpe_risk_free"],
            ) for name in comparison.columns
        }).T
        _save_result(result, output_dir)
        metrics.to_csv(output_dir / "metrics.csv", index_label="portfolio")
        comparison.to_csv(output_dir / "comparison.csv", index_label="date")
        make_chart(comparison, output_dir / "cumulative_returns.png")

        audit_data_issues(database, output_dir)

        write_full_report(output_dir, metrics, alpha, validation, result, data.audit)
        status = {"status": "completed", "complete": True,
                  "runtime_seconds": round(time.perf_counter() - started, 2),
                  "selected_alpha": alpha, "output_dir": str(output_dir)}
        save_json(output_dir / "RUN_STATUS.json", status)
        print(metrics.to_string(float_format=lambda x: f"{x:.6f}"), flush=True)
        print(f"Completed. Read {output_dir / 'REPORT.md'}", flush=True)
        return status
    except Exception as exc:
        for name in ["REPORT.md", "metrics.csv", "cumulative_returns.png", "comparison.csv"]:
            (output_dir / name).unlink(missing_ok=True)
        if isinstance(exc, DataQualityError):
            partial = getattr(exc, "partial_result", None)
            if partial is not None:
                _save_result(partial, output_dir, prefix="INCOMPLETE_")
        status = {"status": "incomplete", "complete": False, "stage": stage,
                  "error_type": type(exc).__name__, "reason": str(exc)}
        save_json(output_dir / "RUN_STATUS.json", status)
        (output_dir / "INCOMPLETE.txt").write_text(
            "This run is incomplete. Do not report it as the final homework result.\n"
            + str(exc) + "\n", encoding="utf-8"
        )
        if isinstance(exc, DataQualityError):
            status.update(status="data_gap_documented", analysis_report_complete=True)
            save_json(output_dir / "RUN_STATUS.json", status)
            (output_dir / "REPORT.md").write_text(
                "# Incomplete experiment — unresolved observation\n\n"
                f"The strict accounting checks stopped at: **{exc}**.\n\n"
                "The INCOMPLETE_ files preserve the coherent recorded prefix. "
                "No complete-period strategy metric is reported. Verify the "
                "security identity, corporate action and actual historical quote "
                "before adding any new evidence to the separate accounting overlay.\n",
                encoding="utf-8",
            )
            print(f"Unresolved data gap: {exc}", flush=True)
            return status
        raise


# 2. Results, interpretation and sourced corporate-action appendix

def _table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(v) for v in row) + " |" for row in rows),
    ])


def write_action_appendix(output, registry):
    lines = [
        "# Corporate actions and security continuity", "",
        "This appendix identifies the economic holdings behind legacy research tickers. "
        "These inputs repair portfolio accounting only. They do not change model "
        "features, labels, predictions, historical membership, or the selected names.", "",
        "Mandatory cash consideration becomes a non-spendable cash claim on the "
        "event's accounting date. It remains at fixed value until the original "
        "90-calendar-day maturity, then releases with the strategy's one modeled "
        "10-basis-point exit charge. Actual broker-credit delays are not observed. "
        "The fee is a simulation convention, not a claim about issuer commissions. "
        "Mandatory share conversions preserve the original maturity and cost basis. "
        "Fractional adjusted-price units are allowed throughout.", "",
        "The cash basis formula is `cash_per_raw_share × old_adj_close / old_close`. "
        "For a share exchange, the new adjusted-unit ratio is "
        "`raw_exchange_ratio × (old_adj/old_close) / (new_adj/new_close)`. "
        "A pure ticker alias with equal raw closes can be continued in the old "
        "adjusted-price unit using `old_adj / successor_adj` at a verified overlap. "
        "No separate dividend is credited when the adjusted-price series already "
        "includes it.", "",
    ]
    for event in registry["events"]:
        lines += [f"## {event['symbol']} — {event.get('historical_issuer', event['event_type'])}", ""]
        fields = [
            ("Action ID", event["action_id"]),
            ("Kind", event["event_type"]),
            ("Effective/accounting date", event["effective_date"]),
        ]
        for key, label in [
            ("cash_per_raw_share", "Cash per actual share/instrument"),
            ("cash_per_adjusted_unit", "Cash per adjusted-price unit"),
            ("successor_symbol", "Successor database identifier"),
            ("raw_exchange_ratio", "Actual-share exchange ratio"),
            ("units_per_adjusted_unit", "Successor units per old adjusted unit"),
            ("price_multiplier", "Successor-price multiplier"),
            ("reference_price_date", "Price-basis reference date"),
            ("reference_raw_close", "Reference old raw close"),
            ("reference_adj_close", "Reference old adjusted close"),
            ("overlay_start", "Quote-continuation start"),
            ("overlay_end", "Quote-continuation end"),
        ]:
            if key in event:
                fields.append((label, event[key]))
        lines += [_table(["Field", "Value"], fields), ""]
        for key in ["entitlement", "application_scope", "settlement_assumption",
                    "issuer_identity_warning", "calculation", "notes"]:
            if event.get(key):
                value = event[key]
                lines += [value if isinstance(value, str) else json.dumps(value), ""]
        urls = set()
        for source in event.get("sources", []):
            if source["url"] not in urls:
                lines += [f"- [{source.get('document', 'Primary source')}]({source['url']}): "
                          f"{source.get('supports', '')}"]
                urls.add(source["url"])
        if event["source_url"] not in urls:
            lines += [f"- [Primary source]({event['source_url']})"]
        lines += [""]
    lines += [
        "## Supplemental daily prices", "",
        "`../supplemental_prices.csv` contains the individual observed quotes, "
        "original provider identifiers, unit multipliers and source URLs. "
        "`../supporting_evidence.zip` preserves retrieval evidence under "
        "`verification/market_data_sources/`, with source records "
        "and overlap checks. `price_overlay_lineage.csv` enumerates the "
        "quotes actually supplied to accounting. Missing sessions were "
        "filled only by sourced observations for the same economic security; "
        "there is no interpolation or unlimited stale-price carry.", "",
    ]
    (output / "CORPORATE_ACTIONS.md").write_text("\n".join(lines), encoding="utf-8")


def write_full_report(output, metrics, alpha, validation, result, data_audit):
    output = Path(output)
    config = json.loads((output / "run_config.json").read_text())
    registry = json.loads((output / "corporate_action_evidence.json").read_text())
    issues = json.loads((output / "data_issues.json").read_text())
    comparison = pd.read_csv(output / "comparison.csv", parse_dates=["date"]).set_index("date")
    fit_audit = pd.read_csv(output / "fit_audit.csv")
    s, b = metrics.loc["strategy"], metrics.loc["benchmark"]
    gap = s.final_value - b.final_value
    outcome = "outperformed" if gap > 0 else "underperformed" if gap < 0 else "matched"
    metric_rows = []
    for key, name, fmt in [
        ("final_value", "Final portfolio value", "money"),
        ("total_return", "Cumulative return", "percent"),
        ("cagr", "CAGR", "percent"),
        ("max_drawdown", "Maximum drawdown (signed)", "percent"),
        ("sharpe", "Daily-return Sharpe; 2% annual risk-free", "number"),
    ]:
        values = [s[key], b[key]]
        formatted = [f"${v:,.2f}" if fmt == "money" else f"{v:.2%}" if fmt == "percent"
                     else f"{v:.4f}" for v in values]
        metric_rows.append([name, *formatted])
    metric_table = _table(["Metric", "Strategy, net of modeled costs", "S&P 500 total return"], metric_rows)
    grouped = validation.loc[validation.model.eq("ridge")].groupby("alpha").rmse.mean()
    baseline = validation.loc[validation.model.eq("training_mean"), "rmse"].mean()
    validation_table = _table(["Model", "Mean annual validation RMSE (decimal return)"],
                             [[f"Ridge, alpha={a:g}", f"{v:.6f}"] for a, v in grouped.items()]
                             + [["Training-mean prediction", f"{baseline:.6f}"]])
    selected_folds = validation.loc[validation.model.eq("ridge") & validation.alpha.eq(alpha)]
    fold_table = _table(["Validation year", "Evaluated labels", "RMSE", "Mean monthly rank IC"], [
        [r.validation_year, f"{r.evaluated_validation_rows:,}", f"{r.rmse:.6f}",
         f"{r.mean_monthly_rank_ic:.6f}"] for r in selected_folds.itertuples()
    ])
    daily = comparison.pct_change(fill_method=None).iloc[1:]
    annual = (1 + daily).groupby(daily.index.year).prod() - 1
    annual.index.name = "year"
    annual.to_csv(output / "annual_returns.csv")
    annual_table = _table(["Year", "Strategy", "Benchmark"], [
        [f"{year} (partial)" if year in [2016, 2026] else year,
         f"{r.strategy:.2%}", f"{r.benchmark:.2%}"] for year, r in annual.iterrows()
    ])
    trades, history, positions = result.trades, result.history, result.open_positions
    terminal = history.iloc[-1]
    closed = trades.loc[trades.side.isin(["SELL", "CASH_CLAIM_RELEASE"])]
    event_table = _table(["Accounting event", "Recorded lot events"], [
        [kind, int(trades.side.eq(kind).sum())]
        for kind in ["BUY", "SELL", "CASH_MERGER_CONVERSION", "CASH_CLAIM_RELEASE", "SECURITY_CONVERSION"]
    ])
    action_symbols = ", ".join(sorted({e["symbol"] for e in registry["events"]}))
    proof = output / "FULL_ACCOUNTING_AUDIT.json"
    audit_note = (
        "The independent reconstruction is included in `FULL_ACCOUNTING_AUDIT.json`. "
        "It reads the original database, evidence overlay and event ledger without "
        "importing the backtester, then recomputes cash, holdings, NAV, maturity "
        "dates, prior-close order budgets and risk metrics."
        if proof.is_file() else
        "For the independent audit tied to the supplied frozen forecast evidence, "
        "run the README's quick-replay command followed by "
        "`audit_hw4.py`. Full regeneration is separately "
        "checked in `../verification/REPRODUCIBILITY_AUDIT.json`. The independent "
        "audit's generated `FULL_ACCOUNTING_AUDIT.json` records actual errors "
        "and pass/fail checks without importing the backtester."
    )
    frozen_hash = config.get("frozen_predictions_sha256", config.get("generated_predictions_sha256", "recorded in run_config.json"))
    dev = issues["development"]
    worst = closed.nsmallest(5, "realized_pnl")
    loss_table = _table(["Original security", "Entry", "Exit/release", "Net realized P&L"], [
        [getattr(r, "original_symbol", r.symbol), str(r.entry_date)[:10], str(r.date)[:10],
         f"${r.realized_pnl:,.2f}"] for r in worst.itertuples()
    ])
    report = f"""# HW4 findings — Yixiao Yang

## 1. Completed experiment and result

The recorded portfolio now covers **April 1, 2016 through September 30, 2026**,
all **{len(history) - 1:,} market sessions**, with the original $1,000,000 principal.
The full course database was verified against its published SHA-256 before use.
The strategy **{outcome}** the S&P 500 total-return benchmark by
**${abs(gap):,.2f}** of final value. The signed cumulative-return difference is
**{100 * (s.total_return - b.total_return):.2f} percentage points**, and the
signed CAGR difference is **{100 * (s.cagr - b.cagr):.2f} percentage points**.
These are simulated results under the stated data and execution assumptions.

{metric_table}

![Strategy and benchmark cumulative returns](cumulative_returns.png)

The original unaugmented run stopped at BID in October 2019. Its failure
summary and original fit/validation evidence are retained in the supporting
evidence ZIP under `verification/original_incomplete_attempt/`. I investigated
the security identity and mandatory consideration, then completed accounting
with source-verified corporate actions and missing historical observations.
**The 249,796 forecasts, alpha, features, ranking and allocation rules were
not retuned.** Original prediction SHA-256: `{frozen_hash}`.
This is a repaired accounting evaluation of the same frozen experiment,
not a claim that the final period remained unseen during the investigation.

## 2. Model, features and information timing

Ridge regression predicts the gross adjusted-price return from the next
session's entry close to the first market session on or after entry plus
90 calendar days. Ridge was chosen as a transparent numeric baseline with
one regularization parameter; strong performance was not assumed.

The 12 inputs are 21- and 63-session momentum; 252-session momentum excluding
the latest 21 sessions; annualized 63-session return volatility; drawdown
from the 252-session high; quarterly net margin, operating income/assets,
debt/assets, cash/assets, free cash flow/assets, accruals/assets; and filing
age in days. Exact formulas are in `../HW4.ipynb` and `../data.py`.

Price features use only the signal close and earlier observations. They
are calculated without filling gaps. Quarterly financial reports become
available at filing date plus one calendar day. Backward joins exclude
future reports, reject filing dates on/before fiscal period end, and prevent
a late filing of an older fiscal period from displacing a newer known one.
Financial ratios become missing after 550 days from fiscal end.

Every fit learns feature clipping at its training 1st/99th percentiles,
median imputation with missing indicators, and standardization only on its
training sample. This clips input features, **not the target return**.
Current sector and active-status snapshots are not model inputs or screens.

Eligibility requires historical entry-quarter membership and 63 complete
trailing daily returns. Inactive securities remain in the union. A missing
future exit quote does not remove a prediction candidate. Once purchased,
a lot remains held if the issuer leaves subsequent quarterly universes.

## 3. Development validation and the unsuccessful modeling result

Development entries are January 2010–December 2015. Annual chronological
validation folds use entry years 2013, 2014 and 2015. A fit includes only
signals before its date and targets whose exact 90-day exit is strictly
earlier than the fit. Validation used outcomes ending by December 31, 2015.
The January–March 2016 buffer permits earlier targets to finish, with no
sampled entry dates in that buffer. The final-period rule makes quarterly
expanding-window refits using newly completed outcomes, holding the design
and chosen alpha fixed. All {len(fit_audit)} final fit dates are exported.

The prespecified rule minimizes the equally weighted mean annual RMSE among
alpha 10, 100 and 1000; ties favor the smaller alpha. It selected **{alpha:g}**.

{validation_table}

{fold_table}

The selected Ridge performs worse than the training-mean reference in
development RMSE, with negative mean rank IC in two validation years.
This is an unsuccessful baseline, not evidence of a reliable predictive edge.
The reference model was a diagnostic, not an unrecorded portfolio-selection
alternative. I preserved the originally specified Ridge experiment rather
than switching models after viewing the final path.

Independent SQL diagnosis found **{dev['completed_by_development_end']:,}**
eligible labels completed by the development cutoff. Of these, 17 exceed a
decimal return of 10, five exceed 100, and two exceed 1,000. The most extreme
AAM observation uses an entry adjusted price of 0.1915 and an exit value of
20,461.50, implying a decimal return of about 106,847.56. These positive finite
numbers pass a basic numeric check but are not verified economic returns.
The largest training target accounts for about 79.4% of centered squared
deviations in the 2013 fold; the largest ten account for about 99.997%.
Feature clipping cannot fix such target contamination. The training-mean
baseline also inherits the contaminated mean. See `data_issues.json` and
`extreme_development_labels.csv` for the original evidence.

## 4. Portfolio construction and accounting

At the last close before each month's first market session, eligible names
are ranked by predicted return, then ticker for ties. Up to 20 fixed orders
are scheduled for the next session's close. A cohort's inclusive budget is
`min(previous-close available cash, previous-close NAV / 3)`, with one
twentieth assigned to each selected slot. Fewer candidates leave cash unused.
Same-day maturity receipts cannot increase orders already decided yesterday.

The account uses fractional adjusted-price units, long-only holdings, no
borrowing and no leverage. A buy with inclusive budget B at adjusted price P
receives `B / (P × 1.001)` units. Ordinary sales return `units × P × 0.999`.
Cash earns 0%. Each lot has its own 90-calendar-day maturity; a market closure
rolls the exit to the next session. Repeated monthly positions in one issuer
can overlap, so there is no aggregate issuer-weight cap.

Dividends are not separately added to adjusted-price returns. At the test
end, still-open lots are marked rather than forcibly sold. Their later
maturities and prices after September 30, 2026 are not used.

{event_table}

Final available cash is **${terminal.cash:,.2f}**, stocks are
**${terminal.stock_value:,.2f}**, and cash claims are
**${terminal.cash_claim_value:,.2f}**, totaling **${terminal.nav:,.2f}**.
There are **{len(positions)}** open lots, with
**{int(result.diagnostics['terminal_stale_lots'])}** stale final marks.
Total modeled costs are **${trades.fee.sum():,.2f}**. Average available cash
as a fraction of daily NAV is **{result.diagnostics['mean_cash_fraction']:.2%}**.
`trades.csv`, `orders.csv`, `open_positions.csv` and `daily_nav.csv` provide
the corresponding accounting trail.

## 5. Corporate actions and missing observations

The original strict rule still cancels a missing-entry order without reranking,
carries an ordinary missing mark for at most five sessions, and stops at a
missing maturity quote or a sixth stale session. The completed path did not
require disabling those checks. Instead, sourced observations or legally
verified consideration resolve the relevant data gaps.

The separate evidence registry covers **{len(registry['events'])}** events or
identity continuations: {action_symbols}. Full dates, amounts, unit formulas
and primary-source links appear in [CORPORATE_ACTIONS.md](CORPORATE_ACTIONS.md)
and `../corporate_actions.json`.

For a mandatory cash merger or redemption, the stock/instrument converts to
a fixed cash claim on the effective daily accounting date. The conversion
does not create a fictitious exchange sale. The claim counts in NAV but is
ring-fenced from new purchases until the lot's original maturity, when one
modeled exit charge is applied. This explicitly preserves the original
capital-allocation horizon. Actual broker payment timing and investor-specific
taxes are not observable or modeled. For mandatory stock mergers, units convert
to the sourced successor ratio while cost and original maturity remain intact.

Some original eligible names are warrants or baby bonds despite being in
the course's stock universe. The frozen no-exercise strategy receives the
mandatory default cash redemption for SOFIW and CIFRW; the optional, more
valuable cashless exercise is not retroactively chosen. RCB is a 6.2% note,
so its mandatory redemption includes the contractually accrued interest;
the evidence appendix shows that calculation. These instrument-classification
problems are substantive limitations of the original selection experiment.

Pure renames use observed successor quotes, with an explicit adjustment-basis
bridge where necessary. The NCR lot entered after the earlier NCR/NATL split,
so it receives no historical spinoff distribution. Historical WOLF prices are
required for its 2024 holding; its 2025 reorganization cannot supply those
earlier values. Likewise, Sotheby's 2019 BID holding is not connected to a
different issuer later reusing BID. No missing position is silently deleted,
assigned its last quote as a cash payoff, or written to zero without evidence.

These repairs only enter portfolio accounting. Features and training targets
still use the original course observations. Consequently, future legal-identity
research or a currently retrieved price cannot improve earlier model inputs.
The original dataset's remaining positive-but-economically-wrong prices are
not all independently certified merely because the ledger now reconciles.

## 6. Benchmark, metrics and risk

The comparator buys $1,000,000 of `^SP500TR` at the April 1, 2016 close and
holds through September 30, 2026. Both paths have a prior-session $1,000,000
anchor, so the strategy's initial transaction costs are included. CAGR uses
elapsed calendar days between the first and final test closes divided by
365.25. Maximum drawdown uses running portfolio highs and is reported as a
negative percentage. Sharpe uses arithmetic daily excess returns over
`1.02**(1/252) - 1`, sample daily standard deviation, and `sqrt(252)`.

Annual compounded returns, including the two partial calendar years, are:

{annual_table}

This is an unhedged stock/instrument-selection strategy. It takes concentrated
exposure through repeated cohorts, pays trading costs and leaves material
cash uninvested. Its price and financial signals do not constrain issuer
quality, liquidity, market beta or aggregate security weight. Corporate-action
losses, extreme source labels and changing exposure all affect the outcome;
the comparison is not risk-matched to a permanently invested large-cap index.

The five largest individual realized lot losses illustrate what actually
happened; they are not a post-test screening list:

{loss_table}

## 7. Verification and reproducibility

Hand-calculated tests cover fees, cash conservation, overlapping lots,
weekend maturity, fixed budgets, stale and missing quotes, terminal value,
claim conversion/release and stock exchanges. Information-timing tests cover
filing availability, historical membership, target purging and invariance
of earlier features/predictions to changed future inputs. Synthetic fixtures
are used only for checks, not for reported investment results.

{audit_note}

The original official database checksum is `{data_audit['sha256']}`.
`run_config.json` records the original protocol and the separate evidence
fingerprint. `../verification/FROZEN_STRATEGY.json` and the compressed forecast
file permit an exact replay of the same decisions. `python hw4.py
--replay --database PATH --output outputs` replays the supplied forecasts. `python
hw4.py --database PATH` regenerates features, development validation,
quarterly predictions and the same source-reconciled accounting. Minor
floating-point prediction differences across numerical-library versions
are possible; the supplied frozen replay is byte-verifiable.

## 8. Main limitations and next experiment

The course data contain {data_audit['securities_in_union']:,} securities,
including {data_audit['currently_inactive_audit_only']:,} currently inactive
names retained for historical use. Quarterly membership is reconstructed,
not a certified point-in-time feed. Vendor financials may be restated;
filing-date joins cannot undo that. Some security types, issuer identities
and share/split bases are inconsistent, and missing outcomes can bias the
available training-label sample. Overlapping 90-day outcomes are dependent.

Accounting completion does not establish that this is a deployable strategy.
The fixed Ridge model has a clearly weak validation record, severe label
contamination and no demonstrated edge over the benchmark. A next experiment
should first establish stable security IDs and instrument types, verify
split-adjusted targets, and compare robust target treatment, a price-only
baseline and the financial model using development data alone. Liquidity,
issuer caps, costs and cash rules should also be set there. Any changes after
this review require a separately declared experiment and genuinely new
evaluation data; they should not be advertised as a newly untouched test
of the same 2016–2026 period.

## 9. Sources and AI-assisted work

- [HW4 assignment]({REPOSITORY}/blob/{SOURCE_COMMIT}/Lectures/Lecture%208/hw4.md).
- [Course data guide]({REPOSITORY}/blob/{SOURCE_COMMIT}/Lectures/modeling/README.md).
- [Adapted data/preprocessing conventions]({REPOSITORY}/blob/{SOURCE_COMMIT}/Lectures/modeling/course_data.py).
- Corporate-action and supplemental-price sources are linked in the evidence
  appendix and stored alongside their inputs. The database itself is unmodified.

The backtester is a separate implementation for this homework. ChatGPT
assisted with code, source investigation, tests, accounting reconciliation,
documentation and report generation. Numerical statements are calculated
from the supplied data and explicit evidence overlay; they are not invented
example outputs. This package prepares the assignment for review and does
not itself publish a GitHub pull request.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    write_action_appendix(output, registry)


# 3. Exact replay of the preserved forecasts

def replay(database, output=None):
    from audit_hw4 import ensure_supporting_files
    ensure_supporting_files()
    started = time.perf_counter()
    database = Path(database).resolve()
    output = Path(output).resolve() if output else HERE / "outputs_replay"
    output.mkdir(parents=True, exist_ok=True)
    for name in ["validation.csv", "fit_audit.csv", "data_audit.json",
                 "data_issues.json", "extreme_development_labels.csv"]:
        source = HERE / "outputs" / name
        if source.is_file() and source.resolve() != (output / name).resolve():
            shutil.copy2(source, output / name)
    frozen = json.loads((HERE / "verification/FROZEN_STRATEGY.json").read_text())
    raw = gzip.decompress((HERE / "outputs/frozen_predictions.csv.gz").read_bytes())
    if hashlib.sha256(raw).hexdigest() != frozen["predictions_sha256"]:
        raise ValueError("Frozen prediction file changed")
    with database.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != EXPECTED_SHA256:
            raise ValueError("Replay requires the exact official course database")
    scores = pd.read_csv(io.BytesIO(raw), parse_dates=["signal_date", "entry_date", "fit_date"])
    config = frozen["original_run_config"].copy()
    registry = load_registry()
    selected = scores.sort_values(["entry_date", "score", "symbol"],
                                 ascending=[True, False, True], kind="stable")
    selected = selected.groupby("entry_date", sort=False).head(config["top_n"])
    needed = set(selected.symbol)
    needed.update(e["successor_symbol"] for e in registry["events"] if "successor_symbol" in e)
    # Loading only prices of frozen potential orders is a memory optimization.
    # It does not discard candidates, change ranking, or inspect their outcomes.
    with duckdb.connect(str(database), read_only=True) as connection:
        calendar = pd.DatetimeIndex(connection.execute(
            "SELECT date FROM calendar WHERE date BETWEEN '2016-03-31' AND ? ORDER BY date",
            [TEST_END.date()],
        ).fetchdf().date)
        observed = connection.execute(
            "SELECT symbol,date,adj_close FROM prices WHERE symbol IN (SELECT unnest(?)) "
            "AND date BETWEEN '2016-03-31' AND ? ORDER BY date,symbol",
            [sorted(needed), TEST_END.date()],
        ).fetchdf()
        benchmark = connection.execute(
            "SELECT date,adj_close FROM benchmark WHERE symbol='^SP500TR' "
            "AND date BETWEEN '2016-03-31' AND ? ORDER BY date", [TEST_END.date()],
        ).fetchdf().set_index("date").adj_close.reindex(calendar)
    observed.adj_close = observed.adj_close.where(
        np.isfinite(observed.adj_close) & observed.adj_close.gt(0))
    prices = observed.pivot(index="date", columns="symbol", values="adj_close").reindex(
        index=calendar, columns=sorted(set(scores.symbol) | needed))
    prices, lineage = overlay_prices(prices, registry, HERE / "supplemental_prices.csv")
    cash, securities = action_frames(registry)
    config["accounting_overlay_sha256"] = registry["file_sha256"]
    config["frozen_predictions_sha256"] = frozen["predictions_sha256"]
    config["accounting_repair_only"] = True
    save_json(output / "run_config.json", config)
    lineage.to_csv(output / "price_overlay_lineage.csv", index=False)
    save_json(output / "corporate_action_evidence.json", registry)
    save_json(output / "RUN_STATUS.json", {"status": "running", "complete": False})
    print(f"Replaying {len(scores):,} frozen scores, {len(selected):,} potential orders, "
          f"{len(registry['events'])} sourced events...", flush=True)
    complete, failure = True, None
    try:
        result = run_backtest(
            prices, scores, TEST_START, TEST_END,
            initial_cash=config["initial_cash"], top_n=config["top_n"],
            cohort_fraction=config["cohort_fraction"], fee_bps=config["fee_bps_per_side"],
            holding_days=config["holding_calendar_days"],
            max_stale_sessions=config["max_stale_sessions"],
            cash_actions=cash, security_actions=securities,
        )
    except DataQualityError as error:
        complete, failure, result = False, str(error), error.partial_result
    _save_result(result, output, prefix="" if complete else "INCOMPLETE_")
    full_benchmark = benchmark_history(benchmark, TEST_START, TEST_END, config["initial_cash"])
    comparison = pd.DataFrame({"strategy": result.history.nav, "benchmark": full_benchmark})
    comparison.to_csv(output / "comparison.csv", index_label="date")
    supported = comparison.loc[result.history.index]
    supported_metrics = pd.DataFrame({
        name: performance_metrics(supported[name], config["initial_cash"],
                                  TEST_START, supported.index[-1], risk_free=0.02)
        for name in supported.columns
    }).T
    supported_metrics.to_csv(output / ("metrics.csv" if complete else "partial_period_metrics.csv"),
                             index_label="portfolio")
    if complete:
        if comparison.isna().any().any():
            raise ValueError("Full accounting paths must cover identical sessions")
        make_chart(comparison, output / "cumulative_returns.png")

        write_full_report(
            output, supported_metrics, config["selected_alpha"],
            pd.read_csv(output / "validation.csv"), result,
            json.loads((output / "data_audit.json").read_text()),
        )
    status = {
        "status": "completed" if complete else "incomplete", "complete": complete,
        "stage": "frozen-prediction accounting replay", "reason": failure,
        "first_entry_session": str(TEST_START.date()),
        "last_valid_session": str(result.history.index[-1].date()),
        "requested_end": str(TEST_END.date()),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "frozen_predictions_sha256": frozen["predictions_sha256"],
        "selected_alpha": config["selected_alpha"],
        "corporate_actions": len(registry["events"]),
        "overlay_price_rows": len(lineage),
    }
    save_json(output / "RUN_STATUS.json", status)
    print(json.dumps(status, indent=2), flush=True)
    print(supported_metrics.to_string(), flush=True)
    return status


# 4. Command-line entry point

def find_database() -> Path:
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        candidate = parent / "Lectures" / "modeling" / "data" / "equities_top2000.duckdb"
        if candidate.is_file():
            return candidate
    return here / "equities_top2000.duckdb"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=None,
                        help="Path to full equities_top2000.duckdb from bCourses")
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: outputs_regenerated, or outputs_replay with --replay")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay", action="store_true",
                        help="Replay the preserved forecasts instead of refitting the model")
    args = parser.parse_args()
    database = args.database or find_database()
    if not database.is_file():
        parser.error(
            f"Full database not found: {database}. Download equities_top2000.duckdb "
            "from bCourses, then rerun with --database followed by its path."
        )
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    from audit_hw4 import ensure_supporting_files
    ensure_supporting_files()
    output = args.output or HERE / (
        "outputs_replay" if args.replay else "outputs_regenerated"
    )
    if args.replay:
        replay(database, output)
    else:
        run_experiment(database, output, args.batch_size)


if __name__ == "__main__":
    main()
