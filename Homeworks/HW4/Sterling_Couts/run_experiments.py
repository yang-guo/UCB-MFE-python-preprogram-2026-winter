"""Reproduce the HW4 audit, development comparison and frozen final experiment.

Run from this folder with its Python 3.12 environment. The final command requires
an explicit model choice; development never evaluates post-2015 performance.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import logging
import platform
from importlib.metadata import version

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hw4.data import build_data, checksum
from hw4.engine import comparison, run_backtest
from hw4.modeling import (
    SPECS,
    ROBUST_SPECS,
    TREE_SPECS,
    MOMENTUM_SPECS,
    ALL_SPECS,
    forecast_metrics,
    predict_period,
)
from hw4.settings import CLASS_DATABASE, DATABASE, PROJECT, TradingRules

LOG = logging.getLogger(__name__)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def versions():
    return {
        "python": platform.python_version(),
        **{
            n: version(n)
            for n in ["numpy", "pandas", "scikit-learn", "scipy", "duckdb", "joblib"]
        },
    }


def save_run(result, folder):
    folder.mkdir(parents=True, exist_ok=True)
    result.history.to_csv(folder / "daily_nav.csv")
    result.fills.to_csv(folder / "fills.csv", index=False)
    result.open_lots.to_csv(folder / "open_lots.csv", index=False)
    result.incidents.to_csv(folder / "incidents.csv", index=False)
    write_json(folder / "status.json", {"status": result.status})


def save_audit(data, folder):
    summary = {k: v for k, v in data.audit.items() if k != "history_diagnostics"}
    write_json(folder / "data_audit.json", summary)
    pd.DataFrame(data.audit["history_diagnostics"]).to_csv(
        folder / "history_gaps.csv", index=False
    )


def develop(
    class_only=False, robust=False, quarterly=False, nonlinear=False, momentum=False
):
    data = build_data(CLASS_DATABASE if class_only else DATABASE, end="2015-12-31")
    label = (
        "class_smoke"
        if class_only
        else ("development_robust" if robust else "development")
    )
    specs = ROBUST_SPECS if robust else SPECS
    if nonlinear:
        label, specs = "development_nonlinear", TREE_SPECS
    if momentum:
        label, specs = "development_momentum", MOMENTUM_SPECS
    if quarterly:
        label += "_quarterly"
        specs = [replace(s, quarterly=True) for s in specs]
    result_dir = PROJECT / "results" / label
    result_dir.mkdir(parents=True, exist_ok=True)
    save_audit(data, result_dir)
    rules = TradingRules()
    write_json(
        result_dir / "protocol.json",
        {
            "validation_years": [2013, 2014, 2015],
            "training_signal_start": "2010-01-01",
            "data_end": "2015-12-31",
            "rules": rules.as_dict(),
            "versions": versions(),
            "models": [asdict(s) for s in specs],
            "screen": "If enabled: trailing 252 sessions all have positive recorded volume and no daily price ratio outside [0.5,2]; trailing 63-return annualized volatility <=80%. Only past information; no future survival filter.",
            "selection": "Compare completed net portfolio performance across years, ranking IC, drawdown and concentration; review tradeoffs before freezing. No selection on final-test results.",
            "additional_experiment": "Quarterly refitting may be compared separately after the feature comparison.",
        },
    )
    rows, audits = [], []
    for spec in specs:
        for year in [2013, 2014, 2015]:
            LOG.info("Development: %s / %s", spec.name, year)
            start, end = f"{year}-01-01", f"{year}-12-31"
            scores, fit_audit, _ = predict_period(
                data.panel, data.quotes.index, spec, start, end
            )
            audits.append(fit_audit.assign(validation_year=year))
            result = run_backtest(
                data.quotes, data.benchmark, scores, start, end, rules
            )
            folder = PROJECT / "outputs" / label / spec.name / str(year)
            save_run(result, folder)
            scores.to_csv(folder / "predictions.csv", index=False)
            row = {
                "model": spec.name,
                "year": year,
                "status": result.status,
                "last_valued_date": result.history.index[-1],
                **forecast_metrics(scores, pd.Timestamp(end)),
            }
            if result.status == "complete":
                metrics = comparison(result, rules)
                row.update(metrics.loc["strategy"].to_dict())
                row["benchmark_return"] = metrics.loc["benchmark", "total_return"]
                row["excess_return"] = row["total_return"] - row["benchmark_return"]
                row["benchmark_drawdown"] = metrics.loc["benchmark", "max_drawdown"]
                row["average_cash_fraction"] = (
                    (result.history.cash / result.history.nav).iloc[1:].mean()
                )
                row["fees"] = result.history.fees.sum()
            else:
                LOG.warning(
                    "%s stopped: %s",
                    spec.name,
                    result.incidents.tail(1).to_dict("records"),
                )
            rows.append(row)
            pd.DataFrame(rows).to_csv(result_dir / "experiments.csv", index=False)
    pd.concat(audits).to_csv(result_dir / "fit_audit.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def final_run(model_name, quarterly=False):
    spec = next(
        s for s in ALL_SPECS if s.name == model_name and not s.momentum_reference
    )
    spec = replace(spec, quarterly=quarterly)
    rules = TradingRules()
    config = {
        "model": asdict(spec),
        "rules": rules.as_dict(),
        "start": "2016-04-01",
        "end": "2026-09-30",
        "versions": versions(),
        "selection_basis": "2013-2015 development; see experiments and decision log",
        "source_sha256": checksum(DATABASE),
        "code_sha256": {
            str(p.relative_to(PROJECT)): checksum(p)
            for p in [
                *sorted((PROJECT / "hw4").glob("*.py")),
                PROJECT / "run_experiments.py",
            ]
        },
    }
    folder = PROJECT / "results/final"
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()
    frozen = folder / "frozen_config.json"
    if frozen.exists():
        if json.loads(frozen.read_text())["sha256"] != fingerprint:
            raise ValueError(
                "Final configuration already frozen. Do not tune against the final test."
            )
    else:
        write_json(frozen, {"sha256": fingerprint, "configuration": config})
    data = build_data(DATABASE, end=config["end"])
    save_audit(data, folder)
    scores, audit, models = predict_period(
        data.panel,
        data.quotes.index,
        spec,
        config["start"],
        config["end"],
        final_period=True,
    )
    audit.to_csv(folder / "fit_audit.csv", index=False)
    result = run_backtest(
        data.quotes, data.benchmark, scores, config["start"], config["end"], rules
    )
    save_run(result, PROJECT / "outputs/final")
    scores.to_csv(PROJECT / "outputs/final/predictions.csv", index=False)
    if result.status != "complete":
        write_json(
            folder / "blocked.json",
            {
                "status": result.status,
                "last_valued_date": result.history.index[-1],
                "incidents": result.incidents.tail(10).to_dict("records"),
            },
        )
        raise RuntimeError(
            "Final run blocked by missing held prices; investigate before claiming performance"
        )
    table = comparison(result, rules)
    table.to_csv(folder / "performance.csv")
    result.history.to_csv(folder / "daily_nav.csv")
    result.incidents.to_csv(folder / "incidents.csv", index=False)
    result.open_lots.to_csv(folder / "open_lots.csv", index=False)
    write_json(
        folder / "forecast_metrics.json",
        forecast_metrics(scores, pd.Timestamp(config["end"])),
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for name, col in [
        ("Strategy (net of costs)", "nav"),
        ("S&P 500 total return", "benchmark_nav"),
    ]:
        nav = result.history[col]
        axes[0].plot(nav.index, 100 * (nav / rules.initial_cash - 1), label=name)
        axes[1].plot(nav.index, 100 * (nav / nav.cummax() - 1), label=name)
    axes[0].set(
        title="Frozen HW4 strategy: April 2016–September 2026",
        ylabel="Cumulative return (%)",
    )
    axes[1].set(ylabel="Drawdown (%)", xlabel="Date")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "performance.png", dpi=160)
    plt.close(fig)
    # Serve the initial fitted model in HW5; never silently substitute a later fit.
    fit_date, pipeline = models[0]
    example = (
        data.panel.loc[data.panel.eligible & data.panel.date.eq(fit_date)]
        .dropna(subset=list(spec.features))
        .head(1)
    )
    if example.empty:
        raise ValueError("No finite reference input for artifact parity check")
    x = example[list(spec.features)]
    prediction = float(pipeline.predict(x)[0])
    artifact_dir = PROJECT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "features": list(spec.features),
        "model_version": f"hw4-{fingerprint[:12]}",
        "fit_date": str(fit_date.date()),
        "versions": versions(),
        "target": "gross 90-calendar-day adjusted return from next close",
    }
    joblib.dump(bundle, artifact_dir / "model.joblib")
    restored = joblib.load(artifact_dir / "model.joblib")
    np.testing.assert_allclose(
        restored["pipeline"].predict(x), [prediction], rtol=1e-12
    )
    write_json(
        artifact_dir / "reference_prediction.json",
        {
            "symbol": example.symbol.iloc[0],
            "signal_date": fit_date,
            "features": x.iloc[0].to_dict(),
            "feature_order": list(spec.features),
            "prediction": prediction,
            "model_version": bundle["model_version"],
            "units": "all numeric inputs and output are fractions; volatility is annualized",
        },
    )
    print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=["smoke", "development", "robust", "nonlinear", "momentum", "final"],
    )
    parser.add_argument(
        "--model", choices=[s.name for s in ALL_SPECS if not s.momentum_reference]
    )
    parser.add_argument("--quarterly", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.stage == "final":
        if not args.model:
            parser.error("Final evaluation requires an explicit --model choice")
        final_run(args.model, args.quarterly)
    else:
        develop(
            class_only=args.stage == "smoke",
            robust=args.stage == "robust",
            quarterly=args.quarterly,
            nonlinear=args.stage == "nonlinear",
            momentum=args.stage == "momentum",
        )
