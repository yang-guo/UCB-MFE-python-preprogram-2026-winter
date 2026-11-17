"""Export all fitted stocks at one HW4 FF5 snapshot; never used by the API."""

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw4", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    sys.path.insert(0, str(args.hw4.resolve()))
    import numpy as np
    import pandas as pd
    from backtesting import FamaFrench5, MarketData
    from strategies.factor import FactorTiltStrategy

    signal = pd.Timestamp("2015-10-30")
    features = ["MKT", "SMB", "HML", "RMW", "CMA"]
    version = "hw4-ff5-universe-2015-10-30-v1"
    print(
        "Loading development data; extracting full-universe fit at", signal.date(), flush=True
    )
    data = MarketData.load(args.database, end="2015-12-31")
    factors = FamaFrench5().build(data, start="2005-01-01", end=str(signal.date()))
    strategy = FactorTiltStrategy(factors, window=60, min_obs=24, ramp_months=3)
    prices = data.price_panel(end=signal)
    captured = {}
    original = np.linalg.lstsq
    refit_code = strategy.refit.__func__.__code__

    def record(design, targets, rcond=None):
        # The original refit discards coefficients. Observe its genuine result
        # and its current symbol/formation dates, without rewriting the fit.
        frame = sys._getframe(1)
        while frame is not None and frame.f_code is not refit_code:
            frame = frame.f_back
        if frame is None:
            raise RuntimeError("Unexpected least-squares call outside HW4 refit")
        context = frame.f_locals
        result = original(design, targets, rcond=rcond)
        symbol = context["symbol"]
        dates = context["factor_90"].index[context["positions"]]
        captured[symbol] = dict(
            coefficients=result[0].copy(),
            design=design.copy(),
            targets=targets.copy(),
            formation_dates=dates.copy(),
            expected=context["expected"].reindex(features).to_numpy(dtype=float),
        )
        return result

    with patch("numpy.linalg.lstsq", side_effect=record):
        strategy.refit(None, SimpleNamespace(signal_date=signal, prices=prices))
    if set(captured) != set(strategy.scores.index) or "AAPL" not in captured:
        raise RuntimeError("Extracted model coverage differs from original HW4 scores")
    execution = data.calendar[data.calendar.get_loc(signal) + 1]
    quarter = execution.to_period("Q").start_time
    universe = data.membership.loc[
        data.membership.quarter_start.eq(quarter) & data.membership.selection_date.le(signal)
    ].sort_values("symbol")
    eligible = set(strategy._cross_sections[signal].index)
    members = set(universe.symbol)
    if not set(captured) <= eligible <= members:
        raise RuntimeError("Fitted stocks/eligibility violate dated universe membership")
    models = {}
    for symbol, row in sorted(captured.items()):
        coef = row["coefficients"]
        np.testing.assert_allclose(
            coef[0] + coef[1:] @ row["expected"],
            strategy.scores[symbol],
            rtol=1e-12,
            atol=1e-12,
        )
        models[symbol] = {
            "intercept": float(coef[0]),
            "coefficients": coef[1:].tolist(),
            "observations": len(row["targets"]),
            "training_formation_start": str(row["formation_dates"].min().date()),
            "training_formation_end": str(row["formation_dates"].max().date()),
        }
    membership = {
        "model_version": version,
        "snapshot_date": str(signal.date()),
        "execution_date": str(execution.date()),
        "quarter_start": str(quarter.date()),
        "definition": "Quarterly top-2000 membership with selection_date <= signal. HW4 currency/FF5 screens determine eligibility separately.",
        "members": [
            {
                "symbol": r.symbol,
                "selection_date": str(r.selection_date.date()),
                "eligible": r.symbol in eligible,
                "model_available": r.symbol in captured,
                "unavailable_reason": None
                if r.symbol in captured
                else (
                    "STRATEGY_INELIGIBLE" if r.symbol not in eligible else "NO_VALID_REGRESSION"
                ),
            }
            for r in universe.itertuples()
        ],
    }
    versions = {
        n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scipy", "scikit-learn"]
    }
    payload = {
        "model_version": version,
        "snapshot_date": str(signal.date()),
        "prediction_horizon_calendar_days": 90,
        "feature_names": features,
        "feature_units": "decimal compounded expected factor returns",
        "models": models,
        "preprocessing": {
            "learned_scaling": None,
            "imputation": None,
            "description": "Raw factor returns plus intercept; no learned preprocessing.",
        },
        "library_versions": versions,
    }
    sample = captured["AAPL"]
    example = dict(symbol="AAPL", **dict(zip(features, sample["expected"].tolist())))
    reference = {
        "input": example,
        "expected": dict(
            prediction=float(strategy.scores["AAPL"]),
            model_version=version,
            symbol="AAPL",
            prediction_horizon_calendar_days=90,
        ),
        "origin": "Original full-universe HW4 FactorTiltStrategy.refit scores at 2015-10-30, reconstructed offline before API wrapping.",
        "expected_predictions": {s: float(v) for s, v in strategy.scores.sort_index().items()},
        "tolerance": {"rtol": 1e-9, "atol": 1e-12},
    }
    out = args.output.resolve()
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    for filename, obj in [
        ("artifacts/ff5.json", payload),
        ("artifacts/membership.json", membership),
        ("example_request.json", example),
        ("reference_prediction.json", reference),
    ]:
        (out / filename).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    source_files = [
        "strategies/factor.py",
        "backtesting/famafrench.py",
        "backtesting/market_data.py",
        "data_loading.py",
    ]
    metadata = {
        "model_version": version,
        "signal_date": str(signal.date()),
        "execution_date": str(execution.date()),
        "configuration": strategy.refit_settings(),
        "universe_count": len(members),
        "eligible_count": len(eligible),
        "model_count": len(models),
        "expected_factor_vector": {k: example[k] for k in features},
        "library_versions": versions,
        "hw4_commit": subprocess.check_output(
            ["git", "-C", str(args.hw4), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            f: hashlib.sha256((args.hw4 / f).read_bytes()).hexdigest() for f in source_files
        },
        "artifact_sha256": hashlib.sha256(
            (out / "artifacts/ff5.json").read_bytes()
        ).hexdigest(),
        "membership_sha256": hashlib.sha256(
            (out / "artifacts/membership.json").read_bytes()
        ).hexdigest(),
        "provenance": reference["origin"],
        "snapshot_note": "Reconstructed dated fit, not recovered from the original notebook kernel.",
        "target": "Adjusted stock return: next-session entry; first benchmark session on/after entry+90 calendar days exit. Only exits strictly before signal enter fitting.",
        "factor_note": "Three monthly compounded factor returns approximate the exact 90-calendar-day stock target; inherited unchanged from HW4.",
    }
    (out / "artifacts/metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    fixture = pd.DataFrame(sample["design"][:, 1:], columns=features)
    fixture.insert(0, "formation_date", sample["formation_dates"].astype(str))
    fixture["target_90d"] = sample["targets"]
    fixture.to_csv(out / "artifacts/training_fixture.csv", index=False)
    print(
        json.dumps(
            {
                k: metadata[k]
                for k in ["model_version", "universe_count", "eligible_count", "model_count"]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
