"""Documented corporate-action correction of the original failed test.

Reuse exactly the frozen forecasts and portfolio configuration. Preserve
every failed run. Supplemental data is used only for holding accounting,
never for features, training, stock selection or parameter search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import duckdb
import numpy as np
import pandas as pd

from backtester import UnresolvedPriceGap, simulate
from evaluate import load_quotes, save_partial, save_results
from research import DATABASE, HERE, OUTPUT, save_json


def supplemental_quotes():
    source = OUTPUT / "CRVO-successor-source.json"
    if not source.exists():
        provenance = json.loads((HERE / "supplemental_quote_source.json").read_text())
        csv = HERE / "supplemental_quotes.csv"
        if hashlib.sha256(csv.read_bytes()).hexdigest() != provenance["csv_sha256"]:
            raise ValueError("Supplemental quote fingerprint mismatch")
        frame = pd.read_csv(csv, parse_dates=["date"])
        frame["date"] = frame.date.astype("datetime64[ns]")
        return frame
    data = json.loads(source.read_text(encoding="utf-8-sig"))["chart"]["result"][0]
    # Check issuer identity before accepting the price series.
    if data["meta"]["symbol"] != "CRVO" or "Cervo" not in data["meta"]["longName"]:
        raise ValueError("Wrong issuer in supplemental quote source")
    frame = pd.DataFrame({
        "date": pd.to_datetime(data["timestamp"], unit="s", utc=True)
                  .tz_convert("America/New_York").normalize().tz_localize(None),
        "symbol": "CRVO",
        "close": data["indicators"]["quote"][0]["close"],
        "adj_close": data["indicators"]["adjclose"][0]["adjclose"],
    })
    frame["date"] = frame.date.astype("datetime64[ns]")
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate supplemental quotes")
    frame = frame.loc[np.isfinite(frame.adj_close) & frame.adj_close.gt(0)].copy()
    # Both vendors express the predecessor's pre-event history in the same
    # post-split units. Check multiple overlapping dates before using ratio 1.
    with duckdb.connect(str(DATABASE), read_only=True) as connection:
        predecessor = connection.execute(
            """SELECT date,adj_close AS predecessor_adjusted FROM prices
               WHERE symbol='DFFN' AND date BETWEEN '2023-08-01' AND '2023-08-16'"""
        ).fetchdf()
    predecessor["date"] = predecessor.date.astype("datetime64[ns]")
    overlap = predecessor.merge(frame, on="date", validate="one_to_one")
    ratio = overlap.predecessor_adjusted / overlap.adj_close
    # Course quotes are rounded to cents; allow a one-cent vendor mismatch,
    # not a fitted multiplier that could hide a split-basis error.
    if len(overlap) < 10 or not np.allclose(
        overlap.predecessor_adjusted, overlap.adj_close, rtol=0, atol=.011
    ):
        raise ValueError("Cannot verify DFFN/CRVO adjustment-basis continuity")
    overlap.to_csv(OUTPUT / "dffn_crvo_overlap_audit.csv", index=False)
    additional_sources = []
    onterris_source = OUTPUT / "ONT-successor-source.json"
    if onterris_source.exists():
        successor = json.loads(onterris_source.read_text(encoding="utf-8-sig"))["chart"]["result"][0]
        if successor["meta"]["symbol"] != "ONT" or "Onterris" not in successor["meta"]["longName"]:
            raise ValueError("Wrong Onterris issuer")
        onterris = pd.DataFrame({
            "date": pd.to_datetime(successor["timestamp"], unit="s", utc=True)
                      .tz_convert("America/New_York").normalize().tz_localize(None),
            "symbol": "ONT",
            "close": successor["indicators"]["quote"][0]["close"],
            "adj_close": successor["indicators"]["adjclose"][0]["adjclose"],
        })
        onterris["date"] = onterris.date.astype("datetime64[ns]")
        with duckdb.connect(str(DATABASE), read_only=True) as connection:
            montrose = connection.execute(
                """SELECT date,adj_close AS predecessor_adjusted FROM prices
                   WHERE symbol='MEG' AND date BETWEEN '2026-04-20' AND '2026-05-22'"""
            ).fetchdf()
        montrose["date"] = montrose.date.astype("datetime64[ns]")
        check = montrose.merge(onterris, on="date", validate="one_to_one")
        if len(check) < 10 or not np.allclose(
            check.predecessor_adjusted, check.adj_close, atol=.011, rtol=0
        ):
            raise ValueError("Cannot establish MEG/ONT price continuity")
        check.to_csv(OUTPUT / "meg_ont_overlap_audit.csv", index=False)
        frame = pd.concat([frame, onterris], ignore_index=True)
        additional_sources.append({
            "symbol": "ONT", "issuer": successor["meta"]["longName"],
            "source": "https://query1.finance.yahoo.com/v8/finance/chart/ONT?period1=1776643200&period2=1790812799&interval=1d&events=div%2Csplits",
            "raw_sha256": hashlib.sha256(onterris_source.read_bytes()).hexdigest(),
            "overlap_dates_checked": len(check),
            "maximum_overlap_price_difference": float(abs(check.predecessor_adjusted - check.adj_close).max()),
        })
    frame.to_csv(HERE / "supplemental_quotes.csv", index=False)
    save_json(HERE / "supplemental_quote_source.json", {
        "symbol": "CRVO", "issuer": data["meta"]["longName"],
        "source": "https://query1.finance.yahoo.com/v8/finance/chart/CRVO?period1=1690848000&period2=1701388800&interval=1d&events=div%2Csplits",
        "retrieved_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "raw_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256((HERE / "supplemental_quotes.csv").read_bytes()).hexdigest(),
        "overlap_dates_checked": len(overlap),
        "maximum_overlap_ratio_error": float(abs(ratio - 1).max()),
        "additional_sources": additional_sources,
        "use": "successor holding accounting only, not predictor inputs",
    })
    return frame


def prepare_actions(records, supplement):
    """Translate verified legal terms into the supplied adjusted-price units."""
    actions = {}
    with duckdb.connect(str(DATABASE), read_only=True) as connection:
        for record in records:
            action = record.copy()
            date = pd.Timestamp(action["effective_date"])
            old = connection.execute(
                """SELECT close,adj_close FROM prices WHERE symbol=?
                   AND date<? AND close>0 AND adj_close>0 ORDER BY date DESC LIMIT 1""",
                [action["symbol"], date],
            ).fetchone()
            if old is None:
                raise ValueError("Missing predecessor adjustment basis")
            if action["kind"] == "cash":
                action["adjusted_cash_per_unit"] = action["cash_per_share"] * old[1] / old[0]
            else:
                new = connection.execute(
                    "SELECT close,adj_close FROM prices WHERE symbol=? AND date=?",
                    [action["successor"], date],
                ).fetchone()
                if new is None:
                    rows = supplement.loc[
                        supplement.symbol.eq(action["successor"])
                        & supplement.date.eq(date)
                    ]
                    if not rows.empty:
                        new = (rows.iloc[0].close, rows.iloc[0].adj_close)
                if new is not None:
                    action["adjusted_unit_ratio"] = action.get(
                        "normalized_unit_ratio",
                        action["legal_share_ratio"] * old[1] / old[0] * new[0] / new[1],
                    )
                    action["successor_initial_mark"] = float(new[1])
            actions[action["symbol"]] = action
    return actions


def run_revision(revision):
    target = OUTPUT / f"final_test_corporate_actions_v{revision}"
    target.mkdir(exist_ok=True)
    if (target / "test_started.json").exists():
        raise RuntimeError("This correction run already exists; preserve it")
    original = json.loads((OUTPUT / "frozen_experiment.json").read_text())
    # Verify original model, predictor code and strategy parameters unchanged.
    for file in ["research.py", "research_config.json"]:
        digest = hashlib.sha256((HERE / file).read_bytes()).hexdigest()
        if digest != original["source_sha256"][file]:
            raise RuntimeError(f"Frozen research configuration changed: {file}")
    if hashlib.sha256((OUTPUT / "frozen_model.joblib").read_bytes()).hexdigest() != original["model_sha256"]:
        raise RuntimeError("Frozen fitted model changed")
    preserved = OUTPUT / "original_frozen_sources/backtester.py"
    if not preserved.exists() and hashlib.sha256((HERE / "backtester.py").read_bytes()).hexdigest() == original["source_sha256"]["backtester.py"]:
        preserved.parent.mkdir(exist_ok=True)
        shutil.copy2(HERE / "backtester.py", preserved)
    if hashlib.sha256(preserved.read_bytes()).hexdigest() != original["source_sha256"]["backtester.py"]:
        raise RuntimeError("Missing original backtester evidence")
    forecasts = OUTPUT / "final_test/predictions.csv"
    records = json.loads((HERE / "corporate_actions.json").read_text())
    supplement = supplemental_quotes()
    actions = prepare_actions(records, supplement)
    save_json(target / "resolved_action_units.json", actions)
    save_json(target / "test_started.json", {
        "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "purpose": "verified corporate-action accounting correction only",
        "original_failed_run": "final_test",
        "frozen_model_sha256": original["model_sha256"],
        "predictions_sha256": hashlib.sha256(forecasts.read_bytes()).hexdigest(),
        "source_sha256": {
            file: hashlib.sha256((HERE / file).read_bytes()).hexdigest()
            for file in ["backtester.py", "evaluate_corporate_actions.py", "corporate_actions.json", "supplemental_quotes.csv"]
        },
        "quote_source": "https://query1.finance.yahoo.com/v8/finance/chart/CRVO?period1=1690848000&period2=1701388800&interval=1d&events=div%2Csplits",
        "quote_source_sha256": json.loads((HERE / "supplemental_quote_source.json").read_text())["raw_sha256"],
        "cash_claim_policy": "mark verified entitlement; restrict cash until original lot maturity; no market-sale fee on cash transfer",
    })
    config = original["config"]
    predictions = pd.read_csv(forecasts, parse_dates=["date"])
    needed = set(predictions.symbol) | {record["successor"] for record in records if record["kind"] == "stock"}
    prices, benchmark = load_quotes("2016-03-31", config["test_end"], needed)
    for symbol, group in supplement.groupby("symbol"):
        prices[symbol] = group.set_index("date").adj_close.reindex(prices.index)
    try:
        result = simulate(
            prices, pd.DatetimeIndex(benchmark.date), predictions,
            config["test_start"], config["test_end"], config["strategy"],
            actions=actions,
        )
    except UnresolvedPriceGap as error:
        save_partial(error, target)
        save_json(target / "evaluation_status.json", {
            "status": "unresolved_data_gap", "reason": str(error),
            "model": original["model"], "no_test_period_retraining": True,
            "revision": revision,
        })
        print(f"Correction run unresolved: {error}", flush=True)
        return
    comparison = save_results(
        result, benchmark.loc[benchmark.date.ge(config["test_start"])],
        config["strategy"], config["test_start"], config["test_end"], target,
    )
    print(comparison.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", type=int, default=1)
    run_revision(parser.parse_args().revision)
