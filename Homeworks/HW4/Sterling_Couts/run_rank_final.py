"""Evaluate the frozen monthly rank model; preserve every accounting attempt.

No model selection, fitting or parameter changes occur here. Event additions
require factual evidence and never alter the saved forecast file. A missing
held price beyond the allowance stops the account, not an invented exit.
"""

import hashlib
import json

import duckdb
import numpy as np
import pandas as pd

from hw4.data import checksum
from hw4.engine import performance
from hw4.settings import DATABASE, PROJECT, TradingRules
from rank_final_inputs import load_final_context
from rank_event_accounting import run_event_account
from rank_accounting_prices import normalize_accounting_quotes, REPAIRS
from research_session import save_json, verify_saved_sources

OUT = PROJECT / "results/rank_final"


def accounting_inputs(context):
    registry = json.loads((PROJECT / "research_events/events.json").read_text())
    events = list(registry["events"])
    additions = PROJECT / "research_events/rank_model_events.json"
    if additions.exists():
        events += json.loads(additions.read_text())["events"]
    assert len(events) == len({e["id"] for e in events})
    q = context["quotes"].copy()
    required = sorted(
        {s for event in events for s in [event["old"], event.get("new")] if s}
    )
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        raw = con.execute(
            "SELECT symbol,date,close FROM prices WHERE symbol IN (SELECT unnest(?)) AND date<='2026-09-30'",
            [required],
        ).df()
    raw["date"] = pd.to_datetime(raw.date).astype("datetime64[ns]")
    raw = raw.pivot(index="date", columns="symbol", values="close").reindex(q.index)
    extra_notes = []
    for filename in ["supplemental_quotes.csv", "rank_supplemental_quotes.csv"]:
        path = PROJECT / "research_events" / filename
        if not path.exists():
            continue
        extra = pd.read_csv(path, parse_dates=["date"])
        assert not extra.duplicated(["date", "symbol"]).any()
        assert extra.date.le("2026-09-30").all()
        extra = extra.loc[extra.date.isin(q.index)]
        for symbol, frame in extra.groupby("symbol"):
            frame = frame.set_index("date")
            if symbol not in q:
                q[symbol] = np.nan
            assert q.loc[frame.index, symbol].isna().all(), (
                "Never overwrite an observed source quote."
            )
            q.loc[frame.index, symbol] = frame.adj_close
            raw.loc[frame.index, symbol] = frame.close
            extra_notes.append(
                dict(
                    file=filename,
                    symbol=symbol,
                    rows=len(frame),
                    first=str(frame.index.min().date()),
                    last=str(frame.index.max().date()),
                )
            )
    sectors = context["sectors"].to_dict()
    proxies = []
    for event in events:
        new = event.get("new")
        if new and new not in sectors:
            sectors[new] = sectors[event["old"]]
            proxies.append(
                dict(
                    symbol=new,
                    sector=sectors[new],
                    origin=event["old"],
                    policy="Inherited-holding sector proxy only; successor is not added to entry universe.",
                )
            )
    q, unit_audit = normalize_accounting_quotes(q)
    save_json(OUT / "accounting_price_unit_audit.json", unit_audit)
    return q, raw, events, sectors, extra_notes, proxies


def run():
    frozen = json.loads((OUT / "frozen_config.json").read_text())
    fit = json.loads((OUT / "fit_audit.json").read_text())
    assert fit["policy_sha256"] == frozen["sha256"]
    assert (
        checksum(PROJECT / "artifacts/rank_monthly_model.joblib")
        == fit["artifact_sha256"]
    )
    prediction_path = PROJECT / "outputs/rank_final/predictions.csv"
    assert checksum(prediction_path) == fit["predictions_sha256"]
    context, protected = load_final_context()
    q, raw, events, sectors, extra, proxies = accounting_inputs(context)
    scores = pd.read_csv(
        prediction_path,
        parse_dates=["date", "entry_date", "scheduled_exit", "label_end", "fit_date"],
    )
    rules = TradingRules(**frozen["configuration"]["rules"])
    inputs = [
        "rank_event_accounting.py",
        "rank_final_inputs.py",
        "rank_accounting_prices.py",
        "run_rank_final.py",
        "research_events/events.json",
        "research_events/supplemental_quotes.csv",
    ]
    inputs += [f"research_events/price_unit_evidence/{s['symbol']}.json" for s in REPAIRS]
    inputs += [
        n
        for n in [
            "research_events/rank_model_events.json",
            "research_events/rank_supplemental_quotes.csv",
        ]
        if (PROJECT / n).exists()
    ]
    protocol = dict(
        policy_sha256=frozen["sha256"],
        model_sha256=fit["artifact_sha256"],
        forecasts_sha256=fit["predictions_sha256"],
        source_sha256=checksum(DATABASE),
        files={n: checksum(PROJECT / n) for n in inputs},
        settlement_sessions=2,
        sector_proxies=proxies,
        accounting_quote_supplements=extra,
        new_model_fits=0,
        parameter_changes=0,
        events_only_affect_accounting=True,
        price_unit_repairs="Evidence-backed four-series normalization, accounting only; forecasts retain original source inputs.",
    )
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    save_json(OUT / "protocols" / f"{digest}.json", protocol)
    result = run_event_account(
        q,
        context["benchmark"],
        scores,
        raw,
        events,
        "2016-04-01",
        "2026-09-30",
        rules,
        2,
        sectors=sectors,
        sector_cap=0.20,
    )
    local = PROJECT / "outputs/rank_final/runs" / digest
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
    if result["status"] == "complete":
        metrics = pd.DataFrame(
            {
                "strategy": performance(result["history"].nav, rules.annual_rf),
                "benchmark": performance(
                    result["history"].benchmark_nav, rules.annual_rf
                ),
            }
        ).T
        metrics.to_csv(local / "performance.csv")
        metrics.to_csv(OUT / "performance.csv")
        result["history"].to_csv(OUT / "daily_nav.csv")
        result["incidents"].to_csv(OUT / "incidents.csv", index=False)
        print(metrics.to_string(), flush=True)
    if len(result["incidents"]):
        print(result["incidents"].tail(15).to_string(index=False), flush=True)
    verify_saved_sources(protected)
    assert checksum(prediction_path) == fit["predictions_sha256"]


if __name__ == "__main__":
    run()
