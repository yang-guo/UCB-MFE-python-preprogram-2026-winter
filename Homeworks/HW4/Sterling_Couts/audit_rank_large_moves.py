"""Check every >20% held-price move against independent provider returns.

This is post-evaluation data verification, never a selection/filtering rule.
Snapshots retain only the nearby quote evidence, URLs, and corporate-action
records. Network failures remain explicit and cannot count as successful checks.
"""

from concurrent.futures import ThreadPoolExecutor
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from hw4.settings import PROJECT

OUT = PROJECT / "results/rank_final_input_revision"


def fetch(group):
    symbol, rows = group
    start = rows.date.min() - pd.Timedelta(days=10)
    end = rows.date.max() + pd.Timedelta(days=10)
    query = urlencode(dict(period1=int(start.tz_localize("UTC").timestamp()),
                           period2=int(end.tz_localize("UTC").timestamp()),
                           interval="1d", events="div,splits"))
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?{query}"
    try:
        with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=30) as r:
            data = json.load(r)["chart"]["result"][0]
        index = pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert(
            "America/New_York"
        ).tz_localize(None).normalize()
        q = pd.DataFrame(dict(close=data["indicators"]["quote"][0]["close"],
                              adj_close=data["indicators"]["adjclose"][0]["adjclose"]), index=index)
        evidence, checks = [], []
        for row in rows.itertuples():
            prior = q.index[q.index < row.date].max()
            observed = q.loc[[prior, row.date]].copy()
            observed["symbol"] = symbol
            evidence.extend(observed.reset_index(names="date").to_dict("records"))
            provider_return = observed.adj_close.iloc[-1] / observed.adj_close.iloc[0] - 1
            checks.append(dict(symbol=symbol, date=row.date, source_return=row.price_return,
                               provider_return=provider_return,
                               difference=row.price_return-provider_return,
                               prior_weight=row.prior_weight,
                               status="checked", source=url))
        return dict(symbol=symbol, source=url, status="checked", nearby_prices=evidence,
                    corporate_actions=data.get("events", {}), checks=checks)
    except Exception as error:
        return dict(symbol=symbol, source=url, status="failed", error=str(error),
                    checks=[dict(symbol=symbol, date=d, status="failed") for d in rows.date])


def run():
    path = PROJECT / "research_events/rank_large_move_evidence.json"
    if path.exists():
        evidence = json.loads(path.read_text())["observations"]
    else:
        moves = pd.read_csv(OUT / "large_move_review.csv", parse_dates=["date"])
        with ThreadPoolExecutor(max_workers=6) as pool:
            evidence = list(pool.map(fetch, moves.groupby("symbol")))
        path.write_text(json.dumps(dict(retrieved_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                                        observations=evidence), default=str, indent=2), encoding="utf-8")
    checks = pd.DataFrame([r for s in evidence for r in s["checks"]])
    checks["date"] = pd.to_datetime(checks.date)
    current = pd.read_csv(OUT / "large_move_review.csv", parse_dates=["date"])
    # The snapshot also retains the REZI discrepancy that was subsequently
    # repaired. Reconcile to the CURRENT account, never silently report stale counts.
    checks = current.merge(checks[["date", "symbol", "provider_return", "status", "source"]],
                           on=["date", "symbol"], how="left", validate="one_to_one")
    assert checks.status.notna().all()
    checks["difference"] = checks.price_return - checks.provider_return
    checks.to_csv(OUT / "large_move_provider_checks.csv", index=False)
    summary = dict(screened_moves=len(checks), provider_checked=int(checks.status.eq("checked").sum()),
                   source_unavailable=int(checks.status.eq("failed").sum()),
                   checked_difference_exceeds_half_percentage_point=int(checks.difference.abs().gt(0.005).sum()))
    (OUT / "large_move_review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(checks.groupby("status").size().to_string())
    print(checks.loc[checks.status.ne("checked") | checks.difference.abs().gt(0.005)].to_string(index=False))


if __name__ == "__main__":
    run()
