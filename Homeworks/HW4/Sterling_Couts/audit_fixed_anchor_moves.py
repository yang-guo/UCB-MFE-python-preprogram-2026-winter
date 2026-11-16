"""Compare large held-price moves with saved or newly retrieved provider evidence.

This review never modifies prices or filters the investment universe. Failed
lookups remain unresolved; a flag or a successful download is not validation.
"""

from concurrent.futures import ThreadPoolExecutor
import json

import pandas as pd

from audit_rank_large_moves import fetch
from hw4.settings import PROJECT
from research_session import save_json


OUT = PROJECT / "results/fixed_anchor_final"
EVIDENCE = PROJECT / "research_events/fixed_anchor_large_move_evidence.json"


def run():
    moves = pd.read_csv(OUT / "large_move_review.csv", parse_dates=["date"])
    if EVIDENCE.exists():
        observations = json.loads(EVIDENCE.read_text())["observations"]
    else:
        prior = json.loads(
            (PROJECT / "research_events/rank_large_move_evidence.json").read_text()
        )["observations"]
        observations = [item for item in prior if item["status"] == "checked"]
        known = {
            (r["symbol"], pd.Timestamp(r["date"]))
            for item in observations
            for r in item["checks"]
        }
        missing = moves.loc[[key not in known for key in zip(moves.symbol, moves.date)]]
        print(
            f"Reviewing {len(moves)} flagged moves; {len(missing)} need new evidence.",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=6) as pool:
            observations += list(pool.map(fetch, missing.groupby("symbol")))
        save_json(
            EVIDENCE,
            dict(
                retrieved_at_utc=pd.Timestamp.now(tz="UTC").isoformat(),
                observations=observations,
            ),
        )
    checks = pd.DataFrame([r for item in observations for r in item["checks"]])
    checks["date"] = pd.to_datetime(checks.date)
    checks = checks.drop_duplicates(["date", "symbol"], keep="last")
    result = moves.merge(
        checks[["date", "symbol", "provider_return", "status", "source"]],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    assert result.status.notna().all()
    result["difference"] = result.price_return - result.provider_return
    result.to_csv(OUT / "large_move_provider_checks.csv", index=False)
    summary = dict(
        screened_moves=len(result),
        provider_checked=int(result.status.eq("checked").sum()),
        source_unavailable=int(result.status.eq("failed").sum()),
        checked_difference_exceeds_half_percentage_point=int(
            result.difference.abs().gt(0.005).sum()
        ),
    )
    save_json(OUT / "large_move_review_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(result.loc[result.difference.abs().gt(0.005)].to_string(index=False))


if __name__ == "__main__":
    run()
