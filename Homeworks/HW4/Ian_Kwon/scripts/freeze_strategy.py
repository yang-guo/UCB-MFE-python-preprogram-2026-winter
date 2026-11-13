"""Freeze a research config as THE strategy for the final test.

    python scripts/freeze_strategy.py outputs/experiments/<id>/config.json --rationale "why this one"

Writes config/frozen_strategy.json and config/FROZEN.lock (SHA-256, time,
rationale, development metrics).  ``protocol.authorize_final`` refuses any
config that does not hash to the lock, and refuses a *different* frozen
config once a final run exists, unless re-selection is acknowledged (which
is then recorded in results/final_runs.jsonl).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantlab import protocol  # noqa: E402
from quantlab.experiment import validate_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--rationale", required=True)
    args = ap.parse_args()
    src = Path(args.config)
    cfg = json.loads(src.read_text())
    validate_config(cfg)
    summary_path = src.parent / "summary.json"
    dev = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    if protocol.read_final_log():
        print("WARNING: a final run already exists; changing the frozen strategy now is holdout re-selection.")
    protocol.FROZEN_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    protocol.FROZEN_CONFIG.write_text(json.dumps(cfg, indent=2))
    lock = {"config_sha256": protocol.digest(cfg), "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": str(src), "rationale": args.rationale, "protocol_sha256": protocol.PROTOCOL_SHA256,
            "development_evidence": {k: dev.get(k) for k in ["experiment_id", "validation_mean_IC",
                                                             "validation_IC_by_year", "dev_CAGR", "dev_Sharpe"]}}
    protocol.FROZEN_LOCK.write_text(json.dumps(lock, indent=2, default=str))
    print(json.dumps(lock, indent=2, default=str))


if __name__ == "__main__":
    main()
