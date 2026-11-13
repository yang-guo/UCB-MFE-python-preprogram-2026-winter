"""Research protocol: the single, hash-locked source of truth for HW4 dates.

Why a lock?  The final test period (Apr 2016 - Sep 2026) must not influence any
research decision.  Instructions alone ("don't look") are easy to break by
accident, so this module enforces the firewall in code:

* ``PROTOCOL`` holds every boundary.  Its SHA-256 is hard-coded below; editing a
  date without also editing the hash raises ``ProtocolTamperError`` at import,
  so a changed holdout definition can never slip through silently.
* ``Mode.RESEARCH`` data loads are capped in SQL at ``buffer_end``: research
  code physically never receives a test-period price.
* ``Mode.FINAL`` loads require a ``FinalAuthorization`` token, which only
  ``authorize_final`` creates, and only for a strategy config that was frozen
  with ``scripts/freeze_strategy.py`` beforehand.  Every final run is appended
  to a committed log, so repeated peeking at the holdout is visible in Git.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd

PROTOCOL = {
    "development_start": "2010-01-01",  # first signal date whose label may train/validate
    "development_end": "2015-12-31",    # last research signal date
    "buffer_end": "2016-03-31",         # research labels must complete by this close
    "test_start": "2016-04-01",         # first final-test entry close
    "test_end": "2026-09-30",           # final valuation date; no later prices
    "holding_calendar_days": 90,
    "initial_capital": 1_000_000,
    "benchmark": "^SP500TR",
    "annual_risk_free": 0.02,
}
PROTOCOL_SHA256 = "fad32ad7b3a7f26c3945ecba9ba702d9df3a0b8e7a911b03d4f092ec3a09afa7"

SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
FROZEN_CONFIG = SUBMISSION_ROOT / "config" / "frozen_strategy.json"
FROZEN_LOCK = SUBMISSION_ROOT / "config" / "FROZEN.lock"
FINAL_RUN_LOG = SUBMISSION_ROOT / "results" / "final_runs.jsonl"


class ProtocolTamperError(RuntimeError):
    """The protocol dates no longer match their recorded hash."""


class HoldoutViolation(RuntimeError):
    """Code attempted to use final-test information outside a final run."""


def digest(obj) -> str:
    """Stable SHA-256 of a JSON-serializable object (keys sorted)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


if digest(PROTOCOL) != PROTOCOL_SHA256:  # pragma: no cover - this is the tripwire
    raise ProtocolTamperError(
        "quantlab.protocol.PROTOCOL changed but PROTOCOL_SHA256 did not. Changing the "
        "holdout boundary is a research-design change: document why, then update the hash.")

DEVELOPMENT_START = pd.Timestamp(PROTOCOL["development_start"])
DEVELOPMENT_END = pd.Timestamp(PROTOCOL["development_end"])
BUFFER_END = pd.Timestamp(PROTOCOL["buffer_end"])
TEST_START = pd.Timestamp(PROTOCOL["test_start"])
TEST_END = pd.Timestamp(PROTOCOL["test_end"])
HOLDING_DAYS = int(PROTOCOL["holding_calendar_days"])
INITIAL_CAPITAL = float(PROTOCOL["initial_capital"])
ANNUAL_RF = float(PROTOCOL["annual_risk_free"])


class Mode(str, Enum):
    RESEARCH = "research"
    FINAL = "final"


def data_end(mode: Mode) -> pd.Timestamp:
    """Last date whose prices/filings may be loaded in this mode."""
    return BUFFER_END if Mode(mode) is Mode.RESEARCH else TEST_END


def last_signal_date(mode: Mode) -> pd.Timestamp:
    """Last signal date a model may score in this mode."""
    return DEVELOPMENT_END if Mode(mode) is Mode.RESEARCH else TEST_END


def assert_research_safe(dates, what="dates") -> None:
    """Raise if any date lies after the research data boundary."""
    dates = pd.to_datetime(pd.Series(dates)).dropna()
    if len(dates) and dates.max() > BUFFER_END:
        raise HoldoutViolation(f"{what} reach {dates.max().date()}, beyond research boundary {BUFFER_END.date()}")


@dataclass(frozen=True)
class FinalAuthorization:
    """Proof that a frozen strategy config was checked before a final data load."""

    config_sha256: str
    authorized_at: str
    prior_final_runs: int
    reselection_acknowledged: bool
    _token: object = None

    def __post_init__(self):
        if self._token is not _AUTH_SENTINEL:
            raise HoldoutViolation("FinalAuthorization can only be created by authorize_final()")


_AUTH_SENTINEL = object()


def read_final_log() -> list[dict]:
    if not FINAL_RUN_LOG.is_file():
        return []
    return [json.loads(line) for line in FINAL_RUN_LOG.read_text().splitlines() if line.strip()]


def authorize_final(config: dict, *, acknowledge_reselection: bool = False, note: str = "") -> FinalAuthorization:
    """Check the config against the frozen lock and log the final run.

    Raises if the strategy was not frozen first, if it differs from the frozen
    file, or if a *different* strategy was already evaluated on the holdout
    (unless explicitly acknowledged, which is then recorded as contamination).
    """
    if not FROZEN_LOCK.is_file() or not FROZEN_CONFIG.is_file():
        raise HoldoutViolation("No frozen strategy. Run scripts/freeze_strategy.py on a research config first.")
    lock = json.loads(FROZEN_LOCK.read_text())
    frozen = json.loads(FROZEN_CONFIG.read_text())
    if digest(frozen) != lock["config_sha256"]:
        raise HoldoutViolation("config/frozen_strategy.json was edited after freezing")
    if digest(config) != lock["config_sha256"]:
        raise HoldoutViolation("The final run's config differs from the frozen strategy")
    prior = read_final_log()
    other = {run["config_sha256"] for run in prior} - {lock["config_sha256"]}
    if other and not acknowledge_reselection:
        raise HoldoutViolation(
            f"A different strategy ({sorted(other)}) was already run on the holdout. Re-selecting after "
            "seeing test results contaminates it; pass acknowledge_reselection=True to proceed and record it.")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    FINAL_RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FINAL_RUN_LOG.open("a") as fh:
        fh.write(json.dumps({"config_sha256": lock["config_sha256"], "at": now,
                             "prior_final_runs": len(prior), "reselection_acknowledged": bool(other),
                             "note": note}) + "\n")
    return FinalAuthorization(lock["config_sha256"], now, len(prior), bool(other), _AUTH_SENTINEL)


def require_final(auth) -> None:
    if not isinstance(auth, FinalAuthorization):
        raise HoldoutViolation("Final-mode data requires a FinalAuthorization from authorize_final()")
