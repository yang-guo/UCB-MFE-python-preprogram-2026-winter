"""Assemble the (signal date x member stock) research panel.

Order of operations matters for leakage and for peer statistics:

1. price features on the full integrity-cleaned history, per symbol, causal
2. sample signal dates (week-end and month-end sessions)
3. dated membership of the *entry* quarter (selection close <= signal date)
4. eligibility screen using only information at the signal close
5. financial as-of join
6. sector-relative + cross-sectional ranks over the full eligible
   cross-section of each date (never one stock at a time)
7. labels (future prices, only in target/label columns)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from . import features as F
from . import integrity as I
from .data import MarketData, Reference
from .labels import label_arrays

LABEL_COLUMNS = ["target", "label_end", "entry_date", "label_has_unreliable_break", "target_rank",
                 "target_demeaned"]


@dataclass(frozen=True)
class PanelConfig:
    min_valid_returns_63: int = 57        # >= 90% of the last 63 daily returns observed
    min_dollar_volume: float = 1_000_000  # median close x volume over 63 sessions
    max_unreliable_breaks_252: int = 1    # unclassified/bad-print breaks tolerated in a year
    holding_days: int = 90
    start: str = "2009-01-01"

    def to_dict(self):
        return asdict(self)


def signal_grid(calendar: pd.DatetimeIndex, start, end) -> np.ndarray:
    """Row indices of week-end and month-end sessions in [start, end]."""
    s = pd.Series(np.arange(len(calendar)), index=calendar)
    s = s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    week_end = s.groupby(s.index.to_period("W-FRI")).max()
    month_end = s.groupby(s.index.to_period("M")).max()
    return np.unique(np.concatenate([week_end.to_numpy(), month_end.to_numpy()]))


def membership_rows(calendar: pd.DatetimeIndex, rows: np.ndarray, membership: pd.DataFrame) -> pd.DataFrame:
    """(row, symbol) pairs where symbol is a member of the next-session entry quarter."""
    T = len(calendar)
    rows = rows[rows + 1 < T]  # the last loaded session has no entry session
    signal = calendar[rows]
    entry = calendar[rows + 1]
    frame = pd.DataFrame({"row": rows, "date": signal, "entry_date": entry,
                          "quarter_start": entry.to_period("Q").start_time})
    out = frame.merge(membership[["quarter_start", "symbol", "selection_date", "universe_rank", "market_cap_proxy"]],
                      on="quarter_start", how="inner")
    if not out.selection_date.le(out.date).all():
        raise AssertionError("Membership selected after the signal close")
    return out


def build_panel(market: MarketData, ref: Reference, audit: I.IntegrityResult, end,
                config: PanelConfig = PanelConfig()) -> pd.DataFrame:
    calendar, symbols = market.calendar, market.symbols
    px = audit.clean
    rows = signal_grid(calendar, config.start, end)
    base = membership_rows(calendar, rows, ref.membership)
    base["col"] = symbols.get_indexer(base.symbol)
    if (base.col < 0).any():
        raise ValueError("Member symbol missing from the price matrix")
    r, c = base.row.to_numpy(), base.col.to_numpy()

    feats = F.price_feature_matrices(px, market.close, market.volume)
    for name, mat in feats.items():
        base[name] = mat[r, c].astype(np.float32)
    base["price_t"] = px[r, c]
    base["log_mcap_proxy"] = np.log(base.market_cap_proxy.where(base.market_cap_proxy > 0)).astype(np.float32)

    breaks_recent = I.trailing_count(audit.break_day, I.COOLOFF_SESSIONS)[r, c]
    artifacts_252 = I.trailing_count(audit.artifact_day | audit.quarantined, 252)[r, c]
    unreliable_252 = I.trailing_count(audit.unreliable_day, 252)[r, c]
    base["integrity_cooloff"] = breaks_recent > 0
    base["integrity_unreliable"] = (artifacts_252 > 0) | (unreliable_252 > config.max_unreliable_breaks_252)
    base["has_quote"] = np.isfinite(base.price_t)
    base["history_ok"] = base.valid_returns_63 >= config.min_valid_returns_63
    base["liquid"] = np.exp(base.log_dollar_volume_63.astype(float)) >= config.min_dollar_volume
    base["eligible"] = (base.has_quote & base.history_ok & base.liquid
                        & ~base.integrity_cooloff & ~base.integrity_unreliable)

    universe = ref.universe.set_index("symbol")
    base["sector"] = base.symbol.map(universe.sector).fillna("Unknown")
    base["industry"] = base.symbol.map(universe.industry).fillna("Unknown")

    base = F.join_financials(base, F.financial_events(ref.fundamentals))

    base = F.sector_relative(base, F.SECTOR_RELATIVE)
    numeric = [n for n, s in F.REGISTRY.items() if n != "availability_imputed"]
    base = F.cross_sectional_rank(base, numeric)

    lab = label_arrays(px, calendar, np.unique(r), config.holding_days, audit.unreliable_day)
    pos = np.searchsorted(np.unique(r), base.row.to_numpy())
    c = base.col.to_numpy()
    base["target"] = lab["target"][pos, c]
    end_idx = lab["label_end_idx"][pos, c]
    base["label_end"] = pd.NaT
    has = end_idx >= 0
    base.loc[has, "label_end"] = calendar[end_idx[has]]
    base["label_end"] = pd.to_datetime(base.label_end)
    base["label_has_unreliable_break"] = lab["label_has_unreliable_break"][pos, c]
    labeled = base.eligible & base.target.notna()
    grp = base.loc[labeled].groupby("date").target
    base["target_rank"] = np.nan
    base.loc[labeled, "target_rank"] = (grp.rank(pct=True) - 0.5).to_numpy()
    base["target_demeaned"] = np.nan
    base.loc[labeled, "target_demeaned"] = (base.loc[labeled, "target"] - grp.transform("mean")).to_numpy()
    base = base.sort_values(["date", "symbol"]).reset_index(drop=True)
    _check_panel(base)
    return base


def _check_panel(p: pd.DataFrame) -> None:
    if p.duplicated(["date", "symbol"]).any():
        raise AssertionError("Duplicate (date, symbol) rows")
    if not (p.entry_date > p.date).all():
        raise AssertionError("Entry must follow the signal date")
    done = p.label_end.notna()
    if not (p.loc[done, "label_end"] >= p.loc[done, "entry_date"] + pd.Timedelta(days=90)).all():
        raise AssertionError("A label ended before 90 calendar days")


def feature_columns(names, ranked=True):
    """Model input columns for a list of registry feature names."""
    unknown = [n for n in names if n not in F.REGISTRY]
    if unknown:
        raise KeyError(f"Unknown features {unknown}")
    return [f"{n}_csr" if ranked and n != "availability_imputed" else n for n in names]
