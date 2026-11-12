"""Month-end research panel on top of Lectures/modeling/course_data.py.

Peer features are computed only after every ticker is concatenated, and only
on historically eligible names (Lecture 7.2).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

COURSE_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
MODELING = COURSE_ROOT / "Lectures" / "modeling"
if str(MODELING) not in sys.path:
    sys.path.insert(0, str(MODELING))

from course_data import (  # noqa: E402
    NUMERIC_FEATURES,
    apply_membership,
    financial_events,
    join_financials,
    load_data,
    load_membership,
    make_labels,
    price_features,
)

# Snapshot sector is not point-in-time, so it is not a model input.
MODEL_FEATURES = list(NUMERIC_FEATURES)
RANK_CAP = 400
MIN_CLOSE = 5.0
MIN_DOLLAR_VOLUME = 1_000_000.0
BREAK_RATIO = 4.0
FACTOR_SPECS = {
    "mom_roa": (("momentum_252_21", 1), ("operating_roa", 1)),
    "mom_roa_lowvol": (("momentum_252_21", 1), ("operating_roa", 1), ("volatility_63", -1)),
    "quality_momentum": (
        ("momentum_252_21", 1),
        ("operating_roa", 1),
        ("fcf_to_assets", 1),
        ("accruals_to_assets", -1),
    ),
}


def month_end_sessions(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(calendar).unique().sort_values()
    next_month = pd.Series(idx, index=idx).shift(-1).dt.to_period("M")
    return idx[idx.to_period("M") != next_month]


def list_symbols(database: Path) -> list[str]:
    with duckdb.connect(str(database), read_only=True) as con:
        return con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchdf().symbol.tolist()


def flag_price_breaks(panel: pd.DataFrame) -> pd.DataFrame:
    """Blank eligibility on a 4x one-session adj_close jump; do not rewrite prices."""
    out = panel.sort_values(["symbol", "date"]).copy()
    prev = out.groupby("symbol").adj_close.shift()
    ratio = out.adj_close / prev
    broken = prev.gt(0) & (ratio.gt(BREAK_RATIO) | ratio.lt(1.0 / BREAK_RATIO))
    out.loc[broken.fillna(False), "eligible"] = False
    return out


def add_peer_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Same-date, same-sector peers among eligible names only."""
    out = panel.copy()
    out["sector_relative_momentum"] = np.nan
    out["roa_sector_rank"] = np.nan
    if out.empty or "eligible" not in out:
        return out
    mask = out.eligible.fillna(False)
    if not mask.any():
        return out
    work = out.loc[mask]
    grouped = work.groupby(["date", "sector"], sort=False)
    n_mom = grouped.momentum_63.transform(lambda s: int(s.notna().sum()))
    med_mom = grouped.momentum_63.transform("median")
    out.loc[mask, "sector_relative_momentum"] = np.where(
        n_mom.to_numpy() >= 3, work.momentum_63.to_numpy() - med_mom.to_numpy(), np.nan
    )
    n_roa = grouped.operating_roa.transform(lambda s: int(s.notna().sum()))
    roa_rank = grouped.operating_roa.rank(pct=True)
    out.loc[mask, "roa_sector_rank"] = np.where(n_roa.to_numpy() >= 3, roa_rank.to_numpy(), np.nan)
    return out


def add_rank_target(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["rank_target"] = np.nan
    known = out.target.notna()
    if known.any():
        out.loc[known, "rank_target"] = out.loc[known].groupby("date").target.rank(pct=True)
    return out


def in_fold_keep(target: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Training-fold mask: keep raw targets inside that fold's 1–99% range."""
    finite = target.to_numpy(dtype=float)
    if not np.isfinite(finite).any():
        return pd.Series(False, index=target.index)
    lo, hi = np.nanquantile(finite, [lower, upper])
    return target.between(lo, hi)


def entry_screen(panel: pd.DataFrame, rank_cap: int | None = RANK_CAP) -> pd.Series:
    """New-buy mask. Held lots are not filtered here."""
    ranked = panel.universe_rank.notna() if rank_cap is None else panel.universe_rank.le(rank_cap)
    dollar_volume = panel.close * panel.volume.fillna(0)
    return panel.eligible.fillna(False) & ranked & panel.close.ge(MIN_CLOSE) & dollar_volume.ge(MIN_DOLLAR_VOLUME)


def two_factor_scores(frame: pd.DataFrame) -> pd.Series:
    return factor_scores(frame, FACTOR_SPECS["mom_roa"])


def factor_scores(frame: pd.DataFrame, spec: tuple[tuple[str, int], ...]) -> pd.Series:
    """Equal-weight same-date ranks. Missing legs stay missing (not filled with 0.5)."""
    parts = []
    for column, sign in spec:
        ranked = frame.groupby("date")[column].rank(pct=True)
        parts.append(ranked if sign > 0 else 1.0 - ranked)
    return sum(parts) / len(parts)


def batched_month_end_panel(database: Path, end: str, batch_size: int = 80) -> tuple[pd.DataFrame, pd.DataFrame]:
    with duckdb.connect(str(database), read_only=True) as con:
        universe = con.execute("SELECT symbol, sector FROM universe ORDER BY symbol").fetchdf()
        benchmark = con.execute(
            "SELECT date, adj_close FROM benchmark WHERE date<=? ORDER BY date", [end]
        ).fetchdf()
    benchmark["date"] = pd.to_datetime(benchmark.date).astype("datetime64[ns]")
    membership = load_membership(database, end=end)
    calendar = pd.DatetimeIndex(benchmark.date)
    ends = set(month_end_sessions(calendar))
    symbols = list_symbols(database)
    parts = []
    t0 = time.time()
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        prices, fundamentals, _, _ = load_data(database, end=end, symbols=batch)
        if prices.empty:
            continue
        panel = price_features(prices, calendar)
        panel = flag_price_breaks(panel)
        panel = join_financials(panel, financial_events(fundamentals), universe)
        panel = apply_membership(panel, membership, calendar)
        panel = panel.loc[panel.date.isin(ends)].copy()
        panel = make_labels(panel, prices, calendar, holding_days=90)
        parts.append(panel)
        done = min(i + batch_size, len(symbols))
        print(f"  features {done}/{len(symbols)} symbols ({time.time() - t0:.0f}s)", flush=True)
    panel = pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    panel = add_peer_features(panel)
    panel = add_rank_target(panel)
    return panel, benchmark
