"""HW4: multi-strategy equity signal system, parameter research, and a
long-only 90-day backtest against ^SP500TR.

Run from the repository root (after ``uv sync --locked``):

    .venv/bin/python Homeworks/HW4/Philip_Szkudlarski/hw4.py --check   # accounting/timing self-tests (no data)
    .venv/bin/python Homeworks/HW4/Philip_Szkudlarski/hw4.py           # full research + final test

The full run needs ``Lectures/modeling/data/equities_top2000.duckdb`` (bCourses).
Outputs: ``cumulative_returns.png`` next to this file and CSV tables in
``outputs/`` (git-ignored).

Overview
--------
Universe   Each quarter, the 500 largest stocks by the course's dated market-cap
           proxy, after removing names whose cap proxy is contradicted by their
           trading activity (see ``load_universe``).
Strategies Six independent signal engines, each with a parameter grid:
             1. RSI oversold + volatility filter
             2. Pairs: same-industry, similar-beta, correlated stocks whose
                price-ratio z-score says one has fallen behind (+ EPS filter)
             3. Free-cash-flow growth without share dilution
             4. Revenue growth vs. industry peers (fundamental momentum)
             5. RSI reversal (momentum turning up from oversold)
             6. 12-1 month price momentum (added during development)
Research   For every strategy/parameter/waiting-period combination we measure
           the average 90-day return of flagged stocks in excess of the same-day
           universe average. Parameters are chosen on 2010-2012 ("train") and
           judged on 2013-2015 ("validation"). Holding horizons and dynamic
           exits are studied too, but only as research (HW4 fixes 90 days).
Model      A gradient-boosting model on ranked indicators + strategy signals
           (walk-forward refits) competes with the rule-based composites.
Portfolio  K equal slots (budget = NAV/K), new names bought at the next close,
           at most K/20 new names a day, each lot held exactly 90 calendar days
           (HW4 rule); no leverage; optional sector cap / inverse-vol sizing.
Selection  ~370 candidates (single strategies, all combinations, the model,
           K in {20, 40}) are backtested on 2013-2015. Among those within 0.10
           Sharpe of the best, the simplest wins; then risk controls are chosen
           the same way. A sector/size regime overlay (lean into groups whose
           relative strength is rising) is adopted only if it adds > 0.10
           validation Sharpe without hurting 2010-2012 (it did not). A
           volatility component (low-vol / low-beta tilts, vol filter, vol
           targeting) takes the lowest-volatility setting that keeps both
           Sharpe ratios within 0.10. The test period (Apr 2016 - Sep 2026) is
           then run with the frozen configuration.
Stage 11   A 90-day review (at maturity, buy the same stock straight back as a
           new 90-day lot if it still ranks in the top m*K, else let it go)
           and a growth/stable rotation (fast revenue growers in calm
           markets, winners that beat the index by > 20-40% handed over to
           stable names, growth names bought back below their 200-day
           average) were tested with the same adoption rule; neither was
           adopted. The rotation's test curve is shown for information.
Stage 12   A growth tilt: while the market is calm, or after defensive
           groups (pharma + utilities, or healthcare/utilities/staples) led
           the universe, new purchases of fast revenue growers are 1.5-2x
           size until the growth basket earned +3-8% over the index; then
           sizes return to normal. Same adoption rule; not adopted.
Stage 13   Dispersion (average stock volatility minus index volatility,
           ranked against two years) splits days into regimes; each
           strategy's excess return by regime is measured, and a combined
           signal leans into the strategies that worked in today's regime
           (weights from finished outcomes only). Not adopted.
Stage 14   FINAL SELECTION (return-based, no volatility optimization): a 7th
           signal (sector revenue leaders), every subset of the 7 signals x
           6 weightings (equal, t-weighted, strategy momentum 126/252 days,
           best-only, dispersion), then the 10 best x every combination of
           overlays (sector regime, mega-cap leadership tilt, growth
           overweight, 90-day review). Score = worse of 2010-12 and
           2013-15 excess CAGR over ^SP500TR; within 1%/yr, fewest parts.
Prices     Unadjusted stock splits in the vendor prices (NFLX -90%, LVLT
           +1,390%...) are repaired at load (repair_price_errors).
Data       One company, one position: duplicate listings (renamed tickers
           stored twice) and share classes are detected and only one listing
           is buyable per day; see data_quality_report for every issue.

Assumptions: 5 bp cost per side on traded notional, cash earns 0%, adjusted
prices (dividends included, never added again), missing quotes are never
filled (orders skipped, exits wait for the next quote, marks carried and
flagged). Fundamentals are used from filing date + 1 day.

Reused code / AI use: the trading ideas and strategy design are mine; most of
the code was written by an AI assistant (Claude Code) at my direction, which
also proposed parts of the methodology (12-1 momentum, selection rules,
trailing-performance weights). The data-cleaning approach (repair one-day price
errors by rescaling, quarantine what can't be fixed, flag every repair) builds
on my HW2 data cleaning (01_data_cleaning.ipynb, branch hw_2_Philip_Szkudlarski). The HW4 data conventions
(filing-date availability, next-session membership, 90-day labels) follow
Lectures/modeling/README.md and course_data.py. I reviewed the design and results.
"""

from __future__ import annotations

import argparse
import itertools
import math
import tempfile
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HW_DIR = Path(__file__).resolve().parent
REPO_ROOT = HW_DIR.parents[2]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_DB = REPO_ROOT / "Lectures/modeling/data/equities_top2000.duckdb"
OUTPUT_DIR = HW_DIR / "outputs"          # git-ignored
CHART_PATH = HW_DIR / "cumulative_returns.png"

# Universe
TOP_N = 500
MAX_DV_RANK = 1000          # 63-day dollar-volume rank among the quarter's top-2000
MIN_DV_SESSIONS = 40        # sessions with prices in the ~3-month volume window
MAX_CAP_TO_ADV = 3000       # cap proxy / average daily dollar volume (bad-input screen)
JUMP_UP, JUMP_DOWN = 3.0, -0.75   # one-day moves treated as suspected data errors
REPAIR_DOWN, REPAIR_UP = -0.70, 1.50   # one-day moves repaired as unadjusted splits / bad prints
ROUND_TRIP_MOVE, ROUND_TRIP_DAYS, ROUND_TRIP_TOL = np.log(1.5), 10, 0.10   # +x then -x glitches
LABEL_CLIP = (-0.95, 3.0)   # research/training labels only (never the backtest)

# Periods
DATA_START = "2008-01-01"   # warm-up for 252-day lookbacks
TRAIN = ("2010-01-01", "2012-12-31")
VALID = ("2013-01-01", "2015-12-31")
TEST_START = "2016-04-01"
TEST_END = "2026-09-30"
WALK_FORWARD_YEARS = 3      # trailing window for walk-forward parameter choice

# Trading rules
HOLDING_DAYS = 90
INITIAL_CAPITAL = 1_000_000.0
COST_BPS = 5.0
MAX_WEIGHT = 0.10
MIN_ORDER_FRACTION = 0.002
STALE_LIMIT = 5
ANNUAL_RF = 0.02

# Research
WAITS = (0, 1, 2, 3, 5)                       # sessions between signal and entry decision
RESEARCH_HORIZONS = (1, 3, 5, 10, 20, 30, 60, 90)   # trading days (research only)
SAMPLE_EVERY = 5                              # thin overlapping labels in event studies
MIN_SIGNAL_DATES = 40                         # sampled dates with >=1 signal to be eligible
MIN_SIGNALS = 150                             # total flagged stock-dates to be eligible
K_GRID = (20, 40)
SIMPLICITY_TOL = 0.10                         # Sharpe tolerance for preferring simpler candidates


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. Universe
# ---------------------------------------------------------------------------
def connect(database):
    database = Path(database)
    if not database.is_file():
        raise FileNotFoundError(
            f"Database not found at {database}. Download it from bCourses into "
            "Lectures/modeling/data/ (see Lectures/modeling/README.md).")
    return duckdb.connect(str(database), read_only=True)


def load_universe(database, top_n=TOP_N, max_dv_rank=MAX_DV_RANK,
                  min_sessions=MIN_DV_SESSIONS, max_cap_to_adv=MAX_CAP_TO_ADV):
    """Quarterly S&P 500 proxy: top ``top_n`` by cap proxy after a sanity screen.

    The course's ``market_cap_proxy`` (selection-date close x latest filed
    shares) contains impossible values in older quarters, e.g. Timber
    Pharmaceuticals ranked #1 at $28.7T in 2016-Q2, because split-adjusted
    prices are mixed with share counts on another basis. Using only data known
    at the selection date, we skip a candidate unless
      * its average daily dollar volume (close x volume, same basis, over the
        ~63 sessions before selection) ranks <= 1000 of the quarter's 2000,
      * it traded on >= 40 of those sessions, and
      * cap proxy / dollar volume <= 3000 (real large caps are ~100-600).
    Then we keep the first ``top_n`` survivors in cap-proxy order.
    """
    with connect(database) as con:
        m = con.execute("""
            WITH m AS (SELECT quarter_start, quarter_end, selection_date, symbol,
                              universe_rank, market_cap_proxy FROM universe_membership),
            dv AS (SELECT m.quarter_start, m.symbol, avg(p.close * p.volume) AS adv, count(*) AS n
                   FROM m JOIN prices p ON p.symbol = m.symbol
                        AND p.date <= m.selection_date
                        AND p.date > m.selection_date - INTERVAL 92 DAY
                   GROUP BY ALL)
            SELECT m.*, dv.adv, dv.n FROM m LEFT JOIN dv USING (quarter_start, symbol)
        """).fetchdf()
    for column in ["quarter_start", "quarter_end", "selection_date"]:
        m[column] = pd.to_datetime(m[column]).astype("datetime64[ns]")
    m["dv_rank"] = m.groupby("quarter_start").adv.rank(ascending=False, method="first")
    m["cap_to_adv"] = m.market_cap_proxy / m.adv
    passes = (m.dv_rank.le(max_dv_rank) & m.n.ge(min_sessions)
              & m.cap_to_adv.le(max_cap_to_adv))
    kept = (m.loc[passes].sort_values(["quarter_start", "universe_rank"])
            .groupby("quarter_start").head(top_n).reset_index(drop=True))
    kept["screened_rank"] = kept.groupby("quarter_start").cumcount() + 1
    return kept, m


def report_universe(kept, all_members):
    counts = kept.groupby("quarter_start").size()
    top = all_members.loc[all_members.universe_rank.le(TOP_N)]
    removed = top.loc[~top.set_index(["quarter_start", "symbol"]).index.isin(
        kept.set_index(["quarter_start", "symbol"]).index)]
    log(f"Universe: {len(counts)} quarters, {counts.min()}-{counts.max()} names each, "
        f"{kept.symbol.nunique()} distinct stocks; screen removed {len(removed):,} "
        f"raw top-{TOP_N} rows ({len(removed) / len(top):.1%})")
    worst = removed.sort_values("universe_rank").head(8)
    print("  Examples removed by the screen:",
          ", ".join(f"{r.symbol} {r.quarter_start.year}-Q{(r.quarter_start.month - 1) // 3 + 1} "
                    f"rank {r.universe_rank} cap ${r.market_cap_proxy / 1e9:,.0f}B"
                    for r in worst.itertuples()))


# ---------------------------------------------------------------------------
# 2. Market data (wide date x symbol matrices)
# ---------------------------------------------------------------------------
@dataclass
class MarketData:
    calendar: pd.DatetimeIndex
    px: pd.DataFrame             # adjusted close (total-return units), NaN = no quote
    close: pd.DataFrame          # raw close (for FCF yield with reported shares)
    bench: pd.Series             # ^SP500TR adjusted close
    info: pd.DataFrame           # symbol -> name, sector, industry
    fundamentals: pd.DataFrame   # quarterly rows
    membership: pd.DataFrame     # screened quarterly membership
    repairs: pd.DataFrame = None  # price errors repaired at load (see repair_price_errors)


def load_market_data(database, membership, start=DATA_START, end=TEST_END):
    symbols = sorted(membership.symbol.unique())
    with connect(database) as con:
        cal = pd.DatetimeIndex(pd.to_datetime(con.execute(
            "SELECT date FROM calendar WHERE date >= ? AND date <= ? ORDER BY date", [start, end]).fetchdf().date))
        prices = con.execute("""SELECT symbol, date, close, adj_close FROM prices
            WHERE date >= ? AND date <= ? AND symbol IN (SELECT unnest(?))""", [start, end, symbols]).fetchdf()
        bench = con.execute("SELECT date, adj_close FROM benchmark WHERE date >= ? AND date <= ? ORDER BY date",
                            [start, end]).fetchdf()
        info = con.execute("SELECT symbol, name, sector, industry FROM universe WHERE symbol IN (SELECT unnest(?))",
                           [symbols]).fetchdf()
        fund = con.execute("""SELECT symbol, fiscal_date, filing_date, revenue, operating_income, net_income, eps,
                operating_cash_flow, capex, free_cash_flow, shares_outstanding, total_assets
            FROM fundamentals WHERE period = 'quarter' AND symbol IN (SELECT unnest(?))
              AND fiscal_date >= DATE '2005-01-01' AND fiscal_date <= ?""", [symbols, end]).fetchdf()
    prices["date"] = pd.to_datetime(prices.date)
    bad = ~np.isfinite(prices.adj_close) | prices.adj_close.le(0)
    off = ~prices.date.isin(cal)
    if bad.any() or off.any():
        log(f"  DATA WARNING: quarantined {int(bad.sum())} invalid adjusted prices and "
            f"{int((off & ~bad).sum())} off-calendar rows (never filled)")
    prices = prices.loc[~bad & ~off]
    px = prices.pivot(index="date", columns="symbol", values="adj_close").reindex(index=cal, columns=symbols)
    close = prices.pivot(index="date", columns="symbol", values="close").reindex(index=cal, columns=symbols)
    px, close, repairs = repair_price_errors(px, close)
    if len(repairs):
        log(f"  DATA WARNING: repaired {len(repairs)} one-day price errors (unadjusted splits / glitches) "
            f"in {repairs.symbol.nunique()} stocks; see outputs/data_quality.csv")
    bench = bench.assign(date=pd.to_datetime(bench.date)).set_index("date").adj_close.reindex(cal)
    assert bench.notna().all(), "Benchmark missing on a market session"
    for c in ["fiscal_date", "filing_date"]:
        fund[c] = pd.to_datetime(fund[c])
    info = info.set_index("symbol").reindex(symbols)
    info["industry"] = info.industry.fillna("Unknown")
    info["sector"] = info.sector.fillna("Unknown")
    return MarketData(cal, px, close, bench, info, fund, membership, repairs)


def repair_price_errors(px, close):
    """Repair one-day price level shifts that are data errors, not returns.

    The vendor's "adjusted" close (and the raw close) contain unadjusted
    stock splits: e.g. NFLX -90% (10:1), ORLY -93% (15:1), BKNG -96% (25:1),
    KLAC -89%, NOW -80%, LVLT +1,390% (1:15 reverse split) -- while share
    counts and most famous splits (AAPL, NVDA, AMZN) are adjusted correctly.
    A held position would book these as real -90% losses or +1,000% gains.

    Rule (fixed, documented in the README):
      * level shift: a one-day move (vs the last quote) below REPAIR_DOWN or
        above REPAIR_UP;
      * round trip: a move beyond +-50% in log terms that is reversed within
        ROUND_TRIP_DAYS quotes to within ROUND_TRIP_TOL (e.g. HON +99% then -51%).
    The move is set to 0 by rescaling all EARLIER prices (latest prices are
    kept). Cost: a few real crashes or spikes (NVAX -85% in 2016, AMC +301%
    in 2021) are neutralized too, a small optimistic bias that is disclosed.
    Returns (px, close, repairs table).
    """
    last = px.ffill().shift(1)
    r = (px / last - 1).where(px.notna())
    lr = np.log1p(r)
    err = ((r < REPAIR_DOWN) | (r > REPAIR_UP)).to_numpy()
    kind = np.where(err, "level shift", "")
    big = (lr.abs() > ROUND_TRIP_MOVE).to_numpy()
    L = lr.to_numpy()
    for t, j in zip(*np.nonzero(big & ~err)):
        later = np.flatnonzero(np.isfinite(L[t + 1:, j]))[:ROUND_TRIP_DAYS] + t + 1
        back = [u for u in later if big[u, j] and abs(L[t, j] + L[u, j]) < ROUND_TRIP_TOL]
        if back:
            err[t, j] = err[back[0], j] = True
            kind[t, j] = kind[back[0], j] = "round trip"
    G = np.where(err, 1 + r.to_numpy(), 1.0)
    P = np.cumprod(G[::-1], axis=0)[::-1]            # product of G[t:]
    F = np.vstack([P[1:], np.ones((1, P.shape[1]))])  # product of G[t+1:]
    t_idx, j_idx = np.nonzero(err)
    repairs = pd.DataFrame({"date": px.index[t_idx], "symbol": px.columns[j_idx],
                            "move": r.to_numpy()[t_idx, j_idx], "kind": kind[t_idx, j_idx]})
    return (px * F), (close * F), repairs


def _company_key(name):
    """Normalized company name: 'Fox Corporation' == 'Fox Corp', 'Dayforce Inc.' == 'Dayforce Inc'."""
    import re
    n = re.sub(r"[^a-z0-9 ]", " ", str(name).lower())
    drop = {"inc", "corp", "corporation", "co", "company", "ltd", "plc", "lp", "l", "p", "the",
            "holdings", "holding", "class", "a", "b", "c", "group"}
    return " ".join(w for w in n.split() if w not in drop)


def duplicate_listings(ret, names=None, min_overlap=60, min_corr=0.99):
    """Pairs of symbols that are the same company.

    * "same series": daily returns correlate >= 0.99 over their overlap -- one
      security stored under an old and a new ticker (FLT/CPAY, GAP/GPS,
      MMC/MRSH, ANTM/ELV, ...), adjusted slightly differently;
    * "same name": share classes or renamed listings with the same company
      name (BRK-A/BRK-B, FOX/FOXA, LEN/LEN-B, BALL/BLL, ...).
    """
    R = ret.to_numpy(dtype=float)
    ok = np.isfinite(R)
    Z = np.where(ok, R, 0.0)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    C = (Z.T @ Z) / len(Z)
    found = {}
    for i, j in np.argwhere(np.triu(C > 0.3, 1)):
        both = ok[:, i] & ok[:, j]
        if both.sum() >= min_overlap and np.corrcoef(R[both, i], R[both, j])[0, 1] >= min_corr:
            found[tuple(sorted((ret.columns[i], ret.columns[j])))] = "same series"
    if names is not None:
        keys = names.reindex(ret.columns).map(_company_key)
        for _, group in keys[keys.str.len() > 0].groupby(keys[keys.str.len() > 0]):
            for a, b in itertools.combinations(sorted(group.index), 2):
                found.setdefault((a, b), "same name")
    return [(a, b, why) for (a, b), why in sorted(found.items())]


def duplicate_block(ret, pairs, base, window=63, min_corr=0.99):
    """Bool matrix: True where a symbol must not be bought because another
    listing of the same company (alphabetically earlier) is buyable that day.
    "Same series" pairs count only while their trailing-63-day correlation is
    >= 0.99 (known at t); "same name" pairs always count."""
    block = pd.DataFrame(False, index=ret.index, columns=ret.columns)
    for pair in pairs:
        a, b, why = pair if len(pair) == 3 else (*pair, "same series")
        first, second = sorted((a, b))
        if why == "same series":
            corr = ret[first].rolling(window, min_periods=20).corr(ret[second])
            active = corr.fillna(0).ge(min_corr)
        else:
            active = pd.Series(True, index=ret.index)
        block[second] |= active & base[first]
    return block


def eligibility(md):
    """Bool matrix: stock may be *bought* after the close of row t.

    Membership is looked up by the quarter of the next session (the entry
    date), as in the course helper; the stock also needs a quote today and 63
    complete trailing returns.
    """
    cal = md.calendar
    next_session = pd.Series(cal[1:].append(pd.DatetimeIndex([pd.NaT])), index=cal)
    quarter = next_session.dt.to_period("Q").dt.start_time
    mem = (md.membership.assign(flag=1.0)
           .pivot_table(index="quarter_start", columns="symbol", values="flag", aggfunc="max")
           .reindex(columns=md.px.columns))
    elig = mem.reindex(quarter.to_numpy()).fillna(0.0).to_numpy() > 0
    ret = md.px.pct_change(fill_method=None)
    full_history = ret.rolling(63, min_periods=63).count().ge(63).to_numpy()
    # Data-quality screen (known at t): a one-day move above +300% or below
    # -75% in the last 252 sessions usually means a mishandled corporate
    # action in the adjusted series (e.g. AAM 2010: $0.44 -> $25,500).
    jump = ((ret > JUMP_UP) | (ret < JUMP_DOWN)).astype(float)
    if getattr(md, "repairs", None) is not None and len(md.repairs):
        # repaired price errors count as suspect data too: no new buys for a year
        rr = md.repairs.loc[md.repairs.symbol.isin(jump.columns)]
        jump.values[cal.get_indexer(rr.date), jump.columns.get_indexer(rr.symbol)] = 1.0
    clean = jump.rolling(252, min_periods=1).max().fillna(0).to_numpy() == 0
    base = pd.DataFrame(elig & md.px.notna().to_numpy() & full_history & clean,
                        index=cal, columns=md.px.columns)
    # One company, one position: duplicate listings (renamed tickers stored
    # twice) and share classes -- buy only one listing on any day.
    md.duplicate_pairs = duplicate_listings(ret.loc[ret.index >= TRAIN[0]], md.info.name)
    dup = duplicate_block(ret, md.duplicate_pairs, base).to_numpy()
    return base & ~dup


# ---------------------------------------------------------------------------
# 3. Technical analysis engine
# ---------------------------------------------------------------------------
def calculate_rsi(px, lookback):
    """Wilder RSI on adjusted closes; NaN until ``lookback`` changes exist."""
    delta = px.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / lookback, adjust=False, min_periods=lookback).mean()
    down = (-delta).clip(lower=0).ewm(alpha=1 / lookback, adjust=False, min_periods=lookback).mean()
    total = up + down
    return (100 * up / total.where(total > 0)).astype("float32")


def calculate_realized_volatility(ret, lookback):
    return (ret.rolling(lookback, min_periods=lookback).std() * np.sqrt(252)).astype("float32")


def calculate_volatility_moving_average(vol, window):
    return vol.rolling(window, min_periods=window).mean().astype("float32")


def calculate_beta(ret, market_ret, lookback):
    """Rolling OLS beta of each stock's daily return on the benchmark's."""
    m = market_ret
    mean_xy = ret.mul(m, axis=0).rolling(lookback, min_periods=lookback).mean()
    mean_x = ret.rolling(lookback, min_periods=lookback).mean()
    mean_m = m.rolling(lookback, min_periods=lookback).mean()
    var_m = (m * m).rolling(lookback, min_periods=lookback).mean() - mean_m ** 2
    return (mean_xy.sub(mean_x.mul(mean_m, axis=0))).div(var_m, axis=0).astype("float32")


class Features:
    """Lazily computed, cached wide indicator matrices."""

    def __init__(self, md):
        self.md = md
        self.ret = md.px.pct_change(fill_method=None)
        self.mret = md.bench.pct_change()
        self._cache = {}

    def get(self, name, *args):
        key = (name,) + args
        if key not in self._cache:
            self._cache[key] = getattr(self, "_" + name)(*args)
        return self._cache[key]

    def _rsi(self, n):
        return calculate_rsi(self.md.px, n)

    def _vol(self, w):
        return calculate_realized_volatility(self.ret, w)

    def _volma(self, w, m):
        return calculate_volatility_moving_average(self.get("vol", w), m)

    def _beta(self, L):
        return calculate_beta(self.ret, self.mret, L)

    def _momentum(self, n, skip=0):
        px = self.md.px
        return (px.shift(skip) / px.shift(n) - 1).astype("float32")

    def _drawdown(self, n):
        px = self.md.px
        return (px / px.rolling(n, min_periods=n).max() - 1).astype("float32")


# ---------------------------------------------------------------------------
# 4. Fundamental analysis engine (point-in-time)
# ---------------------------------------------------------------------------
FUNDAMENTAL_COLUMNS = ["rev_yoy", "rev_yoy_prev", "rev_accel", "rev_avg3", "rev_avg4", "op_margin",
                       "eps_chg", "eps_ok", "fcf_growth_ttm", "fcf_turnaround_ttm", "fcf_growth_q",
                       "fcf_turnaround_q", "share_growth", "fcfps_growth", "ttm_fcf", "shares"]


def _growth(cur, prev, near_zero=None):
    """(cur - prev)/|prev| when prev is meaningfully positive; else NaN.

    Negative or ~zero bases make percentage changes meaningless; turnarounds
    (prev <= 0 < cur) are classified separately by the caller.
    """
    ok = prev > 0
    if near_zero is not None:
        ok &= prev.abs() > near_zero
    return ((cur - prev) / prev.abs()).where(ok)


def fundamental_events(f):
    """Quarter-level metrics with the date they become usable.

    * Only filings dated after fiscal period end are trusted (course rule).
    * Usable from filing date + 1 calendar day; a metric that mixes several
      quarters is usable only once the latest of them is (cummax of dates).
    * YoY comparisons require the lagged row to be ~1 year earlier, TTM sums
      require 4 consecutive quarters.
    * An old fiscal period filed late never replaces a newer known one.
    """
    f = f.loc[f.filing_date.gt(f.fiscal_date)].sort_values(["symbol", "fiscal_date", "filing_date"])
    f = f.drop_duplicates(["symbol", "fiscal_date"], keep="first").copy()
    g = f.groupby("symbol", sort=False)
    lag = lambda col, k: g[col].shift(k)
    days = lambda k: (f.fiscal_date - lag("fiscal_date", k)).dt.days
    yoy_ok = days(4).between(340, 390)
    ttm_ok = days(3).between(250, 300)
    prev_yoy_ok = (lag("fiscal_date", 1).sub(lag("fiscal_date", 5)).dt.days).between(340, 390)

    f["fcf"] = f.free_cash_flow.fillna(f.operating_cash_flow + f.capex)
    f["ttm_fcf"] = (f.fcf + lag("fcf", 1) + lag("fcf", 2) + lag("fcf", 3)).where(ttm_ok)
    f["ttm_fcf_prev"] = g["ttm_fcf"].shift(4).where(yoy_ok)
    near_zero = 0.005 * f.total_assets.abs()

    f["rev_yoy"] = _growth(f.revenue, lag("revenue", 4)).where(yoy_ok & f.revenue.gt(0))
    f["rev_yoy_prev"] = g["rev_yoy"].shift(1).where(prev_yoy_ok)
    f["rev_accel"] = f.rev_yoy - f.rev_yoy_prev
    f["rev_avg3"] = (f.rev_yoy + f.rev_yoy_prev + g["rev_yoy"].shift(2)) / 3
    f["rev_avg4"] = (f.rev_yoy + f.rev_yoy_prev + g["rev_yoy"].shift(2) + g["rev_yoy"].shift(3)) / 4
    f["op_margin"] = (f.operating_income / f.revenue.where(f.revenue > 0))

    eps_prev = lag("eps", 4).where(yoy_ok)
    f["eps_chg"] = _growth(f.eps, eps_prev)
    # EPS filter: <=5% deterioration vs. the same quarter last year; with a
    # zero/negative base we only require that EPS did not get worse.
    f["eps_ok"] = ((eps_prev > 0) & (f.eps_chg >= -0.05)) | ((eps_prev <= 0) & (f.eps >= eps_prev))

    f["fcf_growth_ttm"] = _growth(f.ttm_fcf, f.ttm_fcf_prev, near_zero)
    f["fcf_turnaround_ttm"] = (f.ttm_fcf_prev <= near_zero) & (f.ttm_fcf > near_zero)
    fcf_q_prev = lag("fcf", 4).where(yoy_ok)
    f["fcf_growth_q"] = _growth(f.fcf, fcf_q_prev, near_zero)
    f["fcf_turnaround_q"] = (fcf_q_prev <= near_zero) & (f.fcf > near_zero)

    shares_prev = lag("shares_outstanding", 4).where(yoy_ok)
    f["shares"] = f.shares_outstanding.where(f.shares_outstanding > 0)
    f["share_growth"] = (f.shares / shares_prev.where(shares_prev > 0) - 1)
    fcfps, fcfps_prev = f.ttm_fcf / f.shares, f.ttm_fcf_prev / shares_prev.where(shares_prev > 0)
    f["fcfps_growth"] = _growth(fcfps, fcfps_prev).where(~((fcfps_prev <= 0) & (fcfps > 0)), np.inf)

    for c in ["eps_ok", "fcf_turnaround_ttm", "fcf_turnaround_q"]:
        f[c] = f[c].astype(float)
    f["available"] = f.filing_date + pd.Timedelta(days=1)
    f["available"] = f.groupby("symbol").available.cummax()
    f = f.sort_values(["symbol", "available", "fiscal_date"])
    f = f.loc[f.fiscal_date.eq(f.groupby("symbol").fiscal_date.cummax())]
    return f.drop_duplicates(["symbol", "available"], keep="last")


def fundamentals_wide(events, md, max_age_days=270):
    """As-of join of quarterly metrics onto the daily calendar (wide).

    A value appears from the first session on/after its availability date and
    is blanked once the fiscal period is older than ``max_age_days``.
    """
    cal = md.calendar
    ev = events.loc[events.symbol.isin(md.px.columns)].copy()
    ev["row"] = cal.searchsorted(ev.available)
    ev = ev.loc[ev.row < len(cal)].drop_duplicates(["symbol", "row"], keep="last")
    out = {}

    def spread(col):
        w = ev.pivot(index="row", columns="symbol", values=col).reindex(index=range(len(cal)),
                                                                         columns=md.px.columns)
        return w.ffill()

    fiscal_days = ev.assign(fd=(ev.fiscal_date - pd.Timestamp("1970-01-01")).dt.days.astype(float))
    fiscal = (fiscal_days.pivot(index="row", columns="symbol", values="fd")
              .reindex(index=range(len(cal)), columns=md.px.columns).ffill().to_numpy())
    today = ((cal - pd.Timestamp("1970-01-01")).days.to_numpy(float))[:, None]
    with np.errstate(invalid="ignore"):
        fresh = (today - fiscal) <= max_age_days
    for col in FUNDAMENTAL_COLUMNS:
        w = spread(col).astype(float).to_numpy()
        w[~fresh] = np.nan
        out[col] = pd.DataFrame(w.astype("float32"), index=cal, columns=md.px.columns)
    # Daily FCF yield = TTM FCF / (raw close x latest reported shares).
    cap = out["shares"].to_numpy() * md.close.to_numpy()
    fy = out["ttm_fcf"].to_numpy() / np.where(cap > 0, cap, np.nan)
    fy[np.abs(fy) > 1] = np.nan         # implausible (unit/split mismatch) -> unknown
    out["fcf_yield"] = pd.DataFrame(fy.astype("float32"), index=cal, columns=md.px.columns)
    return out


# ---------------------------------------------------------------------------
# 5. Labels: 90-calendar-day forward returns, point-in-time completion dates
# ---------------------------------------------------------------------------
class Labels:
    """Forward returns for a signal at the close of row t.

    With waiting period w the entry is the close of session t+1+w; the exit is
    the first quoted close on/after entry + 90 calendar days (at most 5
    sessions late, otherwise unknown). ``end_row[t]`` is a conservative row by
    which that outcome is known; training only uses rows whose end_row is
    before the fit date.
    """

    def __init__(self, md, elig):
        cal, px = md.calendar, md.px.to_numpy()
        T = len(cal)
        self.T = T
        xpos = cal.searchsorted(cal + pd.Timedelta(days=HOLDING_DAYS))
        px_bf = md.px.bfill(limit=5).to_numpy()
        R = np.full(px.shape, np.nan)
        valid = xpos < T
        R[valid] = px_bf[xpos[valid]] / px[valid] - 1         # entry at row e (index), exit xpos[e]
        b = md.bench.to_numpy()
        Rb = np.full(T, np.nan)
        Rb[valid] = b[xpos[valid]] / b[valid] - 1
        self.R_entry, self.Rb_entry, self.xpos = R.astype("float32"), Rb, xpos
        self.elig = elig.to_numpy()
        self._cache = {}

    def target(self, wait=0):
        """(raw, excess-vs-universe) matrices aligned to the signal row t."""
        if wait not in self._cache:
            s = 1 + wait
            raw = np.full_like(self.R_entry, np.nan)
            raw[:-s] = np.clip(self.R_entry[s:], *LABEL_CLIP)
            masked = np.where(self.elig, raw, np.nan)
            with np.errstate(all="ignore"):
                mean = np.nanmean(masked, axis=1, keepdims=True)
            self._cache[wait] = (raw, (raw - mean).astype("float32"))
        return self._cache[wait]

    def end_row(self, wait=0):
        """Conservative row index at which the label of row t is complete."""
        s = 1 + wait
        e = np.arange(self.T) + s
        out = np.full(self.T, np.iinfo(np.int64).max)
        ok = e < self.T
        x = self.xpos[e[ok]] + 5
        out[ok] = np.where(self.xpos[e[ok]] < self.T, x, np.iinfo(np.int64).max)
        return out


# ---------------------------------------------------------------------------
# 6. Strategies (signal generation engine)
# ---------------------------------------------------------------------------
# Each strategy returns a wide float32 "strength" matrix: NaN = no BUY signal,
# a positive number = BUY signal (larger = stronger). Signals only exist where
# the stock is in the dated universe (``elig``).

def _finish(strength, elig):
    s = strength.to_numpy(dtype="float32", copy=True) if isinstance(strength, pd.DataFrame) else strength
    s[~elig.to_numpy()] = np.nan
    s[~np.isfinite(s)] = np.nan
    return s


# --- Strategy 1: RSI oversold with a volatility filter ----------------------
RSI_GRID = [dict(n=n, thr=t, vol_w=w, vol_ma=m)
            for n in (7, 10, 14, 21) for t in (20, 25, 30, 35) for w in (10, 21) for m in (10, 20, 30, 60)]


def generate_rsi_signals(feat, elig, n, thr, vol_w, vol_ma):
    """BUY when RSI(n) <= thr and realized vol(vol_w) < its vol_ma-day average."""
    rsi = feat.get("rsi", n)
    vol, vma = feat.get("vol", vol_w), feat.get("volma", vol_w, vol_ma)
    flag = (rsi <= thr) & (vol < vma)
    return _finish((thr - rsi + 1).where(flag), elig)


# --- Strategy 5: RSI reversal ------------------------------------------------
REVERSAL_GRID = [dict(n=n, k=k, kind=kind)
                 for n in (7, 10, 14, 21) for k in (1, 2, 3) for kind in ("early", "cross25", "cross30")]


def generate_rsi_buy_signals(feat, elig, n, k, kind):
    """Momentum turning up from oversold.

    early   : RSI rose k sessions in a row and was <= 30 within the last 10 sessions
    cross25 : RSI crosses above 25 (was below within 5 sessions) after rising k sessions
    cross30 : same with 30 (momentum confirmation)
    """
    rsi = feat.get("rsi", n)
    rising = pd.DataFrame(True, index=rsi.index, columns=rsi.columns)
    for j in range(k):
        rising &= rsi.shift(j) > rsi.shift(j + 1)
    if kind == "early":
        recent_low = rsi.rolling(10, min_periods=1).min()
        flag = rising & (recent_low <= 30) & (rsi < 50)
    else:
        level = 25 if kind == "cross25" else 30
        recent_low = rsi.shift(1).rolling(5, min_periods=1).min()
        flag = rising & (rsi > level) & (recent_low < level)
    return _finish((rsi - recent_low).where(flag), elig)


# --- Strategy 6 (added): 12-1 month price momentum -------------------------
MOMENTUM_GRID = [dict(lookback=L, skip=s, top=q) for L in (126, 252) for s in (0, 21) for q in (0.1, 0.2, 0.3)]


def generate_momentum_signals(feat, elig, lookback, skip, top):
    """BUY stocks in the top ``top`` fraction of the universe by trailing
    return from t-lookback to t-skip (skip=21 drops the last month, which
    tends to reverse). Added because momentum was the most stable indicator
    across train and validation in the model diagnostics."""
    mom = feat.get("momentum", lookback, skip)
    pct = _pct_rank(mom, elig)
    return _finish(pct.where(pct >= 1 - top), elig)


def check_rsi_overbought(feat, n=14, threshold=90):
    """SELL-signal indicator (research only: HW4 requires 90-day holds)."""
    return feat.get("rsi", n) >= threshold


# --- Strategy 3: FCF growth without dilution ---------------------------------
FCF_GRID = [dict(basis=b, g_min=g, dil_max=d, fcfps_min=p)
            for b in ("ttm", "q") for g in (0.1, 0.2, 0.3, 0.4) for d in (0.0, 0.01, 0.02, 0.03) for p in (0.0, 0.1)]


def _pct_rank(w, elig):
    """Same-day percentile rank among eligible names (0..1)."""
    return w.where(elig).rank(axis=1, pct=True)


def generate_fcf_signals(fund, elig, basis, g_min, dil_max, fcfps_min):
    """BUY when FCF grew >= g_min YoY (or turned positive), shares grew <=
    dil_max, and FCF per share grew > fcfps_min. Strength = average same-day
    percentile of FCF growth, FCF/share growth, share reduction and FCF yield."""
    growth = fund[f"fcf_growth_{basis}"]
    turnaround = fund[f"fcf_turnaround_{basis}"] > 0
    flag = (((growth >= g_min) | turnaround) & (fund["share_growth"] <= dil_max)
            & (fund["fcfps_growth"] > fcfps_min))
    score = (_pct_rank(growth.fillna(growth.max().max()).where(growth.notna() | turnaround), elig).fillna(0.5)
             + _pct_rank(fund["fcfps_growth"].clip(upper=10), elig).fillna(0.5)
             + _pct_rank(-fund["share_growth"], elig).fillna(0.5)
             + _pct_rank(fund["fcf_yield"], elig).fillna(0.5)) / 4
    return _finish(score.where(flag), elig)


# --- Strategy 4: relative revenue growth -------------------------------------
REVENUE_GRID = [dict(measure=m, diff_min=d, accel_min=a, peer_decel=p)
                for m in ("rev_yoy", "rev_avg3", "rev_avg4") for d in (0.0, 0.05, 0.10)
                for a in (None, 0.0) for p in (None, 0.0)]


def industry_relative(w, elig, industries, min_peers=3):
    """Value minus the same-day industry median among eligible names."""
    vals = w.where(elig)
    med = vals.T.groupby(industries.to_numpy()).transform("median").T
    cnt = vals.notna().T.groupby(industries.to_numpy()).transform("sum").T
    return (vals - med).where(cnt >= min_peers), med.where(cnt >= min_peers)


def generate_revenue_momentum_signals(fund, elig, industries, measure, diff_min, accel_min, peer_decel, _cache={}):
    """BUY the company growing revenue faster than its industry peers.

    Requires growth differential >= diff_min, positive operating margin
    (acceptable fundamentals), optionally its own growth not decelerating
    (accel >= accel_min) and peers decelerating (industry median accel <=
    -peer_decel)."""
    key = (id(fund), measure)
    if key not in _cache:
        diff, _ = industry_relative(fund[measure], elig, industries)
        _, peer_accel = industry_relative(fund["rev_accel"], elig, industries)
        _cache[key] = (diff, peer_accel)
    diff, peer_accel = _cache[key]
    flag = (diff >= diff_min) & (fund["op_margin"] > 0) & fund[measure].notna()
    if accel_min is not None:
        flag &= fund["rev_accel"] >= accel_min
    if peer_decel is not None:
        flag &= peer_accel <= -peer_decel
    return _finish((diff + 1e-3).where(flag), elig)


# --- Strategy 2: pairs trading through divergence ----------------------------
PAIRS_GRID = [dict(L=L, min_corr=c, max_bdiff=b, z=z)
              for L in (60, 90, 120, 180) for c in (0.5, 0.7) for b in (0.1, 0.2, 9.0) for z in (1.5, 2.0, 2.5)]


class PairsEngine:
    """Same-industry pairs: rolling return correlation, beta difference and
    z-score of the log price ratio (negative z = first stock underperformed).

    Pairs are formed only among stocks that are in the universe on the same
    date; beta uses a 252-day window for both stocks.
    """

    def __init__(self, md, feat, elig, beta_lookback=252):
        self.md, self.feat, self.elig = md, feat, elig.to_numpy()
        self.beta = feat.get("beta", beta_lookback).to_numpy()
        self.logp = np.log(md.px.to_numpy(dtype=float))
        self.ret = feat.ret.to_numpy()
        cols = {s: i for i, s in enumerate(md.px.columns)}
        groups = md.info.loc[md.info.industry != "Unknown"].groupby("industry").groups
        ever = self.elig.any(axis=0)
        pairs = []
        for _, members in groups.items():
            idx = sorted(cols[s] for s in members if ever[cols[s]])
            for a, b in itertools.combinations(idx, 2):
                if (self.elig[:, a] & self.elig[:, b]).any():
                    pairs.append((a, b))
        self.pairs = np.array(pairs)
        self._stats = {}
        log(f"  pairs engine: {len(self.pairs):,} same-industry pairs")

    def stats(self, L):
        if L not in self._stats:
            a, b = self.pairs[:, 0], self.pairs[:, 1]
            spread = pd.DataFrame(self.logp[:, a] - self.logp[:, b])
            mu = spread.rolling(L, min_periods=int(0.9 * L)).mean()
            sd = spread.rolling(L, min_periods=int(0.9 * L)).std()
            z = ((spread - mu) / sd.where(sd > 0)).to_numpy("float32")
            ra, rb = pd.DataFrame(self.ret[:, a]), pd.DataFrame(self.ret[:, b])
            corr = ra.rolling(L, min_periods=int(0.9 * L)).corr(rb).to_numpy("float32")
            self._stats[L] = (z, corr)
        return self._stats[L]

    def signals(self, L, min_corr, max_bdiff, z, eps_ok=None):
        zs, corr = self.stats(L)
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        both = self.elig[:, a] & self.elig[:, b]
        bdiff = np.abs(self.beta[:, a] - self.beta[:, b])
        ok = both & (corr >= min_corr) & (bdiff <= max_bdiff)
        T, N = self.elig.shape
        out = np.full((T, N), np.nan, dtype="float32")
        # Stock a underperformed when z <= -thr; stock b when z >= thr.
        for side, cand, strength in ((a, ok & (zs <= -z), -zs), (b, ok & (zs >= z), zs)):
            vals = np.where(cand, strength, np.nan)
            order = np.argsort(side, kind="stable")
            sorted_side = side[order]
            starts = np.flatnonzero(np.r_[True, sorted_side[1:] != sorted_side[:-1]])
            with np.errstate(all="ignore"):
                red = np.fmax.reduceat(vals[:, order], starts, axis=1)
            cols = sorted_side[starts]
            out[:, cols] = np.fmax(out[:, cols], red)
        if eps_ok is not None:
            out[~(eps_ok.to_numpy() > 0)] = np.nan       # EPS deterioration filter
        return out


# ---------------------------------------------------------------------------
# 7. Research: event studies, parameter choice, waiting/holding analysis
# ---------------------------------------------------------------------------
def event_series(strength, labels, waits=WAITS):
    """Per signal date: mean excess 90-day return of flagged names, and count."""
    flags = ~np.isnan(strength)
    count = flags.sum(axis=1)
    out = {}
    for w in waits:
        _, ex = labels.target(w)
        vals = np.where(flags, ex, np.nan)
        with np.errstate(all="ignore"):
            out[w] = np.nanmean(vals, axis=1)
    return out, count


def window_stats(series, count, labels, cal, start, end, fit_row, wait):
    """Summary of an event series over signal dates [start, end] whose labels
    completed before ``fit_row``; dates thinned to every SAMPLE_EVERY sessions."""
    rows = np.arange(len(cal))
    done = labels.end_row(wait) < fit_row
    in_win = (cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end)) & done
    sample = rows[in_win][::SAMPLE_EVERY]
    v = series[sample]
    have = ~np.isnan(v)
    n_dates, n_sig = int(have.sum()), int(count[sample].sum())
    if n_dates < 3:
        return dict(mean=np.nan, t=np.nan, hit=np.nan, n_dates=n_dates, n_signals=n_sig, coverage=0.0)
    vv = v[have]
    overlap = max(1.0, 63 / SAMPLE_EVERY)       # 90-day labels overlap across samples
    t = vv.mean() / (vv.std(ddof=1) + 1e-12) * np.sqrt(n_dates / overlap)
    return dict(mean=float(vv.mean()), t=float(t), hit=float((vv > 0).mean()),
                n_dates=n_dates, n_signals=n_sig, coverage=n_dates / len(sample))


def admissible(st):
    return st["n_dates"] >= MIN_SIGNAL_DATES and st["n_signals"] >= MIN_SIGNALS and np.isfinite(st["t"])


@dataclass
class StrategySpec:
    name: str
    grid: list
    make: object                # callable(params) -> strength ndarray (T x N)


def research_strategy(spec, labels, cal, fit_rows):
    """Event-study every (params, wait) combination.

    Returns a table of train/validation stats and the walk-forward choice for
    each fit row: the admissible combination with the best t-statistic over
    the trailing WALK_FORWARD_YEARS of completed labels.
    """
    rows, series_store = [], {}
    train_fit = cal.searchsorted(pd.Timestamp(TRAIN[1]) + pd.Timedelta(days=1))
    valid_fit = cal.searchsorted(pd.Timestamp(VALID[1]) + pd.Timedelta(days=200))
    for i, params in enumerate(spec.grid):
        strength = spec.make(params)
        ser, cnt = event_series(strength, labels)
        series_store[i] = (ser, cnt)
        for w in WAITS:
            tr = window_stats(ser[w], cnt, labels, cal, *TRAIN, fit_row=valid_fit, wait=w)
            va = window_stats(ser[w], cnt, labels, cal, *VALID, fit_row=valid_fit, wait=w)
            rows.append(dict(strategy=spec.name, config=i, wait=w, **params,
                             **{f"train_{k}": v for k, v in tr.items()},
                             **{f"valid_{k}": v for k, v in va.items()}))
    table = pd.DataFrame(rows)
    choices = {}
    for fit_row in fit_rows:
        fit_date = cal[fit_row]
        start = fit_date - pd.DateOffset(years=WALK_FORWARD_YEARS)
        best = None
        for i in range(len(spec.grid)):
            ser, cnt = series_store[i]
            for w in WAITS:
                st = window_stats(ser[w], cnt, labels, cal, max(start, pd.Timestamp(TRAIN[0])),
                                  fit_date, fit_row, w)
                if admissible(st) and (best is None or st["t"] > best[2]["t"]):
                    best = (i, w, st)
        choices[fit_row] = best
    return table, choices, series_store


def horizon_study(strength, md, elig, start, end, wait=0):
    """Research only: mean excess return of signals over trading-day horizons."""
    px = md.px.to_numpy()
    cal = md.calendar
    flags = ~np.isnan(strength)
    rows = np.flatnonzero((cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end)))[::SAMPLE_EVERY]
    E = elig.to_numpy()
    out = {}
    for h in RESEARCH_HORIZONS:
        vals = []
        for t in rows:
            e, x = t + 1 + wait, t + 1 + wait + h
            if x >= len(cal):
                continue
            r = np.clip(px[x] / px[e] - 1, *LABEL_CLIP)
            base = np.nanmean(np.where(E[t], r, np.nan))
            sel = r[flags[t]]
            if np.isfinite(sel).any():
                vals.append(np.nanmean(sel) - base)
        out[f"{h}d"] = np.mean(vals) if vals else np.nan
    return out


def exit_rule_study(strength, md, feat, elig, start, end, wait=0):
    """Research only: compare the HW4 fixed 90-day exit with dynamic exits
    (RSI overbought, trailing stop) on the same signal events (gross returns)."""
    px = md.px.to_numpy()
    rsi = feat.get("rsi", 14).to_numpy()
    cal = md.calendar
    flags = ~np.isnan(strength)
    xpos = cal.searchsorted(cal + pd.Timedelta(days=HOLDING_DAYS))
    rows = np.flatnonzero((cal >= pd.Timestamp(start)) & (cal <= pd.Timestamp(end)))[::SAMPLE_EVERY]
    rules = {"fixed 90d": None, "RSI>=80": ("rsi", 80), "RSI>=85": ("rsi", 85), "RSI>=90": ("rsi", 90),
             "trail 10%": ("trail", 0.10), "trail 20%": ("trail", 0.20)}
    res = {k: [] for k in rules}
    for t in rows:
        e = t + 1 + wait
        if e >= len(cal) or xpos[e] >= len(cal):
            continue
        for s in np.flatnonzero(flags[t]):
            path = px[e:xpos[e] + 1, s]
            if not np.isfinite(path[0]):
                continue
            path = pd.Series(path).ffill().to_numpy()
            for name, rule in rules.items():
                exit_px = path[-1]
                if rule is not None:
                    if rule[0] == "rsi":
                        hits = np.flatnonzero(rsi[e + 1:xpos[e] + 1, s] >= rule[1])
                        if len(hits):
                            exit_px = path[hits[0] + 1]
                    else:
                        dd = path / np.maximum.accumulate(path) - 1
                        hits = np.flatnonzero(dd <= -rule[1])
                        if len(hits):
                            exit_px = path[hits[0]]
                res[name].append(float(np.clip(exit_px / path[0] - 1, *LABEL_CLIP)))
    return {k: (np.mean(v) if v else np.nan) for k, v in res.items()}


# ---------------------------------------------------------------------------
# 8. Model: gradient boosting on indicators + strategy signals
# ---------------------------------------------------------------------------
def model_feature_matrices(feat, fund, signals):
    m = {
        "rsi7": feat.get("rsi", 7), "rsi14": feat.get("rsi", 14),
        "vol_ratio": feat.get("vol", 21) / feat.get("volma", 21, 30),
        "vol63": feat.get("vol", 63), "beta252": feat.get("beta", 252),
        "mom21": feat.get("momentum", 21), "mom63": feat.get("momentum", 63),
        "mom252_21": feat.get("momentum", 252, 21), "dd252": feat.get("drawdown", 252),
    }
    for c in ["rev_yoy", "rev_accel", "op_margin", "eps_chg", "fcf_growth_ttm", "share_growth",
              "fcfps_growth", "fcf_yield"]:
        m[c] = fund[c]
    m = {k: v.to_numpy(dtype="float32") for k, v in m.items()}
    m["fcfps_growth"] = np.clip(m["fcfps_growth"], -10, 10)
    for name, s in signals.items():
        m[f"sig_{name}"] = np.nan_to_num(s, nan=0.0)
    return m


def model_scores(mats, labels, elig, cal, fit_rows, end_row, wait=0):
    """Walk-forward HGB: refit at each fit row on completed labels (from 2010),
    features = same-day percentile ranks, target = same-day percentile rank of
    the 90-day return (robust to outliers); predict every eligible row until
    the next fit."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    names = list(mats)
    E = elig.to_numpy()
    # Same-day percentile ranks make features comparable across regimes
    # (raw levels such as volatility shift a lot between 2010-12 and later).
    ranked = []
    for n in names:
        M = mats[n]
        if n.startswith("sig_"):
            ranked.append((M > 0).astype("float32"))
        else:
            ranked.append(pd.DataFrame(np.where(E, M, np.nan)).rank(axis=1, pct=True)
                          .fillna(0.5).to_numpy("float32"))
    X_all = np.stack(ranked, axis=-1)                         # T x N x F
    raw, _ = labels.target(wait)
    done_row = labels.end_row(wait)
    scores = np.full(E.shape, np.nan, dtype="float32")
    train_start = cal.searchsorted(pd.Timestamp(TRAIN[0]))
    bounds = list(fit_rows) + [end_row + 1]
    for fit, nxt in zip(bounds[:-1], bounds[1:]):
        rows = np.arange(train_start, fit)[::SAMPLE_EVERY]
        rows = rows[done_row[rows] < fit]
        Xs, ys = [], []
        for t in rows:
            ok = E[t] & np.isfinite(raw[t])
            if ok.sum() < 20:
                continue
            Xs.append(X_all[t, ok])
            ys.append(pd.Series(raw[t, ok]).rank(pct=True).to_numpy() - 0.5)   # rank target
        X, y = np.concatenate(Xs), np.concatenate(ys)
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=400, l2_regularization=1.0, random_state=0)
        model.fit(X, y)
        for t in range(fit, min(nxt, len(cal))):
            ok = E[t]
            if ok.any():
                scores[t, ok] = model.predict(X_all[t, ok])
        log(f"    model fit {cal[fit]:%Y-%m-%d}: {len(y):,} training rows")
    return scores, names


# ---------------------------------------------------------------------------
# 9. Portfolio construction and backtester (own implementation)
# ---------------------------------------------------------------------------
@dataclass
class Lot:
    lot_id: int
    symbol: str
    units: float              # adjusted-price units (total-return exposure)
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp    # entry_date + 90 calendar days
    tag: str = ""


class ScoreStrategy:
    """Buy the best-scored names not already held into K equal slots.

    ``scores`` maps a signal date to a Series (symbol -> score, best first).
    Each new position gets min(NAV/K, MAX_WEIGHT*NAV, remaining cash);
    ``max_new`` limits names bought per day (staggers entries so lots do not
    all mature on the same day). Orders fill at the next session's close.
    A stock with several signals gets one position (no duplicates).
    """

    def __init__(self, scores, k, max_new=None, max_weight=MAX_WEIGHT, min_order_fraction=MIN_ORDER_FRACTION,
                 vol=None, sectors=None, sector_cap=None, budget_scale=None, renew_rank=None,
                 size_mult=None):
        self.scores, self.k = scores, k
        self.size_mult = size_mult      # optional {date: Series(symbol -> multiplier)}; others 1.0
        # 90-day review: a lot maturing tomorrow is bought back at that close
        # (as a new 90-day lot) if the stock still ranks in the top
        # ``renew_rank`` candidates; otherwise it is sold and the slot reused.
        self.renew_rank = renew_rank if renew_rank is not None else 2 * k
        self.budget_scale = budget_scale  # optional {date: 0..1} volatility-target multiplier
        self.max_new = max_new or k
        self.max_weight, self.min_order_fraction = max_weight, min_order_fraction
        self.vol = vol                  # optional {date: Series(symbol -> volatility)} for inverse-vol sizing
        self.sectors = sectors          # optional Series symbol -> sector
        self.sector_cap = sector_cap    # optional max share of the K slots per sector

    def plan(self, signal_date, cash, nav, held, renewing=frozenset()):
        ranked = self.scores.get(signal_date)
        if ranked is None:
            return []
        slot = min(nav / self.k, self.max_weight * nav)
        keep = [s for s in ranked.index[:self.renew_rank] if s in renewing]
        if self.budget_scale is not None:
            slot *= self.budget_scale.get(signal_date, 1.0)
        min_order = self.min_order_fraction * nav
        vol = self.vol.get(signal_date) if self.vol is not None else None
        if self.sector_cap is not None:
            limit = max(1, math.floor(self.sector_cap * self.k))
            in_sector = pd.Series([self.sectors.get(h) for h in held]).value_counts().to_dict()
        orders, remaining, new = [], cash, 0
        keep_set = set(keep)
        # Buy-backs of maturing names first (they do not count against
        # max_new), then new names in score order.
        for symbol in keep + [s for s in ranked.index if s not in keep_set]:
            renewal = symbol in keep_set
            if remaining < min_order:
                break
            if not renewal:
                if new >= self.max_new:
                    break
                if symbol in held or symbol in renewing:   # renewing but not kept = sold at maturity
                    continue
            if self.sector_cap is not None:
                sec = self.sectors.get(symbol)
                if in_sector.get(sec, 0) >= limit:
                    continue
                in_sector[sec] = in_sector.get(sec, 0) + 1
            budget = slot
            if vol is not None and symbol in vol.index and np.isfinite(vol[symbol]) and vol[symbol] > 0:
                # inverse-volatility sizing around the equal slot, bounded 0.5x-1.5x
                budget = slot * float(np.clip(vol.median() / vol[symbol], 0.5, 1.5))
            if self.size_mult is not None:
                mult = self.size_mult.get(signal_date)
                if mult is not None and symbol in mult.index:
                    budget *= float(mult[symbol])
            budget = min(budget, self.max_weight * nav, remaining)
            if budget < min_order:
                break
            orders.append((symbol, budget))
            remaining -= budget
            new += not renewal
        return orders


TopKStrategy = ScoreStrategy      # name used in the self-checks


@dataclass
class BacktestResult:
    history: pd.DataFrame
    fills: pd.DataFrame
    open_lots: pd.DataFrame
    diagnostics: dict = field(default_factory=dict)


class Backtester:
    """Daily long-only accounting in adjusted-price units.

    Order of events each session d (all at the close):
      1. sell lots whose 90-day maturity has arrived and that have a quote on d
      2. fill orders planned after the previous close (skip if no quote on d)
      3. mark positions (last observed quote; flagged stale after 5 sessions)
      4. plan orders for the next session from this close's cash and NAV
    Sales on d add cash that can be planned for d+1. Purchases are sized so
    that budget = notional + fee, and total budgets never exceed cash.
    """

    def __init__(self, prices, initial_cash=INITIAL_CAPITAL, cost_bps=COST_BPS,
                 holding_days=HOLDING_DAYS, stale_limit=STALE_LIMIT, renew=False):
        self.prices = prices
        self.renew = renew      # 90-day review: let the strategy buy back maturing names
        self.calendar = pd.DatetimeIndex(prices.index)
        self.initial_cash = initial_cash
        self.cost = cost_bps / 10_000
        self.holding = pd.Timedelta(days=holding_days)
        self.stale_limit = stale_limit
        self._px = prices.to_numpy()
        self._col = {s: i for i, s in enumerate(prices.columns)}

    def run(self, strategy, first_entry, end):
        cal = self.calendar
        sessions = cal[(cal >= pd.Timestamp(first_entry)) & (cal <= pd.Timestamp(end))]
        if len(sessions) < 2:
            raise ValueError("Backtest window has fewer than two sessions")
        signal_ix = cal.get_loc(sessions[0]) - 1
        if signal_ix < 0:
            raise ValueError("Need one session before the first entry for the signal")

        cash = self.initial_cash
        lots: dict[int, Lot] = {}
        fills, history, skipped, stale_events = [], [], [], []
        last_px = np.full(len(self._col), np.nan)
        last_seen = np.full(len(self._col), -1)
        next_id = 0

        signal_date = cal[signal_ix]
        orders = strategy.plan(signal_date, cash, cash, set())

        for d in sessions:
            di = cal.get_loc(d)
            quotes = self._px[di]

            # 1. exits at maturity (first quoted close on/after the due date)
            for lot in [l for l in lots.values() if d >= l.due_date]:
                px = quotes[self._col[lot.symbol]]
                if not np.isfinite(px):
                    continue                  # wait for a real quote; never invent a fill
                gross = lot.units * px
                fee = gross * self.cost
                cash += gross - fee
                fills.append(dict(date=d, side="SELL", symbol=lot.symbol, lot_id=lot.lot_id,
                                  units=lot.units, price=px, fee=fee, cash_change=gross - fee,
                                  signal_date=pd.NaT, entry_date=lot.entry_date,
                                  pnl=gross - fee - (lot.units * lot.entry_price + lot.entry_fee)))
                del lots[lot.lot_id]

            # 2. entries planned after the previous close
            for symbol, budget in orders:
                px = quotes[self._col[symbol]] if symbol in self._col else np.nan
                if not np.isfinite(px):
                    skipped.append((d, symbol))
                    continue
                budget = min(budget, cash)
                units = budget / (px * (1 + self.cost))
                notional = units * px
                fee = notional * self.cost
                cash -= notional + fee
                assert cash >= -1e-6, "Spent cash that does not exist"
                lots[next_id] = Lot(next_id, symbol, units, d, px, fee, d + self.holding)
                fills.append(dict(date=d, side="BUY", symbol=symbol, lot_id=next_id, units=units,
                                  price=px, fee=fee, cash_change=-(notional + fee),
                                  signal_date=signal_date, entry_date=d, pnl=np.nan))
                next_id += 1

            # 3. mark to market
            seen = np.isfinite(quotes)
            last_px[seen] = quotes[seen]
            last_seen[seen] = di
            held = {l.symbol for l in lots.values()}
            positions = sum(l.units * last_px[self._col[l.symbol]] for l in lots.values())
            for symbol in held:
                gap = di - last_seen[self._col[symbol]]
                if gap > self.stale_limit:
                    stale_events.append((d, symbol, gap))
            nav = cash + positions
            history.append(dict(date=d, cash=cash, positions=positions, value=nav, n_lots=len(lots)))

            # 4. plan next session's orders
            signal_date = d
            if d == sessions[-1]:
                orders = []
            elif self.renew:
                # Lots due at the next session are sold there first (step 1),
                # so their estimated after-fee proceeds can fund a buy-back
                # at the same close; fills are still capped by actual cash.
                nxt = cal[di + 1]
                due = [l for l in lots.values() if nxt >= l.due_date]
                maturing = {l.symbol for l in due}
                proceeds = sum(l.units * last_px[self._col[l.symbol]] for l in due) * (1 - self.cost)
                orders = strategy.plan(d, cash + proceeds, nav, held - maturing, renewing=maturing)
            else:
                orders = strategy.plan(d, cash, nav, held)

        open_lots = pd.DataFrame([dict(lot.__dict__, mark=last_px[self._col[lot.symbol]],
                                       last_quote=cal[last_seen[self._col[lot.symbol]]],
                                       value=lot.units * last_px[self._col[lot.symbol]]) for lot in lots.values()])
        overdue = open_lots.loc[open_lots.due_date.le(sessions[-1])] if len(open_lots) else open_lots
        diagnostics = dict(
            skipped_orders=len(skipped),
            stale_mark_events=len(stale_events),
            stale_symbols=sorted({s for _, s, _ in stale_events}),
            overdue_lots=len(overdue),
            overdue_value=float(overdue.value.sum()) if len(overdue) else 0.0,
        )
        return BacktestResult(pd.DataFrame(history).set_index("date"), pd.DataFrame(fills),
                              open_lots, diagnostics)


def scores_from_matrix(matrix, cal, columns, start, end, shift=0):
    """Wide score matrix -> {signal date: Series sorted best first}.

    ``shift`` implements a waiting period: a signal from row t is acted on
    after the close of row t+shift (so it fills at t+shift+1)."""
    out = {}
    lo, hi = cal.searchsorted(pd.Timestamp(start)), cal.searchsorted(pd.Timestamp(end), side="right")
    for t in range(max(lo, shift), hi):
        row = matrix[t - shift]
        ok = np.flatnonzero(np.isfinite(row))
        if len(ok):
            s = pd.Series(row[ok], index=columns[ok])
            out[cal[t]] = s.sort_values(ascending=False, kind="stable")
    return out


# ---------------------------------------------------------------------------
# 10. Metrics
# ---------------------------------------------------------------------------
def performance(values, initial=INITIAL_CAPITAL, annual_rf=ANNUAL_RF, start_date=None):
    """Metrics on a daily value series. ``initial`` is prepended at ``start_date``
    (the signal close before the first fill) so entry fees hit the first return."""
    v = values.copy()
    if start_date is not None:
        v = pd.concat([pd.Series([initial], index=[pd.Timestamp(start_date)]), v])
    r = v.pct_change().dropna()
    years = (v.index[-1] - v.index[0]).days / 365.25
    rf_daily = (1 + annual_rf) ** (1 / 252) - 1
    excess = r - rf_daily
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    return pd.Series({
        "Final value ($)": v.iloc[-1],
        "Total return": v.iloc[-1] / initial - 1,
        "CAGR": (v.iloc[-1] / initial) ** (1 / years) - 1,
        "Volatility (ann.)": r.std() * np.sqrt(252),
        "Max drawdown": (v / v.cummax() - 1).min(),
        "Sharpe (rf 2%)": excess.mean() / excess.std() * np.sqrt(252),
        "Sortino (rf 2%)": excess.mean() / downside * np.sqrt(252) if downside > 0 else np.nan,
    })


def trade_stats(result):
    f = result.fills
    sells = f.loc[f.side.eq("SELL")] if len(f) else f
    if not len(sells):
        return pd.Series(dtype=float)
    wins, losses = sells.pnl[sells.pnl > 0], sells.pnl[sells.pnl <= 0]
    return pd.Series({
        "Trades (closed)": len(sells),
        "Win rate": len(wins) / len(sells),
        "Avg win ($)": wins.mean() if len(wins) else 0.0,
        "Avg loss ($)": losses.mean() if len(losses) else 0.0,
        "Profit factor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
        "Avg holding (cal. days)": (sells.date - sells.entry_date).dt.days.mean(),
        "Transaction costs ($)": f.fee.sum(),
        "Avg invested": (result.history.positions / result.history.value).mean(),
    })


def benchmark_values(bench, sessions, initial=INITIAL_CAPITAL):
    """$initial invested in ^SP500TR at the first session's close."""
    b = bench.reindex(sessions)
    assert b.notna().all(), "Benchmark has gaps on market sessions"
    return initial * b / b.iloc[0]


def regime_table(strategy_values, bench_values, bench_full):
    """Daily returns of both portfolios split by market environment."""
    r_s, r_b = strategy_values.pct_change(), bench_values.pct_change()
    mkt = bench_full.pct_change()
    trend = bench_full / bench_full.shift(126) - 1            # 6-month benchmark return
    vol = mkt.rolling(63).std() * np.sqrt(252)
    idx = r_s.index[1:]
    regime = pd.DataFrame(index=idx)
    regime["trend"] = np.select([trend.reindex(idx) > 0.05, trend.reindex(idx) < -0.05],
                                ["bull (6m > +5%)", "bear (6m < -5%)"], "sideways")
    regime["volatility"] = np.where(vol.reindex(idx) > vol.reindex(idx).median(), "high vol", "low vol")
    rows = []
    for col in ["trend", "volatility"]:
        for name, g in regime.groupby(col):
            rows.append(dict(regime=name, days=len(g),
                             strategy_ann_return=r_s.reindex(g.index).mean() * 252,
                             benchmark_ann_return=r_b.reindex(g.index).mean() * 252))
    return pd.DataFrame(rows)


def plot_curves(strategy_values, bench_values, path, title, label="Strategy", extra=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(strategy_values.index, strategy_values / INITIAL_CAPITAL - 1, label=label)
    for (name, v), c in zip((extra or {}).items(), itertools.cycle(["#9467bd", "#2ca02c", "#d62728", "#8c564b", "#17becf"])):
        ax.plot(v.index, v / INITIAL_CAPITAL - 1, label=name, color=c, lw=1, alpha=0.8)
    ax.plot(bench_values.index, bench_values / INITIAL_CAPITAL - 1, label="^SP500TR buy & hold")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("Cumulative return")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Accounting / timing audit of a real run
# ---------------------------------------------------------------------------
def audit(result, initial=INITIAL_CAPITAL, end=None):
    f, h = result.fills, result.history
    buys, sells = f.loc[f.side.eq("BUY")], f.loc[f.side.eq("SELL")]
    # Cash reconciles with the fill log
    assert np.isclose(initial + f.cash_change.sum(), h.cash.iloc[-1], rtol=0, atol=1e-4)
    # Final value = cash + open lots at their last mark
    open_value = result.open_lots.value.sum() if len(result.open_lots) else 0.0
    assert np.isclose(h.cash.iloc[-1] + open_value, h.value.iloc[-1], rtol=1e-10)
    # Fills happen strictly after the signal close
    assert (buys.date > buys.signal_date).all()
    # Sales only on/after 90 calendar days, never after the end date
    assert ((sells.date - sells.entry_date).dt.days >= HOLDING_DAYS).all()
    if end is not None:
        assert f.date.max() <= pd.Timestamp(end)
    assert h.cash.min() >= -1e-6
    # Never more than one open lot per stock
    delay = (sells.date - sells.entry_date).dt.days - HOLDING_DAYS
    print("Audit passed: cash reconciles with fills, final value = cash + open lots, "
          "fills after signals, exits >= 90 days, no negative cash.")
    print(f"  buys={len(buys)}, sells={len(sells)}, open lots at end={len(result.open_lots)}, "
          f"fees=${f.fee.sum():,.0f}; exit delay beyond 90d: max {delay.max()} days "
          f"(weekends/holidays/missing quotes)")
    print(f"  diagnostics: {result.diagnostics}")
    first = buys.iloc[0]
    exit_row = sells.loc[sells.lot_id.eq(first.lot_id)]
    print(f"  Hand check, first lot: {first.symbol} bought {first.date:%Y-%m-%d} "
          f"{first.units:.4f} units @ {first.price:.4f} (+fee {first.fee:.2f})")
    if len(exit_row):
        e = exit_row.iloc[0]
        manual = e.units * e.price * (1 - COST_BPS / 1e4) - (first.units * first.price + first.fee)
        print(f"    sold {e.date:%Y-%m-%d} @ {e.price:.4f}; PnL {e.pnl:,.2f} "
              f"(hand: units*exit*(1-c) - cost = {manual:,.2f})")


# ---------------------------------------------------------------------------
# 11. Building candidate score matrices
# ---------------------------------------------------------------------------
def shift_rows(M, w):
    """Waiting period: the signal of row t is acted on after the close of t+w."""
    if w == 0:
        return M
    out = np.full_like(M, np.nan)
    out[w:] = M[:-w]
    return out


def piecewise_matrix(spec, choices, fit_rows, T, cache):
    """Walk-forward signal: between consecutive fit rows use the parameters
    (and waiting period) chosen at the earlier fit row."""
    out = np.full((T, len(spec.columns)), np.nan, dtype="float32")
    bounds = sorted(fit_rows) + [T]
    # Rows before the first fit use the first choice (= the fixed parameters,
    # chosen on 2010-2012), so train-period checks of this signal are defined.
    for idx, (f, nxt) in enumerate(zip(bounds[:-1], bounds[1:])):
        start = 0 if idx == 0 else f
        choice = choices.get(f)
        if choice is None:
            continue
        i, w, _ = choice
        key = (spec.name, i)
        if key not in cache:
            cache[key] = spec.make(spec.grid[i])
        out[start:nxt] = shift_rows(cache[key], w)[start:nxt]
    return out


def pct_among_flagged(M):
    return pd.DataFrame(M).rank(axis=1, pct=True).to_numpy(dtype="float32")


def combine(mats, weights, min_agree=1):
    """Composite score = weighted sum of each strategy's same-day percentile
    among its own flagged names; only stocks flagged by >= min_agree
    strategies are candidates."""
    total, agree = None, None
    for name, M in mats.items():
        p = pct_among_flagged(M)
        flag = ~np.isnan(p)
        contrib = weights[name] * np.nan_to_num(p)
        total = contrib if total is None else total + contrib
        agree = flag.astype(int) if agree is None else agree + flag
    total = total + agree          # more agreeing strategies rank first
    total[agree < min_agree] = np.nan
    return total.astype("float32")


def fill_with_model(signal_score, model_score):
    """Stocks with a signal first (by composite), then the rest by model score."""
    out = np.where(np.isnan(signal_score), np.nan, signal_score + 100.0)
    fill = np.isnan(out) & ~np.isnan(model_score)
    out[fill] = model_score[fill]
    return out.astype("float32")


# ---------------------------------------------------------------------------
# 11b. Regime overlay: lean into sectors / size buckets that start to outperform
# ---------------------------------------------------------------------------
REGIME_GRID = [dict(groups=g, lookback=L, mode=m, weight=w)
               for g in ("sector", "size", "both") for L in (21, 63, 126)
               for m in ("level", "turn") for w in (0.5, 1.0)]
SIZE_CUTS = (100, 250)      # screened rank: 1-100 mega, 101-250 large, 251-500 smaller
SIZE_LABELS = ("mega (1-100)", "large (101-250)", "smaller (251-500)")


def _entry_quarter(cal):
    """Quarter of the next session (membership for an entry after row t)."""
    nxt = pd.Series(cal[1:].append(pd.DatetimeIndex([pd.NaT])), index=cal)
    return nxt.dt.to_period("Q").dt.start_time.to_numpy()


def group_codes(md):
    """Group-id matrices (T x N, -1 = no group): static snapshot sector, and
    a dated size bucket from the quarter's screened cap rank."""
    cal, cols = md.calendar, md.px.columns
    sector_names = sorted(md.info.sector.unique())
    sec = md.info.sector.map({n: i for i, n in enumerate(sector_names)}).to_numpy()
    sector = np.broadcast_to(sec, (len(cal), len(cols))).copy()
    rank = (md.membership.pivot_table(index="quarter_start", columns="symbol", values="screened_rank",
                                      aggfunc="min").reindex(columns=cols))
    R = rank.reindex(_entry_quarter(cal)).to_numpy()
    size = np.full(R.shape, -1)
    size[R <= SIZE_CUTS[0]] = 0
    size[(R > SIZE_CUTS[0]) & (R <= SIZE_CUTS[1])] = 1
    size[R > SIZE_CUTS[1]] = 2
    return {"sector": (sector, sector_names), "size": (size, list(SIZE_LABELS))}


def group_relative_strength(ret, elig, codes, lookback, mode, min_members=3):
    """Each stock's *group* relative strength at the close of row t.

    Group return = equal-weight mean daily log return of its eligible members;
    relative strength = group minus universe, summed over ``lookback``
    sessions. mode="level" uses that sum; mode="turn" uses its change versus
    the previous ``lookback`` sessions (a group that is *starting* to
    outperform). Only returns up to the close of t are used.
    Returns (stock-level T x N matrix, group-level T x G frame).
    """
    r = np.log1p(np.asarray(ret, dtype=float))
    valid = np.asarray(elig) & np.isfinite(r) & (codes >= 0)
    rv = np.where(valid, r, 0.0)
    univ = rv.sum(1) / np.maximum(valid.sum(1), 1)
    G = int(codes.max()) + 1
    grp = np.full((len(r), G), np.nan)
    for g in range(G):
        m = valid & (codes == g)
        n = m.sum(1)
        grp[:, g] = np.where(n >= min_members, np.where(m, r, 0.0).sum(1) / np.maximum(n, 1), np.nan)
    rel = pd.DataFrame(grp - univ[:, None])
    rs = rel.rolling(lookback, min_periods=int(0.8 * lookback)).sum()
    if mode == "turn":
        rs = rs - rs.shift(lookback)
    rs_np = rs.to_numpy()
    rows = np.arange(len(r))[:, None]
    stock = np.where(codes >= 0, rs_np[rows, np.clip(codes, 0, None)], np.nan)
    return stock.astype("float32"), rs


def regime_overlay(base, group_rs, weight):
    """Score = base percentile + weight x group-strength percentile, both among
    the stocks the base strategy flags that day. Only flagged stocks remain
    candidates, so the overlay re-orders picks toward strong groups."""
    flagged = ~np.isnan(base)
    p_base = pct_among_flagged(base)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows -> NaN
        p_grp = np.nanmean(np.stack([pct_among_flagged(np.where(flagged, g, np.nan)) for g in group_rs]), axis=0)
    p_grp = np.where(flagged, np.nan_to_num(p_grp, nan=0.5), np.nan)
    return (p_base + weight * p_grp).astype("float32")


# ---------------------------------------------------------------------------
# 11c. Volatility component: reduce total portfolio volatility
# ---------------------------------------------------------------------------
VOL_GRID = ([dict(kind="none")]
            + [dict(kind="lowvol_tilt", w=w) for w in (0.5, 1.0, 2.0)]
            + [dict(kind="beta_tilt", w=w) for w in (0.5, 1.0, 2.0)]
            + [dict(kind="vol_filter", q=q) for q in (0.5, 0.7)]
            + [dict(kind="vol_target", target=t) for t in (0.10, 0.12, 0.15)]
            + [dict(kind="lowvol_tilt+vol_target", w=1.0, target=0.15),
               dict(kind="beta_tilt+vol_target", w=1.0, target=0.15),
               dict(kind="vol_filter+vol_target", q=0.7, target=0.15)])


def describe_vol(params):
    if params is None or params.get("kind") == "none":
        return "none"
    return params["kind"] + " (" + ", ".join(f"{k}={v}" for k, v in params.items() if k != "kind") + ")"


def apply_vol_component(M, params, vol63, beta, elig, mkt_vol, cal):
    """Return (score matrix, budget scale map) for one volatility setting.

    lowvol_tilt : score += w x percentile of LOW 63-day volatility (among flagged)
    beta_tilt   : score += w x percentile of LOW 252-day beta
    vol_filter  : drop flagged names above the q-th universe percentile of volatility
    vol_target  : scale every new position by min(1, target / benchmark 63-day vol);
                  the unspent cash waits (long-only, no leverage, so this only de-risks)
    All inputs are known at the signal close.
    """
    kind = params.get("kind", "none")
    scale = None
    if "lowvol_tilt" in kind:
        M = regime_overlay(M, [-vol63], params["w"])
    if "beta_tilt" in kind:
        M = regime_overlay(M, [-beta], params["w"])
    if "vol_filter" in kind:
        pct = pd.DataFrame(np.where(elig, vol63, np.nan)).rank(axis=1, pct=True).to_numpy()
        M = np.where(pct <= params["q"], M, np.nan).astype("float32")
    if "vol_target" in kind:
        ratio = np.clip(params["target"] / mkt_vol, 0.0, 1.0)
        scale = {d: (float(x) if np.isfinite(x) else 1.0) for d, x in zip(cal, ratio)}
    return M, scale


# ---------------------------------------------------------------------------
# 11e. Growth / stable rotation and the 90-day review
# ---------------------------------------------------------------------------
RENEW_MULTIPLES = (1, 2, 3)      # buy back a maturing name if it ranks in the top m*K
ROTATION_GRID = [dict(ext=e, calm=c, dip=d, stable=st)
                 for e in (0.2, 0.4) for c in (True, False) for d in (True, False)
                 for st in ("lowbeta", "current")]
GROWTH_TOP = 0.2                 # "faster-growing" = top 20% of revenue growth that day


def market_is_calm(bench, cal, lookback=63, median_window=252):
    """True when the index's 63-day realized volatility is below its own
    trailing one-year median (known at the close of each row)."""
    b = bench.reindex(cal)
    vol = b.pct_change().rolling(lookback).std()
    return (vol < vol.rolling(median_window, min_periods=200).median()).to_numpy()


def growth_rotation_scores(fund, md, elig, beta, vol63, current, ext, calm, dip, stable):
    """Rotate between fast revenue growers and stable companies.

    Growth names : top GROWTH_TOP of revenue growth (YoY) among eligible
                   stocks, with positive operating margin.
    Buy growth   : when the market is calm (if ``calm``) AND the stock has not
                   already beaten the index by more than ``ext`` over the past
                   6 months (big winners are not bought again: their slot goes
                   to stable names when the lot matures). If ``dip``, a growth
                   name trading below its 200-day average is a buy-back
                   opportunity in any regime and ranks first.
    Stable names : fill the remaining slots, ranked by low beta + low
                   volatility ("lowbeta", positive operating margin) or by the
                   current final strategy's ranking ("current").
    Score: growth buys 2..3.5 (always ahead), stable names 0..1.
    """
    cal = md.calendar
    E = elig.to_numpy()
    pct = lambda X: pd.DataFrame(np.where(E, X, np.nan)).rank(axis=1, pct=True).to_numpy()
    opm = fund["op_margin"].to_numpy()
    g = fund["rev_yoy"].to_numpy()
    pg = pct(np.where(np.isfinite(opm) & (opm > 0), g, np.nan))
    growth = np.isfinite(pg) & (pg >= 1 - GROWTH_TOP)
    below = md.px.to_numpy() < md.px.rolling(200, min_periods=180).mean().to_numpy()
    b = md.bench.reindex(cal)
    rel6m = md.px.pct_change(126, fill_method=None).sub(b.pct_change(126), axis=0).to_numpy()
    extended = np.nan_to_num(rel6m, nan=0.0) > ext
    regime = market_is_calm(md.bench, cal)[:, None] if calm else np.ones((len(cal), 1), bool)
    dip_buy = growth & below if dip else np.zeros_like(growth)
    buy_growth = growth & ((regime & ~extended) | dip_buy)
    if stable == "lowbeta":
        st = np.where(np.isfinite(opm) & (opm > 0), 0.5 * pct(-beta) + 0.5 * pct(-vol63), np.nan)
    else:
        st = pct_among_flagged(current)
    s = np.where(buy_growth, 2 + np.nan_to_num(pg) + 0.5 * dip_buy, st)
    s[~E] = np.nan
    return s.astype("float32")


def describe_rotation(p):
    return (f"growth rotation (beat-index cutoff {p['ext']:.0%}, calm-market switch {'on' if p['calm'] else 'off'}, "
            f"200-day dip buy-back {'on' if p['dip'] else 'off'}, stable = {p['stable']})")


# ---------------------------------------------------------------------------
# 11f. Growth tilt: temporarily overweight fast revenue growers
# ---------------------------------------------------------------------------
DEFENSIVE_GROUPS = {
    "pharma+utilities": lambda info: info.industry.str.startswith("Drug Manufacturers") | info.sector.eq("Utilities"),
    "defensive sectors": lambda info: info.sector.isin(["Healthcare", "Utilities", "Consumer Defensive"]),
}
GROWTH_TILT_GRID = (
    [dict(trigger="lowvol", mult=m, exit=x, add=a)
     for m in (1.5, 2.0) for x in (0.03, 0.05, 0.08) for a in (False, True)]
    + [dict(trigger=t, mult=m, exit=x, add=a, group=g, lookback=L, threshold=h)
       for t in ("defensive", "both") for m in (1.5, 2.0) for x in (0.03, 0.05, 0.08) for a in (False, True)
       for g in DEFENSIVE_GROUPS for L in (63, 126) for h in (0.03, 0.05)])
MAX_EPISODE = 252           # a tilt that never pays off is ended after one year


def growth_names(fund, elig):
    """Fast revenue growers: top GROWTH_TOP of YoY revenue growth among
    eligible stocks with positive operating margin; also returns the
    same-day growth percentile (0..1)."""
    E = elig.to_numpy()
    opm = fund["op_margin"].to_numpy()
    g = np.where(E & np.isfinite(opm) & (opm > 0), fund["rev_yoy"].to_numpy(), np.nan)
    pg = pd.DataFrame(g).rank(axis=1, pct=True).to_numpy()
    return np.isfinite(pg) & (pg >= 1 - GROWTH_TOP), pg


def _basket_returns(ret, members):
    """Equal-weight return on day t of the stocks that were members at t-1."""
    mem = np.vstack([np.zeros((1, members.shape[1]), bool), members[:-1]]) & np.isfinite(ret)
    n = mem.sum(1)
    return np.where(n > 0, np.where(mem, ret, 0.0).sum(1) / np.maximum(n, 1), np.nan)


def growth_tilt_episodes(md, elig, growth, trigger, exit, group=None, lookback=None, threshold=None):
    """Daily on/off state of the growth tilt, known at each close.

    Triggers
      lowvol    : the index's 63-day volatility is below its one-year median.
      defensive : the defensive group (equal weight) beat the eligible
                  universe by more than ``threshold`` over ``lookback`` sessions.
      both      : either trigger, each with its own episode.
    An episode starts when its trigger fires. It ends once the growth basket
    has earned ``exit`` more than ^SP500TR since the start (the excess is
    "realized" and the overweight is no longer needed), after MAX_EPISODE
    sessions, or (lowvol only) when volatility is no longer low. After an
    ending, the trigger must switch off once before a new episode can start.
    """
    cal = md.calendar
    E = elig.to_numpy()
    ret = md.px.pct_change(fill_method=None).to_numpy()
    g_ret = np.nan_to_num(_basket_returns(ret, growth))
    b_ret = np.nan_to_num(md.bench.reindex(cal).pct_change().to_numpy())
    triggers = {}
    if trigger in ("lowvol", "both"):
        triggers["lowvol"] = market_is_calm(md.bench, cal)
    if trigger in ("defensive", "both"):
        dmask = DEFENSIVE_GROUPS[group](md.info).to_numpy()
        d_ret = _basket_returns(ret, E & dmask[None, :])
        u_ret = _basket_returns(ret, E)
        rel = pd.Series(np.log1p(np.nan_to_num(d_ret)) - np.log1p(np.nan_to_num(u_ret)))
        triggers["defensive"] = (rel.rolling(lookback, min_periods=int(0.8 * lookback)).sum() > threshold).to_numpy()
    active = np.zeros(len(cal), bool)
    for name, trig in triggers.items():
        on, armed, excess, length = False, True, 0.0, 0
        for t in range(len(cal)):
            if on:
                excess = (1 + excess) * (1 + g_ret[t]) / (1 + b_ret[t]) - 1
                length += 1
                if excess >= exit or length >= MAX_EPISODE or (name == "lowvol" and not trig[t]):
                    on, armed = False, False
            if not on:
                if not trig[t]:
                    armed = True
                elif armed:
                    on, armed, excess, length = True, False, 0.0, 0
            active[t] |= on
    return active


def apply_growth_tilt(M, growth, pg, active, mult, add):
    """Score matrix and size multipliers for the tilt.

    While the tilt is on, fast growers among the base strategy's picks are
    bought at ``mult`` x the normal slot (still capped at MAX_WEIGHT). With
    ``add``, fast growers the base strategy does not flag also become
    candidates (ranked as a median pick plus their growth percentile).
    When the tilt is off, new purchases are normal size again; existing
    lots shrink back to normal at their 90-day maturity.
    """
    on = active[:, None] & growth
    size = np.where(on, mult, 1.0).astype("float32")
    if not add:
        return M, size
    p = pct_among_flagged(M)
    boosted = np.where(on, np.nan_to_num(p, nan=0.5) + 0.5 * np.nan_to_num(pg), p)
    return boosted.astype("float32"), size


def describe_tilt(p):
    if p is None:
        return "none"
    txt = f"growth tilt ({p['trigger']}, {p['mult']}x size, back to normal after +{p['exit']:.0%} excess"
    if p["trigger"] != "lowvol":
        txt += f", {p['group']} beat universe by > {p['threshold']:.0%} over {p['lookback']}d"
    return txt + (", adds non-FCF growers" if p["add"] else "") + ")"


# ---------------------------------------------------------------------------
# 11g. Dispersion: single-stock volatility versus index volatility
# ---------------------------------------------------------------------------
DISPERSION_GRID = [dict(window=w, buckets=nb, mode=m)
                   for w in (21, 63) for nb in (2, 3) for m in ("weights", "best", "fcf+lean")]
DISPERSION_HISTORY = 504     # dispersion is ranked against the trailing two years
BUCKET_NAMES = {2: ("low", "high"), 3: ("low", "mid", "high")}


def dispersion_series(md, elig, feat, window):
    """Dispersion at each close: average realized volatility of eligible
    stocks minus the index's realized volatility (same window, annualized).
    High dispersion = stocks move on their own news (low correlation), low
    dispersion = everything moves with the market. Also returns its
    percentile against the trailing DISPERSION_HISTORY sessions (point in time).
    """
    vol = feat.get("vol", window).to_numpy()
    E = elig.to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        stock_vol = np.nanmean(np.where(E, vol, np.nan), axis=1)
    index_vol = calculate_realized_volatility(md.bench.reindex(md.calendar).pct_change().to_frame(), window
                                              ).iloc[:, 0].to_numpy()
    d = pd.Series(stock_vol - index_vol, index=md.calendar)
    pct = d.rolling(DISPERSION_HISTORY, min_periods=252).rank(pct=True)
    return d, pd.Series(stock_vol, index=md.calendar), pd.Series(index_vol, index=md.calendar), pct


def dispersion_buckets(pct, nb):
    p = np.asarray(pct, dtype=float)
    return np.where(np.isfinite(p), np.minimum(np.floor(np.nan_to_num(p) * nb), nb - 1), -1).astype(int)


def bucket_table(events, end_rows, buckets, nb, lo, hi, done_before):
    """Mean 90-day excess return of each strategy's signals by dispersion
    bucket, over signal rows [lo, hi) whose outcomes finished before
    ``done_before``; rows thinned to every SAMPLE_EVERY sessions."""
    rows = np.arange(lo, hi)[::SAMPLE_EVERY]
    rows = rows[(end_rows[rows] < done_before) & (buckets[rows] >= 0)]
    out = {}
    for name, ev in events.items():
        v, b = ev[rows], buckets[rows]
        out[name] = {}
        for k_ in range(nb):
            x = v[(b == k_) & np.isfinite(v)]
            n_eff = max(len(x) / max(1.0, 63 / SAMPLE_EVERY), 1)
            out[name][k_] = (float(x.mean()) if len(x) else np.nan,
                             float(x.mean() / (x.std(ddof=1) + 1e-12) * np.sqrt(n_eff)) if len(x) > 2 else np.nan,
                             len(x))
    return out


def dispersion_weights(events, end_rows, buckets, nb, fit_rows, T, start_row, mode, anchor="fcf"):
    """Per-row strategy weights from the bucket table estimated at each fit
    row (expanding window from ``start_row``, finished outcomes only). Rows
    before the first fit use the first fit's table (in-sample for 2010-12).

    weights  : each strategy weighted by its positive mean excess in today's bucket
    best     : only the strategy with the highest mean excess in today's bucket
    fcf+lean : the anchor strategy always at weight 1; another strategy joins
               (weight up to 1) when its mean excess in today's bucket is positive,
               scaled by its mean relative to the anchor's
    """
    names = list(events)
    W = {n: np.zeros(T, dtype="float32") for n in names}
    bounds = sorted(fit_rows) + [T]
    for idx, (f, nxt) in enumerate(zip(bounds[:-1], bounds[1:])):
        tab = bucket_table(events, end_rows, buckets, nb, start_row, f, f)
        lo = 0 if idx == 0 else f
        for k_ in range(-1, nb):
            rows = np.arange(lo, nxt)[buckets[lo:nxt] == k_]
            if not len(rows):
                continue
            if k_ < 0:          # no dispersion history yet: average over buckets
                m = {n: np.nanmean([tab[n][j][0] for j in range(nb)]) for n in names}
            else:
                m = {n: tab[n][k_][0] for n in names}
            m = {n: (v if np.isfinite(v) else 0.0) for n, v in m.items()}
            if mode == "weights":
                pos = {n: max(v, 0.0) for n, v in m.items()}
                tot = sum(pos.values())
                w = {n: (v / tot if tot > 0 else float(n == anchor)) for n, v in pos.items()}
            elif mode == "best":
                best = max(m, key=m.get)
                w = {n: float(n == best) for n in names}
            else:
                a = max(m[anchor], 1e-6)
                w = {n: (1.0 if n == anchor else float(np.clip(max(m[n], 0.0) / a, 0.0, 1.0))) for n in names}
            for n in names:
                W[n][rows] = w[n]
    return W


def combine_dynamic(mats, W):
    """Score = sum over strategies of weight(t) x same-day percentile among
    that strategy's flagged names; candidates = names flagged by at least one
    strategy with a positive weight today."""
    total, cand = None, None
    for n, M in mats.items():
        p = pct_among_flagged(M)
        w = W[n][:, None]
        contrib = w * np.nan_to_num(p)
        live = (~np.isnan(p)) & (w > 0)
        total = contrib if total is None else total + contrib
        cand = live if cand is None else cand | live
    return np.where(cand, total, np.nan).astype("float32")


# ---------------------------------------------------------------------------
# 11h. Full factor search: sector revenue leaders, dynamic strategy weights,
#      mega-cap leadership tilt
# ---------------------------------------------------------------------------
SECTOR_REV_GRID = [dict(measure=m, top=q, margin=mg)
                   for m in ("rev_yoy", "rev_avg4") for q in (0.1, 0.2, 0.3) for mg in (True, False)]
SEARCH_METHODS = ("equal", "t-weighted", "trailing 126d", "trailing 252d", "trailing 252d best", "dispersion")
MEGA_GRID = [dict(lookback=L, w=w, add=a) for L in (63, 126) for w in (0.5, 1.0) for a in (False, True)]
MEGA_CUT = 100              # "mega cap" = top 100 of the dated screened cap rank
SEARCH_TOL = 0.01           # candidates within 1%/yr of the best score count as ties
SEARCH_TOP = 10             # best signal combinations carried into the overlay search


def generate_sector_revenue_leaders(fund, elig, sectors, measure, top, margin, min_peers=5):
    """BUY the fastest revenue growers *within their own sector*: the top
    ``top`` fraction of ``measure`` among the sector's eligible stocks that
    day (optionally only with positive operating margin). Strength = the
    within-sector percentile, so each sector contributes its own leaders
    instead of the whole list coming from the fastest-growing sector."""
    vals = fund[measure].where(elig)
    if margin:
        vals = vals.where(fund["op_margin"] > 0)
    key = sectors.reindex(vals.columns).to_numpy()
    pct = vals.T.groupby(key).rank(pct=True).T
    cnt = vals.notna().T.groupby(key).transform("sum").T
    flag = (pct >= 1 - top) & (cnt >= min_peers)
    return _finish(pct.where(flag), elig)


def cap_rank_matrix(md):
    """Dated screened cap rank (1 = largest) for an entry after row t."""
    rank = (md.membership.pivot_table(index="quarter_start", columns="symbol", values="screened_rank",
                                      aggfunc="min").reindex(columns=md.px.columns))
    return rank.reindex(_entry_quarter(md.calendar)).to_numpy(dtype="float32")


def mega_leading(group_rs_frame, threshold=0.0):
    """True at rows where the mega-cap bucket (group 0) has beaten the
    universe over the lookback (summed log excess return > threshold)."""
    return (group_rs_frame[0] > threshold).to_numpy()


def apply_mega_tilt(M, cap_rank, elig, leading, w, add):
    """When mega caps lead, rank larger companies higher (score = percentile
    among candidates + w x size percentile); with ``add`` the top MEGA_CUT
    companies also become candidates. Otherwise the ranking is unchanged."""
    E = np.asarray(elig)
    size_pct = pd.DataFrame(np.where(E, -cap_rank, np.nan)).rank(axis=1, pct=True).to_numpy()
    p = pct_among_flagged(M)
    if add:
        mega = E & (cap_rank <= MEGA_CUT) & leading[:, None]
        p = np.where(np.isnan(p) & mega, 0.5, p)
    out = np.where(leading[:, None], p + w * np.nan_to_num(size_pct), p)
    return out.astype("float32")


def trailing_strategy_weights(events, end_rows, T, lookback, mode, min_obs=20):
    """Strategy momentum: weight each strategy by the mean 90-day excess
    return of its signals over the last ``lookback`` signal rows whose
    outcomes are already known at row t (end_row < t).

    prop : weights proportional to the positive trailing means
    best : all weight on the strategy with the best trailing mean
    If no strategy has a usable positive mean, weights are equal.
    """
    names = list(events)
    finished = np.searchsorted(end_rows, np.arange(T), side="left")      # rows < this are known at t
    means = {}
    for n in names:
        v = np.asarray(events[n], dtype=float)
        cs = np.r_[0.0, np.cumsum(np.nan_to_num(v))]
        cn = np.r_[0, np.cumsum(np.isfinite(v))]
        hi = finished
        lo = np.maximum(hi - lookback, 0)
        cnt = cn[hi] - cn[lo]
        means[n] = np.where(cnt >= min_obs, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), np.nan)
    Mx = np.vstack([means[n] for n in names])                 # S x T
    if mode == "best":
        filled = np.where(np.isfinite(Mx), Mx, -np.inf)
        W = (np.arange(len(names))[:, None] == filled.argmax(0)[None, :]).astype("float32")
        W[:, ~np.isfinite(filled.max(0))] = 1.0 / len(names)
    else:
        pos = np.clip(np.nan_to_num(Mx, nan=0.0), 0, None)
        tot = pos.sum(0)
        W = np.where(tot > 0, pos / np.where(tot > 0, tot, 1), 1.0 / len(names)).astype("float32")
    return {n: W[i] for i, n in enumerate(names)}


def combine_weighted(P, F, W):
    """Composite of precomputed per-strategy percentiles ``P`` (flags ``F``)
    with per-row weights ``W``: sum of weight x percentile, plus one point per
    agreeing strategy with positive weight. Candidates = names flagged by a
    strategy whose weight is positive that day."""
    total, cand = None, None
    for n, w in W.items():
        wc = w[:, None]
        live = F[n] & (wc > 0)
        c = wc * np.nan_to_num(P[n]) + live
        total = c if total is None else total + c
        cand = live if cand is None else cand | live
    return np.where(cand, total, np.nan).astype("float32")


# ---------------------------------------------------------------------------
# 11d. Data-quality report
# ---------------------------------------------------------------------------
def data_quality_report(database, kept, all_members, md, elig, fund):
    """Count every data issue found, and how the pipeline handles it."""
    syms = list(md.px.columns)
    rows = []

    def add(area, issue, count, handling):
        rows.append(dict(area=area, issue=issue, count=count, handling=handling))

    top = all_members.loc[all_members.universe_rank.le(TOP_N)]
    removed = top.loc[~top.set_index(["quarter_start", "symbol"]).index.isin(
        kept.set_index(["quarter_start", "symbol"]).index)]
    add("universe", "raw top-500 rows with implausible cap proxy / thin trading (e.g. TMBR $28.7T 2016-Q2, "
        "APDN, SPAC AAM, BRK-A)", f"{len(removed):,} of {len(top):,} ({len(removed) / len(top):.1%})",
        "screen: dollar-volume rank <= 1000, >= 40 trading days, cap/ADV <= 3000")
    add("universe", "cap proxy mixes split-adjusted prices with differently-based share counts "
        "(BKNG, NFLX, ORLY ranks inflated)", "not countable", "legit large caps kept; rank order approximate")
    with connect(database) as con:
        q = lambda sql, params=(): con.execute(sql, list(params)).fetchone()[0]
        add("prices", "zero / invalid adjusted closes (whole database; BHRB 2000-01)",
            q("SELECT count(*) FROM prices WHERE adj_close IS NULL OR adj_close <= 0"),
            "quarantined, never filled")
        add("prices", "price rows on dates missing from the market calendar (whole database)",
            q("SELECT count(*) FROM prices p ANTI JOIN calendar c ON p.date = c.date"), "quarantined")
        add("fundamentals", "filing date missing or not after fiscal period end (whole database)",
            q("SELECT count(*) FROM fundamentals WHERE filing_date IS NULL OR filing_date <= fiscal_date"),
            "row excluded (cannot date its availability)")
        add("fundamentals", "eps_diluted populated (whole database)",
            f"{q('SELECT avg((eps_diluted IS NOT NULL)::int) FROM fundamentals'):.2%} of rows",
            "basic EPS used for the EPS filter")
        add("fundamentals", "market_cap missing (whole database)",
            f"{q('SELECT avg((market_cap IS NULL)::int) FROM fundamentals'):.1%} of rows",
            "not used; FCF yield = TTM FCF / (close x reported shares)")
        add("fundamentals", "reported_currency missing (whole database)",
            f"{q('SELECT avg((reported_currency IS NULL)::int) FROM fundamentals'):.1%} of rows",
            "ratios only (same-statement), so currency cancels")
        add("fundamentals", "values are the vendor's latest (restated) figures, not as-filed vintages",
            "all rows", "filing-date timing enforced; restatement look-ahead remains (limitation)")
        inactive = q("SELECT count(*) FROM universe WHERE symbol IN (SELECT unnest(?)) AND is_active_snapshot = 0",
                     [syms])
        delisted = q("SELECT count(*) FROM universe WHERE symbol IN (SELECT unnest(?)) "
                     "AND delisted_at_snapshot IS NOT NULL", [syms])
    add("survivorship", "universe stocks currently marked inactive (acquired / delisted / renamed)",
        f"{inactive} of {len(syms)}", "kept (no survivor filter); prices stop when quotes stop")
    add("survivorship", "inactive stocks with a recorded delisting date / proceeds", f"{delisted} / no proceeds table",
        "never infer a payout: held lots are marked at last price and flagged; write-off sensitivity reported")
    rp = md.repairs if getattr(md, "repairs", None) is not None else pd.DataFrame(columns=["symbol", "kind", "move"])
    for kind_, text in (("level shift", "one-day price moves below -70% or above +150% in BOTH adjusted and raw "
                         "close: unadjusted splits / reverse splits (NFLX -90%, ORLY -93%, BKNG -96%, KLAC -89%, "
                         "NOW -80%, TSCO -80%, LVLT +1,390%, AAM $0.44 -> $25,500)"),
                        ("round trip", "a move beyond +-50% reversed within 10 quotes (e.g. HON +99% then -51%)")):
        sub = rp.loc[rp.kind.eq(kind_)]
        add("prices", text, f"{len(sub)} days in {sub.symbol.nunique()} stocks",
            "repaired at load: the move is set to 0 by rescaling earlier prices (held lots no longer book fake "
            "-90% losses or +1,000% gains); no new buys for 252 sessions after. A few real crashes (NVAX -85% "
            "2016) are neutralized too -- small optimistic bias")
    dup = (kept.merge(md.info.name.rename("name"), left_on="symbol", right_index=True)
           .groupby(["quarter_start", "name"]).symbol.nunique())
    add("universe", "quarterly top-500 lists containing 2+ listings of one company (before the fix below)",
        f"{int((dup > 1).sum())} company-quarters", "see the one-company-one-position rule")
    dp = getattr(md, "duplicate_pairs", [])
    series = [f"{a}/{b}" for a, b, w in dp if w == "same series"]
    named = [f"{a}/{b}" for a, b, w in dp if w == "same name"]
    add("prices", "same security stored under an old and a new ticker (returns correlate >= 0.99)",
        f"{len(series)} pairs ({', '.join(series)})",
        "one company, one position: only the alphabetically first listing is buyable while the trailing "
        "63-day correlation is >= 0.99; the old ticker's quotes can stop at the rename (stuck lot)")
    add("universe", "share classes / same-name listings of one company",
        f"{len(named)} pairs ({', '.join(named[:12])}{', ...' if len(named) > 12 else ''})",
        "one company, one position: only one listing is buyable on a given day")
    add("classification", "sector/industry are a current snapshot, not dated history", "all stocks",
        "used for sector cap, pair and revenue peer groups (small look-ahead, limitation)")
    E = elig.to_numpy()
    for c in ["rev_yoy", "fcf_growth_q", "share_growth", "fcf_yield"]:
        cov = np.isfinite(fund[c].to_numpy()[E]).mean()
        add("fundamentals", f"coverage of {c} on buyable stock-days", f"{cov:.1%}",
            "missing -> stock simply has no FCF/revenue signal that day")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 11e. Charts
# ---------------------------------------------------------------------------
def make_charts(out_dir, curves, bench_curve, validation_curves, horizon_tab, vol_tab, event_tab):
    """Write the performance and research charts as PNG files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(exist_ok=True)
    pct = matplotlib.ticker.PercentFormatter(1.0)
    colors = ["#1f77b4", "#9467bd", "#2ca02c", "#d62728", "#8c564b", "#e377c2", "#17becf"]

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=120)
        plt.close(fig)

    # 1. drawdowns
    fig, ax = plt.subplots(figsize=(10, 4))
    for (label, v), c in zip(curves.items(), colors):
        ax.plot(v.index, v / v.cummax() - 1, label=label, color=c, lw=1)
    ax.plot(bench_curve.index, bench_curve / bench_curve.cummax() - 1, label="^SP500TR", color="#ff7f0e", lw=1)
    ax.yaxis.set_major_formatter(pct); ax.set_title("Drawdown from previous peak (test period)")
    ax.grid(alpha=0.3); ax.legend(); save(fig, "drawdowns.png")

    # 2. rolling 12-month return and 63-day volatility
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for (label, v), c in zip(list(curves.items()) + [("^SP500TR", bench_curve)], colors[:len(curves)] + ["#ff7f0e"]):
        axes[0].plot(v.index, v / v.shift(252) - 1, label=label, color=c, lw=1)
        axes[1].plot(v.index, v.pct_change().rolling(63).std() * np.sqrt(252), label=label, color=c, lw=1)
    axes[0].set_title("Rolling 12-month return"); axes[1].set_title("Rolling 63-day annualized volatility")
    for a in axes:
        a.yaxis.set_major_formatter(pct); a.grid(alpha=0.3)
    axes[0].legend(); save(fig, "rolling_return_and_volatility.png")

    # 3. calendar-year returns
    def yearly(v):
        y = v.resample("YE").last()
        y = pd.concat([pd.Series([INITIAL_CAPITAL], index=[v.index[0] - pd.Timedelta(days=1)]), y])
        r = y.pct_change().dropna()
        r.index = r.index.year
        return r
    table = pd.DataFrame({**{k: yearly(v) for k, v in curves.items()}, "^SP500TR": yearly(bench_curve)})
    table.to_csv(OUTPUT_DIR / "test_annual_returns.csv")
    pd.DataFrame({**curves, "^SP500TR": bench_curve}).to_csv(OUTPUT_DIR / "test_curves.csv")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    width = 0.8 / table.shape[1]
    for i, (col, c) in enumerate(zip(table.columns, colors[:len(curves)] + ["#ff7f0e"])):
        ax.bar(np.arange(len(table)) + i * width, table[col], width, label=col, color=c)
    ax.set_xticks(np.arange(len(table)) + width * (table.shape[1] - 1) / 2, table.index)
    ax.yaxis.set_major_formatter(pct); ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Calendar-year returns (2016 and 2026 partial)"); ax.grid(alpha=0.3, axis="y"); ax.legend()
    save(fig, "annual_returns.png")

    # 4. validation equity curves of every single strategy
    fig, ax = plt.subplots(figsize=(10, 5))
    for (label, v), c in zip(validation_curves.items(), itertools.cycle(colors + ["#7f7f7f", "#bcbd22"])):
        style = dict(color="#ff7f0e", lw=2) if label == "^SP500TR" else dict(color=c, lw=1.2)
        if label.startswith("random"):
            style = dict(color="grey", lw=1.2, ls="--")
        ax.plot(v.index, v / INITIAL_CAPITAL - 1, label=label, **style)
    ax.yaxis.set_major_formatter(pct); ax.set_title("Validation 2013-2015: each strategy alone (K=20, fixed params)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2); save(fig, "validation_strategies.png")

    # 5. event study: train vs validation excess return of the chosen parameters
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(event_tab))
    ax.bar(x - 0.2, event_tab.train_mean_excess, 0.4, label="train 2010-12")
    ax.bar(x + 0.2, event_tab.valid_mean_excess, 0.4, label="validation 2013-15")
    ax.set_xticks(x, event_tab.strategy); ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.set_title("Mean 90-day return of flagged stocks minus universe mean (chosen parameters)")
    ax.grid(alpha=0.3, axis="y"); ax.legend(); save(fig, "event_study_train_vs_validation.png")

    # 6. holding-horizon research
    fig, ax = plt.subplots(figsize=(9, 4))
    for (name, row), c in zip(horizon_tab.iterrows(), colors):
        ax.plot(range(len(row)), row.to_numpy(), marker="o", label=name, color=c)
    ax.set_xticks(range(len(horizon_tab.columns)), horizon_tab.columns)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1)); ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Excess return by holding horizon (trading days, 2010-2015, research only)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8); save(fig, "holding_horizons.png")

    # 7. volatility component: risk vs risk-adjusted return on validation
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(vol_tab.valid_vol, vol_tab.valid_sharpe, c=["red" if c else "#1f77b4" for c in vol_tab.chosen])
    for r in vol_tab.itertuples():
        ax.annotate(r.label, (r.valid_vol, r.valid_sharpe), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.xaxis.set_major_formatter(pct)
    ax.set_xlabel("validation volatility (2013-2015)"); ax.set_ylabel("validation Sharpe")
    ax.set_title("Volatility component settings (red = chosen)"); ax.grid(alpha=0.3)
    save(fig, "volatility_component.png")


def plot_dispersion(out_dir, disp, stock_vol, index_vol, patterns):
    """Dispersion over time, and each strategy's excess return by regime."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pct = matplotlib.ticker.PercentFormatter(1.0)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    ax = axes[0]
    ax.plot(stock_vol.index, stock_vol, lw=0.8, label="average stock volatility (63d)")
    ax.plot(index_vol.index, index_vol, lw=0.8, label="^SP500TR volatility (63d)")
    ax.fill_between(disp.index, 0, disp, color="grey", alpha=0.3, label="dispersion = difference")
    ax.axvspan(pd.Timestamp(TRAIN[0]), pd.Timestamp(VALID[1]), color="#2ca02c", alpha=0.06, label="development 2010-15")
    ax.yaxis.set_major_formatter(pct); ax.set_xlim(pd.Timestamp(TRAIN[0]), disp.index[-1])
    ax.set_ylim(0, 0.8); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Dispersion: single-stock volatility vs index volatility")
    v = patterns.loc[(patterns.window == 63) & (patterns.buckets == 2)]
    names_ = list(dict.fromkeys(v.strategy))
    x = np.arange(len(names_))
    ax = axes[1]
    for i, (col, period, b) in enumerate((("train_mean", "2010-12", "low"), ("train_mean", "2010-12", "high"),
                                          ("valid_mean", "2013-15", "low"), ("valid_mean", "2013-15", "high"))):
        vals = [v.loc[(v.strategy == n) & (v.bucket == b), col].iloc[0] for n in names_]
        ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, label=f"{period}, {b} dispersion",
               color=["#aec7e8", "#1f77b4", "#ffbb78", "#ff7f0e"][i])
    ax.set_xticks(x, names_); ax.axhline(0, color="black", lw=0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.set_title("Mean 90-day excess return of each strategy's signals, by dispersion regime")
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "dispersion.png", dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 12. Main experiment
# ---------------------------------------------------------------------------
def first_session_on_or_after(calendar, date):
    calendar = pd.DatetimeIndex(calendar)
    return calendar[calendar.searchsorted(pd.Timestamp(date))]


def previous_session(calendar, date):
    calendar = pd.DatetimeIndex(calendar)
    return calendar[calendar.searchsorted(pd.Timestamp(date)) - 1]


def run_candidate(px, matrix, cal, columns, k, start_entry, end, bench, vol=None, sectors=None, sector_cap=None,
                  budget_scale=None, renew=False, renew_rank=None, size_mult=None):
    """Backtest one score matrix over [start_entry, end]; returns metrics."""
    signal_start = cal[cal.get_loc(start_entry) - 1]
    scores = scores_from_matrix(matrix, cal, columns, signal_start, end)
    vol_map = None
    if vol is not None:
        lo, hi = cal.get_loc(signal_start), cal.get_loc(pd.Timestamp(end)) if pd.Timestamp(end) in cal else len(cal) - 1
        vol_map = {cal[t]: pd.Series(vol[t], index=columns).dropna() for t in range(lo, hi + 1)}
    mult_map = None
    if size_mult is not None:
        lo = cal.get_loc(signal_start)
        hi = cal.get_loc(pd.Timestamp(end)) if pd.Timestamp(end) in cal else len(cal) - 1
        mult_map = {}
        for t in range(lo, hi + 1):
            j = np.flatnonzero(size_mult[t] != 1.0)
            if len(j):
                mult_map[cal[t]] = pd.Series(size_mult[t, j], index=columns[j])
    strategy = ScoreStrategy(scores, k, max_new=max(1, math.ceil(k / 20)), vol=vol_map, size_mult=mult_map,
                             sectors=sectors, sector_cap=sector_cap, budget_scale=budget_scale,
                             renew_rank=renew_rank)
    result = Backtester(px, renew=renew).run(strategy, start_entry, end)
    m = performance(result.history.value, start_date=signal_start)
    b = performance(benchmark_values(bench, result.history.index), start_date=signal_start)
    yearly = result.history.value.resample("YE").last()
    yearly = pd.concat([pd.Series([INITIAL_CAPITAL], index=[signal_start]), yearly]).pct_change().dropna()
    byear = benchmark_values(bench, result.history.index).resample("YE").last()
    byear = pd.concat([pd.Series([INITIAL_CAPITAL], index=[signal_start]), byear]).pct_change().dropna()
    beat_years = int((yearly.to_numpy() > byear.to_numpy()).sum())
    return result, m, b, beat_years


def main(database, dev_only=False):
    t_start = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    np.seterr(all="ignore")

    # ---- Stage 1: data -----------------------------------------------------
    kept, all_members = load_universe(database)
    report_universe(kept, all_members)
    md = load_market_data(database, kept)
    cal, cols = md.calendar, md.px.columns
    end = min(cal[-1], pd.Timestamp(TEST_END))
    elig = eligibility(md)
    feat = Features(md)
    fund = fundamentals_wide(fundamental_events(md.fundamentals), md)
    labels = Labels(md, elig)
    pairs = PairsEngine(md, feat, elig)
    log(f"Data ready: {md.px.shape[1]} stocks x {len(cal)} sessions "
        f"({cal[0]:%Y-%m-%d} to {cal[-1]:%Y-%m-%d})")
    dq = data_quality_report(database, kept, all_members, md, elig, fund)
    dq.to_csv(OUTPUT_DIR / "data_quality.csv", index=False)
    print("\nData-quality report (see outputs/data_quality.csv):")
    for r in dq.itertuples():
        print(f"  [{r.area}] {r.issue}: {r.count} -> {r.handling}")

    specs = [
        StrategySpec("rsi_oversold", RSI_GRID, lambda p: generate_rsi_signals(feat, elig, **p)),
        StrategySpec("pairs", PAIRS_GRID, lambda p: pairs.signals(**p, eps_ok=fund["eps_ok"])),
        StrategySpec("fcf", FCF_GRID, lambda p: generate_fcf_signals(fund, elig, **p)),
        StrategySpec("revenue", REVENUE_GRID,
                     lambda p: generate_revenue_momentum_signals(fund, elig, md.info.industry, **p)),
        StrategySpec("rsi_reversal", REVERSAL_GRID, lambda p: generate_rsi_buy_signals(feat, elig, **p)),
        StrategySpec("momentum", MOMENTUM_GRID, lambda p: generate_momentum_signals(feat, elig, **p)),
    ]
    for s in specs:
        s.columns = cols

    valid_fits = [cal.searchsorted(pd.Timestamp(f"{y}-01-01")) for y in range(2013, 2016)]
    first_entry = first_session_on_or_after(cal, TEST_START)
    first_signal_row = cal.get_loc(first_entry) - 1
    test_fits = [first_signal_row] + [cal.searchsorted(pd.Timestamp(f"{y}-04-01")) - 1
                                      for y in range(first_entry.year + 1, end.year + 1)
                                      if pd.Timestamp(f"{y}-04-01") <= end]
    all_fits = valid_fits + test_fits

    # ---- Stage 2-3 & 5-6: signals, event studies, walk-forward choice ------
    log("Stage 2-6: parameter research (event studies on 2010-2015 only)")
    tables, choices, cache = [], {}, {}
    for spec in specs:
        t0 = time.time()
        table, ch, _ = research_strategy(spec, labels, cal, all_fits)
        tables.append(table)
        choices[spec.name] = ch
        log(f"  {spec.name}: {len(spec.grid)} configs x {len(WAITS)} waits in {time.time() - t0:.0f}s")
    research = pd.concat(tables, ignore_index=True)
    research.to_csv(OUTPUT_DIR / "research_all_configs.csv", index=False)

    # Fixed parameters = the walk-forward choice at the first validation fit
    # (trailing 2010-2012 outcomes completed before 2013-01-02).
    fixed, fixed_rows = {}, []
    print("\nChosen parameters (event study, train 2010-2012; t-stat adjusted for overlap):")
    for spec in specs:
        i, w, st = choices[spec.name][valid_fits[0]]
        fixed[spec.name] = (i, w)
        row = research.loc[(research.strategy == spec.name) & (research.config == i) & (research.wait == w)].iloc[0]
        fixed_rows.append(dict(strategy=spec.name, params=spec.grid[i], wait=w,
                               train_mean_excess=row.train_mean, train_t=row.train_t,
                               valid_mean_excess=row.valid_mean, valid_t=row.valid_t,
                               signals_per_day=row.valid_n_signals / max(row.valid_n_dates, 1)))
        wf = ", ".join(f"{cal[f]:%Y}: cfg {c[0]}/w{c[1]}" for f, c in choices[spec.name].items() if c)
        print(f"  {spec.name:13s} {spec.grid[i]} wait={w}  train {row.train_mean:+.2%} (t={row.train_t:.2f})"
              f"  valid {row.valid_mean:+.2%} (t={row.valid_t:.2f})")
        print(f"  {'':13s} walk-forward choices -> {wf}")
    pd.DataFrame(fixed_rows).to_csv(OUTPUT_DIR / "chosen_parameters.csv", index=False)

    fixed_mats, wf_mats, raw_mats = {}, {}, {}
    for spec in specs:
        i, w = fixed[spec.name]
        if (spec.name, i) not in cache:
            cache[(spec.name, i)] = spec.make(spec.grid[i])
        raw_mats[spec.name] = cache[(spec.name, i)]
        fixed_mats[spec.name] = shift_rows(raw_mats[spec.name], w)
        wf_mats[spec.name] = piecewise_matrix(spec, choices[spec.name], all_fits, len(cal), cache)

    # Research tables: waiting periods, holding horizons, exit rules, redundancy.
    dev_start, dev_end = TRAIN[0], VALID[1]
    wait_tab = research.loc[research.apply(lambda r: r.config == fixed[r.strategy][0], axis=1),
                            ["strategy", "wait", "train_mean", "train_t", "valid_mean", "valid_t"]]
    horizon_tab = pd.DataFrame({n: horizon_study(raw_mats[n], md, elig, dev_start, dev_end, fixed[n][1])
                                for n in raw_mats}).T
    exit_tab = pd.DataFrame({n: exit_rule_study(raw_mats[n], md, feat, elig, dev_start, dev_end, fixed[n][1])
                             for n in raw_mats}).T
    flags = {n: ~np.isnan(M[cal.searchsorted(pd.Timestamp(dev_start)):cal.searchsorted(pd.Timestamp(dev_end))])
             for n, M in raw_mats.items()}
    overlap = pd.DataFrame({a: {b: (flags[a] & flags[b]).sum() / max(flags[a].sum(), 1) for b in flags}
                            for a in flags})
    wait_tab.to_csv(OUTPUT_DIR / "research_waiting_periods.csv", index=False)
    horizon_tab.to_csv(OUTPUT_DIR / "research_holding_horizons.csv")
    exit_tab.to_csv(OUTPUT_DIR / "research_exit_rules.csv")
    overlap.to_csv(OUTPUT_DIR / "research_signal_overlap.csv")
    print("\nWaiting period (mean 90-day excess return, chosen params):")
    print(wait_tab.pivot(index="strategy", columns="wait", values="valid_mean").map(lambda x: f"{x:+.2%}").to_string())
    print("\nHolding horizon in trading days (mean excess return, 2010-2015, research only):")
    print(horizon_tab.map(lambda x: f"{x:+.2%}").to_string())
    print("\nExit rules on the same events (mean gross return, 2010-2015, research only):")
    print(exit_tab.map(lambda x: f"{x:+.2%}").to_string())
    print("\nSignal overlap (share of column strategy's signals also flagged by row strategy):")
    print(overlap.map(lambda x: f"{x:.0%}").to_string())

    # ---- Stage 7: model + combinations, validation backtests ----------------
    log("Stage 7: return model (walk-forward fits) and candidate backtests on 2013-2015")
    mats = model_feature_matrices(feat, fund, raw_mats)
    valid_end_row = cal.searchsorted(pd.Timestamp(VALID[1]), side="right") - 1
    model_valid, feature_names = model_scores(mats, labels, elig, cal, valid_fits, valid_end_row)

    train_t = {n: max(r["train_t"], 0.1) for n, r in
               pd.DataFrame(fixed_rows).set_index("strategy").iterrows()}
    names = [s.name for s in specs]
    candidates = {}
    for mode, src in (("fixed", fixed_mats), ("walkfwd", wf_mats)):
        for n in names:
            candidates[f"{n} [{mode}]"] = lambda src=src, n=n, mv=None: src[n]
    for r in range(2, len(names) + 1):
        for combo in itertools.combinations(names, r):
            label = "+".join(combo)
            candidates[f"{label} [equal]"] = (lambda c=combo: combine({n: fixed_mats[n] for n in c},
                                                                     {n: 1.0 for n in c}))
            candidates[f"{label} [t-weighted]"] = (lambda c=combo: combine({n: fixed_mats[n] for n in c},
                                                                          {n: train_t[n] for n in c}))
            candidates[f"{label} [+model fill]"] = (lambda c=combo, mv=None: ("fill", c))
    candidates["model (HGB)"] = lambda: "model"
    rng = np.random.default_rng(0)
    random_scores = np.where(elig.to_numpy(), rng.random(elig.shape), np.nan).astype("float32")
    candidates["baseline: random picks"] = lambda: random_scores

    def complexity(name):
        """Number of moving parts, used to prefer simpler candidates."""
        if name.startswith("baseline"):
            return 99
        if name.startswith("model"):
            return len(names) + 1
        n = name.split(" [")[0].count("+") + 1
        return n + (1 if "model fill" in name or "walkfwd" in name else 0)

    valid_entry = cal[valid_fits[0] + 1]
    rows, cache_m = [], {}

    def realize(name, model_matrix):
        built = candidates[name]()
        if isinstance(built, str):
            return model_matrix
        if isinstance(built, tuple):
            base = combine({n: fixed_mats[n] for n in built[1]}, {n: 1.0 for n in built[1]})
            return fill_with_model(base, model_matrix)
        return built

    validation_curves = {}
    for name in candidates:
        M = realize(name, model_valid)
        for k in K_GRID:
            res, m, b, beat = run_candidate(md.px, M, cal, cols, k, valid_entry, VALID[1], md.bench)
            if k == 20 and (name.endswith("[fixed]") or name.startswith("baseline")):
                validation_curves[name.replace(" [fixed]", "").replace("baseline: ", "")] = res.history.value
            ts = trade_stats(res)
            rows.append(dict(candidate=name, k=k, **m.to_dict(), beat_years=beat,
                             avg_invested=ts.get("Avg invested", np.nan),
                             excess_cagr=m["CAGR"] - b["CAGR"]))
    val = pd.DataFrame(rows).sort_values("Sharpe (rf 2%)", ascending=False)
    val.to_csv(OUTPUT_DIR / "validation_backtests.csv", index=False)
    bench_val = performance(benchmark_values(md.bench, cal[(cal >= valid_entry) & (cal <= pd.Timestamp(VALID[1]))]),
                            start_date=cal[valid_fits[0]])
    log(f"  {len(val)} validation backtests done")
    show = ["candidate", "k", "CAGR", "Sharpe (rf 2%)", "Max drawdown", "excess_cagr", "beat_years", "avg_invested"]
    print("\nTop 15 candidates on validation 2013-2015 (by Sharpe):")
    print(val[show].head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("Single strategies:")
    print(val.loc[val.candidate.str.contains(r"\[(?:fixed|walkfwd)\]")][show].to_string(index=False,
          float_format=lambda x: f"{x:.3f}"))
    print(f"Benchmark same window: CAGR {bench_val['CAGR']:.3f}, Sharpe {bench_val['Sharpe (rf 2%)']:.3f}, "
          f"MaxDD {bench_val['Max drawdown']:.3f}")

    # Selection rule (fixed before looking at results): among candidates within
    # SIMPLICITY_TOL Sharpe of the best, take the one with the fewest moving
    # parts; ties -> higher Sharpe. With ~370 backtests on 3 years, small
    # Sharpe differences are mostly noise, so simpler wins.
    pool = val.loc[~val.candidate.str.startswith("baseline")].copy()
    top_sharpe = pool["Sharpe (rf 2%)"].max()
    pool = pool.loc[pool["Sharpe (rf 2%)"] >= top_sharpe - SIMPLICITY_TOL]
    pool["complexity"] = pool.candidate.map(complexity)
    best = pool.sort_values(["complexity", "Sharpe (rf 2%)"], ascending=[True, False]).iloc[0]
    chosen, k = best.candidate, int(best.k)
    base_row = val.loc[val.candidate.str.startswith("baseline")].sort_values("Sharpe (rf 2%)").iloc[-1]
    print(f"Random-picks baseline: CAGR {base_row.CAGR:.3f}, Sharpe {base_row['Sharpe (rf 2%)']:.3f}")
    print(f"\nSelected signal set: {chosen}, K={k} (validation Sharpe {best['Sharpe (rf 2%)']:.2f}, "
          f"CAGR {best.CAGR:.2%}; best overall Sharpe was {top_sharpe:.2f})")

    # ---- Stage 8: portfolio risk controls (validation) ----------------------
    log("Stage 8: position sizing and sector caps on 2013-2015")
    vol63 = feat.get("vol", 63).to_numpy()
    sectors = md.info.sector
    M_valid = realize(chosen, model_valid)
    risk_rows = []
    for sizing in ("equal", "inverse-vol"):
        for cap in (None, 0.3):
            _, m, _, beat = run_candidate(md.px, M_valid, cal, cols, k, valid_entry, VALID[1], md.bench,
                                          vol=vol63 if sizing == "inverse-vol" else None,
                                          sectors=sectors, sector_cap=cap)
            risk_rows.append(dict(sizing=sizing, sector_cap=cap, **m.to_dict(), beat_years=beat,
                                  complexity=(sizing != "equal") + (cap is not None)))
    risk = pd.DataFrame(risk_rows)
    risk.to_csv(OUTPUT_DIR / "validation_risk_controls.csv", index=False)
    print(risk[["sizing", "sector_cap", "CAGR", "Sharpe (rf 2%)", "Max drawdown", "beat_years"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    rbest = risk.loc[risk["Sharpe (rf 2%)"] >= risk["Sharpe (rf 2%)"].max() - SIMPLICITY_TOL]
    rbest = rbest.sort_values(["complexity", "Sharpe (rf 2%)"], ascending=[True, False]).iloc[0]
    sizing, sector_cap = rbest.sizing, (None if pd.isna(rbest.sector_cap) else float(rbest.sector_cap))
    print(f"Risk controls: sizing={sizing}, sector cap={sector_cap}")

    # ---- Stage 9: sector / size regime overlay (train + validation) --------
    # Adoption rule fixed in advance: adopt the best overlay only if it beats
    # the no-overlay validation Sharpe by > SIMPLICITY_TOL and does not lower
    # the 2010-2012 Sharpe.
    log("Stage 9: sector/size regime overlay (lean into groups that start to outperform)")
    codes = group_codes(md)
    rs_cache = {}

    def group_rs(groups, L, mode):
        names_ = ["sector", "size"] if groups == "both" else [groups]
        out = []
        for g in names_:
            if (g, L, mode) not in rs_cache:
                rs_cache[(g, L, mode)] = group_relative_strength(feat.ret, elig, codes[g][0], L, mode)
            out.append(rs_cache[(g, L, mode)][0])
        return out

    train_entry = cal[cal.searchsorted(pd.Timestamp(TRAIN[0])) + 1]
    risk_kw = dict(vol=vol63 if sizing == "inverse-vol" else None, sectors=sectors, sector_cap=sector_cap)

    def two_windows(M):
        _, mv, _, bv_ = run_candidate(md.px, M, cal, cols, k, valid_entry, VALID[1], md.bench, **risk_kw)
        _, mt, _, _ = run_candidate(md.px, M, cal, cols, k, train_entry, TRAIN[1], md.bench, **risk_kw)
        return mv, mt, bv_

    base_v, base_t, _ = two_windows(M_valid)
    # Diagnostic: does group strength predict the next 90 days? (rank IC, dev)
    _, ex0 = labels.target(0)
    E = elig.to_numpy()
    ic_rows, reg_rows = [], []
    sample = np.flatnonzero((cal >= pd.Timestamp(TRAIN[0])) & (cal <= pd.Timestamp(VALID[1])))[::21]
    for g in ("sector", "size"):
        for L in (21, 63, 126):
            for mode in ("level", "turn"):
                x = group_rs(g, L, mode)[0]
                ics = [pd.Series(x[t, E[t]]).corr(pd.Series(ex0[t, E[t]]), method="spearman") for t in sample]
                ic_rows.append(dict(groups=g, lookback=L, mode=mode, dev_rank_ic=np.nanmean(ics)))
    for params in REGIME_GRID:
        Mr = regime_overlay(M_valid, group_rs(params["groups"], params["lookback"], params["mode"]),
                            params["weight"])
        mv, mt, _ = two_windows(Mr)
        reg_rows.append(dict(**params, valid_cagr=mv.CAGR, valid_sharpe=mv["Sharpe (rf 2%)"],
                             valid_maxdd=mv["Max drawdown"], train_cagr=mt.CAGR, train_sharpe=mt["Sharpe (rf 2%)"]))
    regime = pd.DataFrame(reg_rows).sort_values("valid_sharpe", ascending=False)
    regime_ic = pd.DataFrame(ic_rows)
    regime.to_csv(OUTPUT_DIR / "validation_regime_overlay.csv", index=False)
    regime_ic.to_csv(OUTPUT_DIR / "research_regime_ic.csv", index=False)
    print("Group relative strength -> next-90-day stock excess return (rank IC, 2010-2015, monthly):")
    print(regime_ic.pivot_table(index=["groups", "mode"], columns="lookback", values="dev_rank_ic")
          .map(lambda x: f"{x:+.4f}").to_string())
    print(f"No overlay: valid Sharpe {base_v['Sharpe (rf 2%)']:.3f} (CAGR {base_v.CAGR:.3f}), "
          f"train Sharpe {base_t['Sharpe (rf 2%)']:.3f} (CAGR {base_t.CAGR:.3f})")
    print("Top 10 overlays on validation:")
    print(regime.head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    top = regime.iloc[0]
    adopt = (top.valid_sharpe > base_v["Sharpe (rf 2%)"] + SIMPLICITY_TOL
             and top.train_sharpe >= base_t["Sharpe (rf 2%)"])
    regime_params = ({k_: top[k_] for k_ in ("groups", "lookback", "mode", "weight")} if adopt else None)
    if regime_params:
        regime_params["lookback"] = int(regime_params["lookback"])
        regime_params["weight"] = float(regime_params["weight"])
    print(f"Regime overlay {'ADOPTED: ' + str(regime_params) if adopt else 'NOT adopted'} "
          f"(best valid Sharpe {top.valid_sharpe:.3f} vs {base_v['Sharpe (rf 2%)']:.3f} + {SIMPLICITY_TOL} needed; "
          f"its train Sharpe {top.train_sharpe:.3f} vs {base_t['Sharpe (rf 2%)']:.3f})")

    # ---- Stage 10: volatility component (train + validation) --------------
    # Goal: lower total volatility. Rule fixed in advance: among settings that
    # lower volatility in BOTH 2010-12 and 2013-15 and keep both Sharpe ratios
    # within SIMPLICITY_TOL of the current configuration, take the lowest
    # validation volatility (ties -> higher Sharpe).
    log("Stage 10: volatility component (low-vol / low-beta tilts, vol filter, volatility targeting)")
    beta252 = feat.get("beta", 252).to_numpy()
    mkt_vol = (md.bench.pct_change().rolling(63).std() * np.sqrt(252)).to_numpy()
    E_np = elig.to_numpy()

    def with_regime(M):
        if not regime_params:
            return M
        return regime_overlay(M, group_rs(regime_params["groups"], regime_params["lookback"],
                                          regime_params["mode"]), regime_params["weight"])

    M_cur = with_regime(M_valid)
    vol_rows = []
    for params in VOL_GRID:
        Mv, scale = apply_vol_component(M_cur, params, vol63, beta252, E_np, mkt_vol, cal)
        _, mv, _, _ = run_candidate(md.px, Mv, cal, cols, k, valid_entry, VALID[1], md.bench,
                                    budget_scale=scale, **risk_kw)
        _, mt, _, _ = run_candidate(md.px, Mv, cal, cols, k, train_entry, TRAIN[1], md.bench,
                                    budget_scale=scale, **risk_kw)
        vol_rows.append(dict(label=describe_vol(params), params=params,
                             valid_vol=mv["Volatility (ann.)"], valid_sharpe=mv["Sharpe (rf 2%)"],
                             valid_cagr=mv.CAGR, valid_maxdd=mv["Max drawdown"],
                             train_vol=mt["Volatility (ann.)"], train_sharpe=mt["Sharpe (rf 2%)"],
                             train_cagr=mt.CAGR, train_maxdd=mt["Max drawdown"]))
    vol_tab = pd.DataFrame(vol_rows)
    base_vol = vol_tab.iloc[0]
    ok = ((vol_tab.valid_sharpe >= base_vol.valid_sharpe - SIMPLICITY_TOL)
          & (vol_tab.train_sharpe >= base_vol.train_sharpe - SIMPLICITY_TOL)
          & (vol_tab.valid_vol < base_vol.valid_vol) & (vol_tab.train_vol < base_vol.train_vol))
    pick = (vol_tab.loc[ok].sort_values(["valid_vol", "valid_sharpe"], ascending=[True, False]).iloc[0]
            if ok.any() else base_vol)
    vol_params = None if pick.params["kind"] == "none" else pick.params
    vol_tab["admissible"] = ok
    vol_tab["chosen"] = vol_tab.label.eq(pick.label)
    vol_tab.drop(columns="params").to_csv(OUTPUT_DIR / "validation_volatility_component.csv", index=False)
    print(vol_tab.drop(columns="params").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Volatility component chosen: {describe_vol(vol_params)}")

    # ---- Stage 11: 90-day review and growth/stable rotation ----------------
    # HW4 sells every lot at its 90-day maturity. The review decides, at that
    # close, whether to buy the same stock straight back as a new 90-day lot
    # (it still ranks in the top m*K) or to let it go. Then the growth /
    # stable rotation is tested as a replacement strategy. Both use the
    # Stage 9 adoption rule, fixed before running: adopt only if validation
    # Sharpe rises by > SIMPLICITY_TOL and the 2010-2012 Sharpe does not fall.
    log("Stage 11: 90-day review (sell or buy back at maturity) and growth/stable rotation")
    M_cur, scale_cur = apply_vol_component(M_cur, vol_params or {"kind": "none"}, vol63, beta252, E_np,
                                           mkt_vol, cal)

    def two_windows_renew(M, renew_rank, scale=None):
        kw = dict(budget_scale=scale, renew=renew_rank is not None, **risk_kw)
        out = []
        for a, b_ in ((valid_entry, VALID[1]), (train_entry, TRAIN[1])):
            res, m_, _, _ = run_candidate(md.px, M, cal, cols, k, a, b_, md.bench, renew_rank=renew_rank, **kw)
            buys = res.fills.loc[res.fills.side.eq("BUY")]
            m_["rebought at maturity"] = float(buys.groupby("symbol").date.diff().dt.days.le(HOLDING_DAYS + 7).mean())
            out.append(m_)
        return out

    review_rows = []
    for mult in (None,) + RENEW_MULTIPLES:
        rank = None if mult is None else mult * k
        mv, mt = two_windows_renew(M_cur, rank, scale_cur)
        review_rows.append(dict(review="off" if rank is None else f"buy back if in top {rank}", renew_rank=rank,
                                valid_sharpe=mv["Sharpe (rf 2%)"], valid_cagr=mv.CAGR,
                                valid_vol=mv["Volatility (ann.)"], valid_maxdd=mv["Max drawdown"],
                                train_sharpe=mt["Sharpe (rf 2%)"], train_cagr=mt.CAGR,
                                valid_rebought=mv["rebought at maturity"]))
    review = pd.DataFrame(review_rows)
    review.to_csv(OUTPUT_DIR / "validation_90day_review.csv", index=False)
    print(review.drop(columns="renew_rank").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    r_off, r_best = review.iloc[0], review.iloc[1:].sort_values("valid_sharpe", ascending=False).iloc[0]
    renew_rank = (int(r_best.renew_rank) if r_best.valid_sharpe > r_off.valid_sharpe + SIMPLICITY_TOL
                  and r_best.train_sharpe >= r_off.train_sharpe else None)
    cur_v = r_best if renew_rank else r_off
    print(f"90-day review {'ADOPTED: buy back if still in the top ' + str(renew_rank) if renew_rank else 'NOT adopted'}"
          f" (best {r_best.valid_sharpe:.3f} vs {r_off.valid_sharpe:.3f} + {SIMPLICITY_TOL} needed; "
          f"train {r_best.train_sharpe:.3f} vs {r_off.train_sharpe:.3f})")

    rot_rows = []
    for params in ROTATION_GRID:
        Mr = growth_rotation_scores(fund, md, elig, beta252, vol63, M_cur, **params)
        for mult in (None,) + RENEW_MULTIPLES:
            rank = None if mult is None else mult * k
            mv, mt = two_windows_renew(Mr, rank)
            rot_rows.append(dict(label=describe_rotation(params), params=params, renew_rank=rank,
                                 valid_sharpe=mv["Sharpe (rf 2%)"], valid_cagr=mv.CAGR,
                                 valid_vol=mv["Volatility (ann.)"], valid_maxdd=mv["Max drawdown"],
                                 train_sharpe=mt["Sharpe (rf 2%)"], train_cagr=mt.CAGR,
                                 train_vol=mt["Volatility (ann.)"]))
    rot = pd.DataFrame(rot_rows).sort_values("valid_sharpe", ascending=False)
    rot.drop(columns="params").to_csv(OUTPUT_DIR / "validation_growth_rotation.csv", index=False)
    print("Growth/stable rotation, top 10 of", len(rot), "on validation:")
    print(rot.drop(columns="params").head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    calm_on = rot.params.map(lambda p: p["calm"])
    print(f"  mean validation Sharpe with calm-market switch {rot.valid_sharpe[calm_on].mean():.3f}, "
          f"without {rot.valid_sharpe[~calm_on].mean():.3f}; train {rot.train_sharpe[calm_on].mean():.3f} vs "
          f"{rot.train_sharpe[~calm_on].mean():.3f}")
    rbest_rot = rot.iloc[0]
    rotation = None
    if (rbest_rot.valid_sharpe > cur_v.valid_sharpe + SIMPLICITY_TOL
            and rbest_rot.train_sharpe >= cur_v.train_sharpe):
        rotation = dict(rbest_rot.params)
        renew_rank = None if pd.isna(rbest_rot.renew_rank) else int(rbest_rot.renew_rank)
    rot_info = (dict(rbest_rot.params), None if pd.isna(rbest_rot.renew_rank) else int(rbest_rot.renew_rank))
    print(f"Growth rotation {'ADOPTED: ' + describe_rotation(rotation) if rotation else 'NOT adopted'} "
          f"(best valid Sharpe {rbest_rot.valid_sharpe:.3f} vs current {cur_v.valid_sharpe:.3f} + {SIMPLICITY_TOL} needed; "
          f"train {rbest_rot.train_sharpe:.3f} vs {cur_v.train_sharpe:.3f})")

    # ---- Stage 12: growth tilt (temporary overweight of fast growers) -----
    # Same adoption rule, against the current final configuration.
    log("Stage 12: growth tilt (overweight fast revenue growers in calm markets / after defensives lead)")
    growth, growth_pct = growth_names(fund, elig)
    tilt_rows, episode_cache = [], {}
    if rotation is None:
        for params in GROWTH_TILT_GRID:
            key = tuple((k_, v_) for k_, v_ in params.items() if k_ not in ("mult", "add"))
            if key not in episode_cache:
                episode_cache[key] = growth_tilt_episodes(md, elig, growth, **{k_: v_ for k_, v_ in key})
            active = episode_cache[key]
            Mt, size = apply_growth_tilt(M_cur, growth, growth_pct, active, params["mult"], params["add"])
            res_ = {}
            for nm, (a, b_) in (("valid", (valid_entry, VALID[1])), ("train", (train_entry, TRAIN[1]))):
                _, m_, _, _ = run_candidate(md.px, Mt, cal, cols, k, a, b_, md.bench, budget_scale=scale_cur,
                                            renew=renew_rank is not None, renew_rank=renew_rank,
                                            size_mult=size, **risk_kw)
                res_.update({f"{nm}_sharpe": m_["Sharpe (rf 2%)"], f"{nm}_cagr": m_.CAGR,
                             f"{nm}_vol": m_["Volatility (ann.)"], f"{nm}_maxdd": m_["Max drawdown"]})
            dev_rows = (cal >= pd.Timestamp(TRAIN[0])) & (cal <= pd.Timestamp(VALID[1]))
            tilt_rows.append(dict(label=describe_tilt(params), params=params,
                                  share_of_days_on=float(active[dev_rows].mean()), **res_))
        tilt = pd.DataFrame(tilt_rows).sort_values("valid_sharpe", ascending=False)
        tilt.drop(columns="params").to_csv(OUTPUT_DIR / "validation_growth_tilt.csv", index=False)
        print(f"Current configuration: valid Sharpe {cur_v.valid_sharpe:.3f}, train Sharpe {cur_v.train_sharpe:.3f}")
        print("Growth tilt, top 10 of", len(tilt), "on validation:")
        print(tilt.drop(columns="params").head(10).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        trig = tilt.params.map(lambda p_: p_["trigger"])
        print("  mean by trigger:", tilt.groupby(trig)[["valid_sharpe", "train_sharpe", "valid_vol"]].mean()
              .round(3).to_dict("index"))
        tbest = tilt.iloc[0]
        tilt_params = (dict(tbest.params) if tbest.valid_sharpe > cur_v.valid_sharpe + SIMPLICITY_TOL
                       and tbest.train_sharpe >= cur_v.train_sharpe else None)
        tilt_info = dict(tbest.params)
        print(f"Growth tilt {'ADOPTED: ' + describe_tilt(tilt_params) if tilt_params else 'NOT adopted'} "
              f"(best valid Sharpe {tbest.valid_sharpe:.3f} vs {cur_v.valid_sharpe:.3f} + {SIMPLICITY_TOL} needed; "
              f"train {tbest.train_sharpe:.3f} vs {cur_v.train_sharpe:.3f})")
    else:
        tilt_params, tilt_info = None, None
        print("Growth tilt skipped: the rotation replaced the strategy")

    # ---- Stage 13: dispersion -> lean into the strategies that suit it ------
    # Same adoption rule. Patterns are measured on train and validation
    # separately; the weights used in a backtest come only from outcomes that
    # finished before each fit date (expanding window from 2010).
    log("Stage 13: dispersion (stock vol vs index vol) as a signal to lean into strategies")
    disp_mats = {n: (wf_mats[n] if n == "fcf" else fixed_mats[n]) for n in names}
    events = {n: event_series(M_, labels, waits=(0,))[0][0] for n, M_ in disp_mats.items()}
    end0 = labels.end_row(0)
    start_row = cal.searchsorted(pd.Timestamp(TRAIN[0]))
    valid_lo, valid_hi = cal.searchsorted(pd.Timestamp(VALID[0])), cal.searchsorted(pd.Timestamp(VALID[1]), side="right")
    disp_cache, pattern_rows = {}, []
    for w_ in (21, 63):
        d_, sv_, iv_, pct_ = dispersion_series(md, elig, feat, w_)
        disp_cache[w_] = (d_, sv_, iv_, pct_)
        for nb in (2, 3):
            bk = dispersion_buckets(pct_, nb)
            t_tab = bucket_table(events, end0, bk, nb, start_row, valid_lo, first_signal_row)
            v_tab = bucket_table(events, end0, bk, nb, valid_lo, valid_hi, first_signal_row)
            for n in names:
                for j in range(nb):
                    pattern_rows.append(dict(window=w_, buckets=nb, strategy=n, bucket=BUCKET_NAMES[nb][j],
                                             train_mean=t_tab[n][j][0], train_t=t_tab[n][j][1], train_n=t_tab[n][j][2],
                                             valid_mean=v_tab[n][j][0], valid_t=v_tab[n][j][1], valid_n=v_tab[n][j][2]))
    patterns = pd.DataFrame(pattern_rows)
    patterns.to_csv(OUTPUT_DIR / "research_dispersion_patterns.csv", index=False)
    d63 = disp_cache[63][0]
    dev_mask = (cal >= pd.Timestamp(TRAIN[0])) & (cal <= pd.Timestamp(VALID[1]))
    print(f"Dispersion (63d avg stock vol - index vol), 2010-2015: mean {d63[dev_mask].mean():.1%}, "
          f"range {d63[dev_mask].min():.1%} to {d63[dev_mask].max():.1%}; "
          f"avg stock vol {disp_cache[63][1][dev_mask].mean():.1%} vs index {disp_cache[63][2][dev_mask].mean():.1%}")
    view = patterns.loc[(patterns.window == 63) & (patterns.buckets == 2)]
    spread = view.pivot(index="strategy", columns="bucket", values=["train_mean", "valid_mean"])
    consistent = pd.DataFrame({
        "train high-low": spread[("train_mean", "high")] - spread[("train_mean", "low")],
        "valid high-low": spread[("valid_mean", "high")] - spread[("valid_mean", "low")]})
    consistent["same sign"] = np.sign(consistent["train high-low"]) == np.sign(consistent["valid high-low"])
    print("Mean 90-day excess return by dispersion regime (63d, above/below 2-year median):")
    print(view.pivot(index="strategy", columns="bucket", values=["train_mean", "valid_mean"])
          .map(lambda x: f"{x:+.2%}").to_string())
    print(consistent.to_string(float_format=lambda x: f"{x:+.2%}"))
    for w_ in (21, 63):
        for nb in (2, 3):
            v_ = patterns.loc[(patterns.window == w_) & (patterns.buckets == nb)]
            sp = v_.groupby("strategy").apply(lambda g: pd.Series({
                "tr": g.train_mean.iloc[-1] - g.train_mean.iloc[0], "va": g.valid_mean.iloc[-1] - g.valid_mean.iloc[0]}),
                include_groups=False)
            print(f"  window {w_}, {nb} buckets: high-minus-low spread has the same sign in train and validation "
                  f"for {int((np.sign(sp.tr) == np.sign(sp.va)).sum())} of {len(sp)} strategies")

    disp_rows = []
    for params in DISPERSION_GRID:
        pct_ = disp_cache[params["window"]][3]
        bk = dispersion_buckets(pct_, params["buckets"])
        # validation fits only: --dev-only never reads a 2016+ outcome
        W = dispersion_weights(events, end0, bk, params["buckets"], valid_fits, len(cal), start_row, params["mode"])
        Md = combine_dynamic(disp_mats, W)
        Md, sc_ = apply_vol_component(Md, vol_params or {"kind": "none"}, vol63, beta252, E_np, mkt_vol, cal)
        res_ = {}
        for nm, (a, b_) in (("valid", (valid_entry, VALID[1])), ("train", (train_entry, TRAIN[1]))):
            _, m_, _, _ = run_candidate(md.px, Md, cal, cols, k, a, b_, md.bench, budget_scale=sc_,
                                        renew=renew_rank is not None, renew_rank=renew_rank, **risk_kw)
            res_.update({f"{nm}_sharpe": m_["Sharpe (rf 2%)"], f"{nm}_cagr": m_.CAGR,
                         f"{nm}_vol": m_["Volatility (ann.)"], f"{nm}_maxdd": m_["Max drawdown"]})
        vrows = slice(valid_lo, valid_hi)
        avg_w = {f"avg_weight_{n}": float(W[n][vrows].mean()) for n in names}
        disp_rows.append(dict(label=f"dispersion {params['window']}d, {params['buckets']} regimes, {params['mode']}",
                              params=params, **res_, **avg_w))
    disp = pd.DataFrame(disp_rows).sort_values("valid_sharpe", ascending=False)
    disp.drop(columns="params").to_csv(OUTPUT_DIR / "validation_dispersion.csv", index=False)
    print(f"Current configuration: valid Sharpe {cur_v.valid_sharpe:.3f}, train Sharpe {cur_v.train_sharpe:.3f}")
    print(disp.drop(columns="params").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    dbest = disp.iloc[0]
    disp_params = (dict(dbest.params) if rotation is None and tilt_params is None
                   and dbest.valid_sharpe > cur_v.valid_sharpe + SIMPLICITY_TOL
                   and dbest.train_sharpe >= cur_v.train_sharpe else None)
    disp_info = dict(dbest.params)
    print(f"Dispersion signal {'ADOPTED: ' + dbest.label if disp_params else 'NOT adopted'} "
          f"(best valid Sharpe {dbest.valid_sharpe:.3f} vs {cur_v.valid_sharpe:.3f} + {SIMPLICITY_TOL} needed; "
          f"train {dbest.train_sharpe:.3f} vs {cur_v.train_sharpe:.3f}; train is in-sample for the weights)")

    # ---- Stage 14: full factor search, no volatility optimization ----------
    # Rule fixed before running: score = min(train excess CAGR, validation
    # excess CAGR) over ^SP500TR, i.e. a candidate must beat the index in
    # BOTH development windows; no volatility or Sharpe criterion. Among
    # candidates within SEARCH_TOL of the best score, fewest parts wins.
    # The winner becomes the final strategy.
    log("Stage 14: full factor search (all signal combinations x weightings x overlays), return-based")
    t14 = time.time()
    spec7 = StrategySpec("sector_rev", SECTOR_REV_GRID,
                         lambda p_: generate_sector_revenue_leaders(fund, elig, md.info.sector, **p_))
    spec7.columns = cols
    table7, ch7, _ = research_strategy(spec7, labels, cal, all_fits)
    table7.to_csv(OUTPUT_DIR / "research_sector_revenue_leaders.csv", index=False)
    if ch7.get(valid_fits[0]):
        i7, w7, _ = ch7[valid_fits[0]]
    else:
        r_ = table7.sort_values("train_t", ascending=False).iloc[0]
        i7, w7 = int(r_.config), int(r_.wait)
    row7 = table7.loc[(table7.config == i7) & (table7.wait == w7)].iloc[0]
    print(f"Sector revenue leaders: chosen {SECTOR_REV_GRID[i7]} wait={w7}: train {row7.train_mean:+.2%} "
          f"(t={row7.train_t:.2f}), valid {row7.valid_mean:+.2%} (t={row7.valid_t:.2f}), "
          f"{row7.valid_n_signals / max(row7.valid_n_dates, 1):.0f} signals/day")
    sleeves = {n: (wf_mats[n] if n == "fcf" else fixed_mats[n]) for n in names}
    sleeves["sector_rev"] = shift_rows(spec7.make(SECTOR_REV_GRID[i7]), w7)
    snames = list(sleeves)
    sl_t = dict(train_t, sector_rev=max(float(row7.train_t), 0.1))
    P14 = {n: pct_among_flagged(M_) for n, M_ in sleeves.items()}
    F14 = {n: ~np.isnan(P14[n]) for n in snames}
    ev14 = {n: event_series(M_, labels, waits=(0,))[0][0] for n, M_ in sleeves.items()}
    T_ = len(cal)
    disp_bk = dispersion_buckets(disp_cache[63][3], 2)
    trail_cache = {}

    def weights_for(subset, method, fits):
        if method == "equal":
            return {n: np.ones(T_, "float32") for n in subset}
        if method == "t-weighted":
            return {n: np.full(T_, sl_t[n], "float32") for n in subset}
        if method.startswith("trailing"):
            L = int(method.split()[1][:-1])
            mode = "best" if method.endswith("best") else "prop"
            key = (subset, L, mode)
            if key not in trail_cache:
                trail_cache[key] = trailing_strategy_weights({n: ev14[n] for n in subset}, end0, T_, L, mode)
            return trail_cache[key]
        anchor = max(subset, key=lambda n: sl_t[n])
        return dispersion_weights({n: ev14[n] for n in subset}, end0, disp_bk, 2, fits, T_, start_row,
                                  "weights", anchor=anchor)

    def dev_eval(M_, size=None, renew=None):
        out = {}
        for nm, (a, b_) in (("train", (train_entry, TRAIN[1])), ("valid", (valid_entry, VALID[1]))):
            _, m_, b0, _ = run_candidate(md.px, M_, cal, cols, k, a, b_, md.bench, renew=renew is not None,
                                         renew_rank=renew, size_mult=size, **risk_kw)
            out.update({f"{nm}_cagr": m_.CAGR, f"{nm}_excess": m_.CAGR - b0.CAGR,
                        f"{nm}_sharpe": m_["Sharpe (rf 2%)"], f"{nm}_vol": m_["Volatility (ann.)"],
                        f"{nm}_maxdd": m_["Max drawdown"]})
        out["score"] = min(out["train_excess"], out["valid_excess"])
        return out

    # A: every subset of the 7 sleeves x every weighting method
    search_rows = []
    for r_ in range(1, len(snames) + 1):
        for subset in itertools.combinations(snames, r_):
            for method in (("single",) if r_ == 1 else SEARCH_METHODS):
                W_ = weights_for(subset, "equal" if method == "single" else method, valid_fits)
                Mc = combine_weighted(P14, F14, W_)
                search_rows.append(dict(sleeves="+".join(subset), method=method, n_sleeves=r_, **dev_eval(Mc)))
    sa = pd.DataFrame(search_rows).sort_values("score", ascending=False)
    sa.to_csv(OUTPUT_DIR / "search_signal_combinations.csv", index=False)
    log(f"  stage A: {len(sa)} signal combinations in {(time.time() - t14) / 60:.1f} min")
    cols_show = ["sleeves", "method", "score", "train_excess", "valid_excess", "train_cagr", "valid_cagr",
                 "valid_sharpe", "valid_vol"]
    print("Top 15 signal combinations (score = worse of train / validation excess CAGR):")
    print(sa[cols_show].head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("Average score by weighting method:", sa.groupby("method").score.mean().round(4).to_dict())
    top50 = sa.head(50)
    print("Share of the top 50 that include each sleeve:",
          {n: round(float(top50.sleeves.str.split("+").map(lambda l: n in l).mean()), 2) for n in snames})
    marg = {}
    for n in snames:
        has = sa.sleeves.str.split("+").map(lambda l: n in l)
        marg[n] = round(float(sa.score[has].mean() - sa.score[~has].mean()), 4)
    print("Average score with minus without each sleeve:", marg)

    # B: the best SEARCH_TOP combinations x every combination of overlays
    sector_opts = [None, dict(groups="sector", lookback=21, mode="turn", weight=0.5),
                   dict(groups="sector", lookback=63, mode="level", weight=0.5)]
    size_codes = codes["size"][0]
    cap_rank = cap_rank_matrix(md)
    mega_lead = {L: mega_leading(group_relative_strength(feat.ret, elig, size_codes, L, "level")[1])
                 for L in (63, 126)}
    tilt_opts = [None, dict(trigger="lowvol", mult=1.5, exit=0.05, add=True)] + ([tilt_info] if tilt_info else [])
    tilt_active = {}
    for tp_ in tilt_opts[1:]:
        key = tuple((k_, v_) for k_, v_ in tp_.items() if k_ not in ("mult", "add"))
        tilt_active[id(tp_)] = growth_tilt_episodes(md, elig, growth, **dict(key))

    build_cache = {}

    def build(spec_, fits):
        subset = tuple(spec_["sleeves"].split("+"))
        method = spec_["method"]
        key = (spec_["sleeves"], method, str(spec_.get("sector")), str(spec_.get("mega")), len(fits))
        if key not in build_cache:
            if len(build_cache) > 40:
                build_cache.clear()
            M_ = combine_weighted(P14, F14, weights_for(subset, "equal" if method == "single" else method, fits))
            if spec_.get("sector"):
                sp_ = spec_["sector"]
                M_ = regime_overlay(M_, group_rs(sp_["groups"], sp_["lookback"], sp_["mode"]), sp_["weight"])
            if spec_.get("mega"):
                mp_ = spec_["mega"]
                M_ = apply_mega_tilt(M_, cap_rank, E_np, mega_lead[mp_["lookback"]], mp_["w"], mp_["add"])
            build_cache[key] = M_
        M_ = build_cache[key]
        size = None
        if spec_.get("tilt"):
            tp_ = spec_["tilt"]
            M_, size = apply_growth_tilt(M_, growth, growth_pct, tilt_active[id(tp_)], tp_["mult"], tp_["add"])
        return M_, size

    def describe_spec(spec_):
        parts = [f"{spec_['sleeves']} [{spec_['method']}]"]
        if spec_.get("sector"):
            parts.append(f"sector overlay {spec_['sector']['lookback']}d {spec_['sector']['mode']}")
        if spec_.get("mega"):
            mp_ = spec_["mega"]
            parts.append(f"mega-cap tilt {mp_['lookback']}d w={mp_['w']}{' +adds' if mp_['add'] else ''}")
        if spec_.get("tilt"):
            parts.append("growth overweight " + spec_["tilt"]["trigger"])
        if spec_.get("renew"):
            parts.append(f"90-day review top {spec_['renew']}")
        return " | ".join(parts)

    base_rows = sa.head(SEARCH_TOP)[["sleeves", "method", "n_sleeves"]].to_dict("records")
    ob_rows = []
    for base in base_rows:
        for sp_ in sector_opts:
            for mp_ in [None] + MEGA_GRID:
                for tp_ in tilt_opts:
                    for rn in (None, 2 * k):
                        spec_ = dict(sleeves=base["sleeves"], method=base["method"], sector=sp_, mega=mp_,
                                     tilt=tp_, renew=rn)
                        M_, size = build(spec_, valid_fits)
                        n_over = sum(x is not None for x in (sp_, mp_, tp_, rn))
                        ob_rows.append(dict(spec=spec_, label=describe_spec(spec_), n_overlays=n_over,
                                            complexity=base["n_sleeves"] + (base["method"] not in ("single", "equal"))
                                            + n_over,
                                            sector_on=sp_ is not None, mega_on=mp_ is not None,
                                            tilt_on=tp_ is not None, review_on=rn is not None,
                                            **dev_eval(M_, size, rn)))
    sb = pd.DataFrame(ob_rows).sort_values("score", ascending=False)
    sb.drop(columns="spec").to_csv(OUTPUT_DIR / "search_with_overlays.csv", index=False)
    log(f"  stage B: {len(sb)} overlay combinations; total {(time.time() - t14) / 60:.1f} min")
    print("Top 15 with overlays:")
    print(sb[["label", "score", "train_excess", "valid_excess", "valid_cagr", "valid_sharpe", "valid_vol"]]
          .head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("Average score change when an overlay is on (stage B):",
          {o: round(float(sb.score[sb[o]].mean() - sb.score[~sb[o]].mean()), 4)
           for o in ("sector_on", "mega_on", "tilt_on", "review_on")})
    pool14 = sb.loc[sb.score >= sb.score.max() - SEARCH_TOL]
    win = pool14.sort_values(["complexity", "score"], ascending=[True, False]).iloc[0]
    final14 = win.spec
    print(f"\nSTAGE 14 WINNER (new final strategy): {win.label}\n"
          f"  train excess {win.train_excess:+.2%}/yr, validation excess {win.valid_excess:+.2%}/yr "
          f"(best score {sb.score.max():+.2%}; {len(pool14)} candidates within {SEARCH_TOL:.0%})")
    W_win = weights_for(tuple(final14["sleeves"].split("+")),
                        "equal" if final14["method"] == "single" else final14["method"], valid_fits)
    vr = slice(valid_lo, valid_hi)
    print("  average weights in validation: " + ", ".join(f"{n} {w_[vr].mean():.2f}" for n, w_ in W_win.items()))
    # Data check of the winner on development data: largest lots and concentration.
    M_w, size_w = build(final14, valid_fits)
    for nm, (a, b_) in (("train", (train_entry, TRAIN[1])), ("valid", (valid_entry, VALID[1]))):
        res_w, _, _, _ = run_candidate(md.px, M_w, cal, cols, k, a, b_, md.bench, renew=final14.get("renew") is not None,
                                       renew_rank=final14.get("renew"), size_mult=size_w, **risk_kw)
        f_ = res_w.fills
        sl_ = f_.loc[f_.side.eq("SELL")].copy()
        by_ = f_.loc[f_.side.eq("BUY")].set_index("lot_id")
        sl_["lot_return"] = sl_.price / by_.loc[sl_.lot_id, "price"].to_numpy() - 1
        top_share = sl_.nlargest(10, "pnl").pnl.sum() / max(sl_.pnl.sum(), 1)
        print(f"  {nm}: largest lots " + ", ".join(f"{r.symbol} {r.lot_return:+.0%}" for r in
                                                  sl_.nlargest(5, "lot_return").itertuples())
              + f"; top-10 lots = {top_share:.0%} of total PnL; {len(sl_)} lots")

    review_desc = f"buy back if in top {renew_rank}" if renew_rank else "off"
    print(f"\nFINAL CONFIGURATION: {chosen}, K={k}, sizing={sizing}, sector cap={sector_cap}, "
          f"regime overlay={regime_params}, volatility component={describe_vol(vol_params)}, "
          f"90-day review={review_desc}, rotation={describe_rotation(rotation) if rotation else 'none'}, "
          f"growth tilt={describe_tilt(tilt_params)}, dispersion={disp_params}")
    print(f"NEW FINAL (Stage 14, replaces the above): {describe_spec(final14)}")
    pd.Series(dict(candidate=chosen, k=k, sizing=sizing, sector_cap=sector_cap, regime=regime_params,
                   volatility=describe_vol(vol_params), review=review_desc,
                   rotation=describe_rotation(rotation) if rotation else "none", growth_tilt=describe_tilt(tilt_params), dispersion=disp_params, stage14_final=describe_spec(final14),
                   validation_sharpe=pick.valid_sharpe, validation_vol=pick.valid_vol,
                   validation_cagr=pick.valid_cagr)).to_csv(OUTPUT_DIR / "final_configuration.csv")
    if dev_only:
        log("--dev-only: stopping before the test period")
        return

    # ---- Final test: run once -----------------------------------------------
    log(f"Final test {first_entry:%Y-%m-%d} to {end:%Y-%m-%d} (single run)")
    needs_model = "model" in chosen
    model_test = None
    if needs_model:
        model_test, _ = model_scores(mats, labels, elig, cal, test_fits, cal.get_loc(end))
    M_base = with_regime(realize(chosen, model_test))
    curves = {}
    if vol_params:
        # The configuration before the volatility component (first test run),
        # shown for comparison only.
        prev, _, _, _ = run_candidate(md.px, M_base, cal, cols, k, first_entry, end, md.bench, **risk_kw)
        curves["without vol component" + (" or review" if renew_rank else "")] = prev.history.value
    M, scale = apply_vol_component(M_base, vol_params or {"kind": "none"}, vol63, beta252, E_np, mkt_vol, cal)
    if renew_rank and not rotation:
        prev, _, _, _ = run_candidate(md.px, M, cal, cols, k, first_entry, end, md.bench,
                                      budget_scale=scale, **risk_kw)
        curves["without 90-day review"] = prev.history.value
    rp, rr = rot_info
    M_rot = growth_rotation_scores(fund, md, elig, beta252, vol63, M, **rp)
    if rotation:
        M, scale = M_rot, None
    else:
        # The user's growth/stable rotation idea, best dev setting, NOT adopted:
        # shown for information only; it did not change the final choice.
        alt, _, _, _ = run_candidate(md.px, M_rot, cal, cols, k, first_entry, end, md.bench,
                                     renew=rr is not None, renew_rank=rr, **risk_kw)
        curves["growth rotation (not adopted)"] = alt.history.value
    size = None
    if tilt_info is not None:
        tp = {k_: v_ for k_, v_ in tilt_info.items() if k_ not in ("mult", "add")}
        M_tilt, size_tilt = apply_growth_tilt(M, growth, growth_pct, growth_tilt_episodes(md, elig, growth, **tp),
                                              tilt_info["mult"], tilt_info["add"])
        if tilt_params:
            M, size = M_tilt, size_tilt
        else:
            # Best dev setting of the growth tilt, NOT adopted: information only.
            alt, _, _, _ = run_candidate(md.px, M_tilt, cal, cols, k, first_entry, end, md.bench,
                                         budget_scale=scale, renew=renew_rank is not None,
                                         renew_rank=renew_rank, size_mult=size_tilt, **risk_kw)
            curves["growth tilt (not adopted)"] = alt.history.value
    bk = dispersion_buckets(disp_cache[disp_info["window"]][3], disp_info["buckets"])
    W_test = dispersion_weights(events, end0, bk, disp_info["buckets"], all_fits, len(cal), start_row, disp_info["mode"])
    M_disp, sc_disp = apply_vol_component(combine_dynamic(disp_mats, W_test), vol_params or {"kind": "none"},
                                          vol63, beta252, E_np, mkt_vol, cal)
    if disp_params:
        M, scale, size = M_disp, sc_disp, None
    else:
        # Best dev setting of the dispersion signal, NOT adopted: information only.
        alt, _, _, _ = run_candidate(md.px, M_disp, cal, cols, k, first_entry, end, md.bench, budget_scale=sc_disp,
                                     renew=renew_rank is not None, renew_rank=renew_rank, **risk_kw)
        curves["dispersion-weighted (not adopted)"] = alt.history.value
    test_rows = slice(first_signal_row, cal.get_loc(end) + 1)
    print("Dispersion weights in the test period (average): "
          + ", ".join(f"{n} {W_test[n][test_rows].mean():.2f}" for n in names))
    result, m, b, beat = run_candidate(md.px, M, cal, cols, k, first_entry, end, md.bench,
                                       budget_scale=scale, renew=renew_rank is not None,
                                       renew_rank=renew_rank, size_mult=size, **risk_kw)
    # Stage 14 winner = new final strategy; the previous final is kept for comparison.
    curves["Sharpe-based pipeline (stages 7-13)"] = result.history.value
    # The previously submitted final (walk-forward FCF + low-beta tilt), on the repaired prices.
    M_old, _ = apply_vol_component(wf_mats["fcf"], dict(kind="beta_tilt", w=0.5), vol63, beta252, E_np, mkt_vol, cal)
    old, _, _, _ = run_candidate(md.px, M_old, cal, cols, k, first_entry, end, md.bench, **risk_kw)
    curves["previous final (FCF + low-beta tilt)"] = old.history.value
    M14, size14 = build(final14, all_fits)
    result, m, b, beat = run_candidate(md.px, M14, cal, cols, k, first_entry, end, md.bench,
                                       renew=final14.get("renew") is not None, renew_rank=final14.get("renew"),
                                       size_mult=size14, **risk_kw)
    W_t = weights_for(tuple(final14["sleeves"].split("+")),
                      "equal" if final14["method"] == "single" else final14["method"], all_fits)
    yrs = pd.Series(cal.year, index=cal)
    tr_rows = (cal >= first_entry) & (cal <= end)
    wtab = pd.DataFrame({n: pd.Series(w_, index=cal)[tr_rows].groupby(yrs[tr_rows]).mean() for n, w_ in W_t.items()})
    for L_, lead in mega_lead.items():
        wtab[f"mega caps leading ({L_}d)"] = pd.Series(lead, index=cal)[tr_rows].groupby(yrs[tr_rows]).mean()
    wtab.to_csv(OUTPUT_DIR / "test_dynamic_weights_by_year.csv")
    print("Strategy weights and mega-cap leadership by test year (share of days):")
    print(wtab.to_string(float_format=lambda x: f"{x:.2f}"))
    sv = result.history.value
    bv = benchmark_values(md.bench, sv.index)
    first_signal = cal[first_signal_row]
    metrics = pd.DataFrame({"Strategy": performance(sv, start_date=first_signal),
                            "^SP500TR": performance(bv, start_date=first_signal)})
    for label_, v in curves.items():
        metrics[label_] = performance(v, start_date=first_signal)
    print(metrics.to_string(float_format=lambda x: f"{x:,.4f}"))
    diff = metrics.loc["Final value ($)", "Strategy"] - metrics.loc["Final value ($)", "^SP500TR"]
    print(f"{'Beat' if diff > 0 else 'Did not beat'} the benchmark: final value difference ${diff:,.0f}, "
          f"CAGR difference {metrics.loc['CAGR', 'Strategy'] - metrics.loc['CAGR', '^SP500TR']:+.2%}; "
          f"beat it in {beat} of {sv.index.year.nunique()} calendar years")
    if result.diagnostics["overdue_value"]:
        print(f"Sensitivity: writing off overdue lots with no quote -> final value "
              f"${sv.iloc[-1] - result.diagnostics['overdue_value']:,.0f}")
    ts = trade_stats(result)
    print("Trade statistics:")
    print(ts.to_string(float_format=lambda x: f"{x:,.3f}"))
    regimes = regime_table(sv, bv, md.bench)
    print("Market environments (annualized mean daily return):")
    print(regimes.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== Accounting / timing audit ===")
    audit(result, end=end)
    sells = result.fills.loc[result.fills.side.eq("SELL")].copy()
    buys = result.fills.loc[result.fills.side.eq("BUY")].set_index("lot_id")
    sells["lot_return"] = sells.price / buys.loc[sells.lot_id, "price"].to_numpy() - 1
    print("  Largest lot returns (check for data errors):")
    print(sells.nlargest(5, "lot_return")[["symbol", "entry_date", "date", "lot_return"]].to_string(index=False))
    print("  Holdings per stock never exceed one open lot:",
          bool(result.fills.loc[result.fills.side.eq("BUY")].groupby("symbol").date.diff().dropna()
               .dt.days.ge(HOLDING_DAYS).all()))

    # Daily signal report for the last test session (what the system says today)
    last = cal.get_loc(end)
    report = []
    for n in names:
        for j in np.flatnonzero(~np.isnan(fixed_mats[n][last])):
            report.append(dict(date=end, strategy=n, symbol=cols[j], name=md.info.name.iloc[j],
                               price=md.close.iloc[last, j], strength=fixed_mats[n][last, j],
                               rsi14=feat.get("rsi", 14).iloc[last, j], rev_yoy=fund["rev_yoy"].iloc[last, j],
                               fcf_growth=fund["fcf_growth_ttm"].iloc[last, j]))
    pd.DataFrame(report).to_csv(OUTPUT_DIR / "signals_latest_day.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "test_metrics.csv")
    ts.to_csv(OUTPUT_DIR / "test_trade_stats.csv")
    regimes.to_csv(OUTPUT_DIR / "test_regimes.csv", index=False)
    result.history.to_csv(OUTPUT_DIR / "test_history.csv")
    result.fills.to_csv(OUTPUT_DIR / "test_fills.csv", index=False)
    final_label = "Final: " + describe_spec(final14)
    if len(final_label) > 70:
        final_label = final_label[:67] + "..."
    shown = {k_: v_ for k_, v_ in curves.items() if "not adopted" not in k_}
    plot_curves(sv, bv, CHART_PATH, f"Strategy vs ^SP500TR, {first_entry:%Y-%m-%d} to {end:%Y-%m-%d}",
                label=final_label, extra=shown)
    validation_curves["^SP500TR"] = benchmark_values(md.bench, cal[(cal >= valid_entry) & (cal <= pd.Timestamp(VALID[1]))])
    make_charts(HW_DIR / "charts", {final_label: sv, **shown}, bv, validation_curves, horizon_tab,
                vol_tab, pd.DataFrame(fixed_rows))
    plot_dispersion(HW_DIR / "charts", *disp_cache[63][:3], patterns)
    log(f"Done in {(time.time() - t_start) / 60:.1f} min. Chart: {CHART_PATH}; tables: {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# --check: hand-verifiable tests of the backtester, metrics and universe
# ---------------------------------------------------------------------------
class FixedPicks:
    """Test strategy: buy given (symbol, budget) orders on chosen signal dates."""

    def __init__(self, plan_by_date):
        self.plan_by_date = plan_by_date

    def plan(self, signal_date, cash, nav, held):
        return self.plan_by_date.get(signal_date, [])


def self_check():
    cost = COST_BPS / 1e4
    cal = pd.bdate_range("2021-01-04", "2021-06-30")     # Mon-Fri sessions
    px = pd.DataFrame({"A": 10.0, "B": 20.0, "C": 5.0}, index=cal)
    px.loc[px.index >= "2021-04-05", "A"] = 12.0          # A rises before its exit
    px.loc["2021-01-06", "C"] = np.nan                    # C has no quote on entry day

    # Case 1: signal Mon 2021-01-04 -> buy A ($300k) and B ($200k) at Tue 01-05 close.
    # Both are due 01-05 + 90 days = Mon 2021-04-05 and sell at that close.
    strat = FixedPicks({pd.Timestamp("2021-01-04"): [("A", 300_000), ("B", 200_000)],
                        pd.Timestamp("2021-01-05"): [("C", 100_000)]})
    res = Backtester(px).run(strat, "2021-01-05", "2021-06-30")
    f = res.fills
    a_buy = f.loc[(f.symbol == "A") & (f.side == "BUY")].iloc[0]
    a_units = 300_000 / (10 * (1 + cost))
    assert a_buy.date == pd.Timestamp("2021-01-05") and np.isclose(a_buy.units, a_units)
    assert np.isclose(a_buy.units * a_buy.price + a_buy.fee, 300_000)       # budget includes fee
    # C's order (planned 01-05 for 01-06) is skipped: no quote, no invented fill
    assert "C" not in set(f.symbol) and res.diagnostics["skipped_orders"] == 1
    a_sell = f.loc[(f.symbol == "A") & (f.side == "SELL")].iloc[0]
    assert a_sell.date == pd.Timestamp("2021-04-05")
    assert np.isclose(a_sell.cash_change, a_units * 12 * (1 - cost))
    expected_cash = 1_000_000 - 500_000 + a_units * 12 * (1 - cost) + (200_000 / (1 + cost)) * (1 - cost)
    assert np.isclose(res.history.cash.iloc[-1], expected_cash)
    assert np.isclose(res.history.value.iloc[-1], expected_cash)            # everything sold
    fees_day1 = 300_000 * cost / (1 + cost) + 200_000 * cost / (1 + cost)
    assert np.isclose(res.history.value.iloc[0], 1_000_000 - fees_day1)

    # Case 2: missing quotes at maturity. Signal Wed 01-06 -> buy B Thu 01-07,
    # due Wed 2021-04-07. B is unquoted 04-07 and 04-08 -> sells Fri 04-09.
    px2 = px.copy()
    px2.loc[["2021-04-07", "2021-04-08"], "B"] = np.nan
    res2 = Backtester(px2).run(FixedPicks({pd.Timestamp("2021-01-06"): [("B", 100_000)]}),
                               "2021-01-05", "2021-06-30")
    b_sell = res2.fills.loc[res2.fills.side.eq("SELL")].iloc[0]
    assert b_sell.date == pd.Timestamp("2021-04-09")
    # Weekend maturity: signal Fri 01-08 -> buy Mon 01-11, due Sun 2021-04-11
    # -> sells at the next session, Mon 04-12.
    res3 = Backtester(px).run(FixedPicks({pd.Timestamp("2021-01-08"): [("A", 100_000)]}),
                              "2021-01-05", "2021-06-30")
    s3 = res3.fills.loc[res3.fills.side.eq("SELL")].iloc[0]
    assert pd.Timestamp("2021-04-11").day_name() == "Sunday"
    assert s3.date == pd.Timestamp("2021-04-12")

    # Case 3: never spend the same cash twice -- budgets larger than cash are capped.
    res4 = Backtester(px).run(FixedPicks({pd.Timestamp("2021-01-04"): [("A", 800_000), ("B", 800_000)]}),
                              "2021-01-05", "2021-02-26")
    assert res4.history.cash.min() >= -1e-6
    assert np.isclose(res4.fills.loc[res4.fills.side.eq("BUY")].cash_change.sum(), -1_000_000)

    # Case 4: a held stock that stops quoting is marked at its last price, flagged
    # stale, kept open (no invented exit), and reported as overdue.
    px5 = px.copy()
    px5.loc[px5.index >= "2021-03-01", "A"] = np.nan
    res5 = Backtester(px5).run(FixedPicks({pd.Timestamp("2021-01-04"): [("A", 100_000)]}),
                               "2021-01-05", "2021-06-30")
    assert res5.diagnostics["overdue_lots"] == 1 and "A" in res5.diagnostics["stale_symbols"]
    assert np.isclose(res5.open_lots.value.iloc[0], 100_000 / (1 + cost))   # marked at last price 10
    audit(res, end="2021-06-30")

    # Case 5: metrics on a series that doubles over 2020-01-01..2021-01-01 (366
    # days) with a 50% drawdown: CAGR = 2 ** (365.25 / 366) - 1.
    idx = pd.DatetimeIndex(["2020-01-01", "2020-06-01", "2021-01-01"])
    m = performance(pd.Series([1.0, 0.5, 2.0], index=idx), initial=1.0)
    assert np.isclose(m["CAGR"], 2 ** (365.25 / 366) - 1) and np.isclose(m["Max drawdown"], -0.5)
    assert np.isclose(m["Final value ($)"], 2.0) and np.isclose(m["Total return"], 1.0)

    # Case 6: strategy never re-buys held names, respects the weight cap, and
    # staggers entries with max_new.
    s = ScoreStrategy({pd.Timestamp("2021-01-04"): pd.Series([3, 2, 1.0], index=["A", "B", "C"])}, k=2)
    orders = s.plan(pd.Timestamp("2021-01-04"), cash=1_000_000, nav=1_000_000, held={"A"})
    assert [o[0] for o in orders] == ["B", "C"] and all(b == 100_000 for _, b in orders)
    s = ScoreStrategy({pd.Timestamp("2021-01-04"): pd.Series([3, 2, 1.0], index=["A", "B", "C"])}, k=20, max_new=1)
    assert [o[0] for o in s.plan(pd.Timestamp("2021-01-04"), 1e6, 1e6, set())] == ["A"]

    # Case 7: universe screen drops a name whose cap proxy is contradicted by
    # its trading volume, then keeps the top_n survivors in cap order.
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "fake.duckdb"
        con = duckdb.connect(str(db))
        con.execute("""CREATE TABLE universe_membership(quarter_start DATE, quarter_end DATE, selection_date DATE,
            symbol VARCHAR, universe_rank INT, market_cap_proxy DOUBLE)""")
        con.execute("CREATE TABLE prices(symbol VARCHAR, date DATE, close DOUBLE, volume DOUBLE)")
        days = pd.bdate_range("2016-01-04", "2016-03-31")
        caps = {"JUNK": 5e13, "BIG": 6e11, "MID": 2e11, "SMALL": 1e10}
        vols = {"JUNK": 1e3, "BIG": 1e8, "MID": 5e7, "SMALL": 1e7}       # shares/day at $10
        for rank, (sym, cap) in enumerate(caps.items(), start=1):
            con.execute("INSERT INTO universe_membership VALUES ('2016-04-01','2016-06-30','2016-03-31',?,?,?)",
                        [sym, rank, cap])
            con.executemany("INSERT INTO prices VALUES (?,?,?,?)",
                            [[sym, d.date(), 10.0, vols[sym]] for d in days])
        con.close()
        kept, _ = load_universe(db, top_n=2)
        assert list(kept.symbol) == ["BIG", "MID"], kept.symbol.tolist()   # JUNK: cap/ADV = 5e6

    # Case 8: point-in-time fundamentals -- a quarter is invisible before
    # filing date + 1 day, and YoY growth needs the same quarter a year ago.
    f = pd.DataFrame({"symbol": "X", "fiscal_date": pd.to_datetime(
        ["2019-03-31", "2019-06-30", "2019-09-30", "2019-12-31", "2020-03-31"]),
        "filing_date": pd.to_datetime(["2019-05-01", "2019-08-01", "2019-11-01", "2020-02-01", "2020-05-01"]),
        "revenue": [100.0, 100, 100, 100, 120], "operating_income": 10.0, "net_income": 5.0,
        "eps": [1.0, 1, 1, 1, 0.96], "operating_cash_flow": 20.0, "capex": -5.0, "free_cash_flow": np.nan,
        "shares_outstanding": 100.0, "total_assets": 1000.0})
    ev = fundamental_events(f)
    last = ev.iloc[-1]
    assert last.available == pd.Timestamp("2020-05-02")
    assert np.isclose(last.rev_yoy, 0.2) and np.isclose(last.eps_chg, -0.04) and last.eps_ok == 1.0

    # Case 10: group relative strength -- sector 0 rises 1%/day, sector 1 is
    # flat: sector 0 stocks get positive RS, sector 1 negative; "turn" is ~0
    # for a constant trend. The overlay re-orders equal base scores by group.
    T_, rets = 40, np.zeros((40, 6))
    rets[:, :3] = 0.01
    codes_ = np.array([[0, 0, 0, 1, 1, 1]] * T_)
    stock_rs, _ = group_relative_strength(rets, np.ones((T_, 6), bool), codes_, 10, "level")
    assert (stock_rs[-1, :3] > 0).all() and (stock_rs[-1, 3:] < 0).all()
    turn_rs, _ = group_relative_strength(rets, np.ones((T_, 6), bool), codes_, 10, "turn")
    assert np.allclose(turn_rs[-1], 0, atol=1e-9)
    base_ = np.ones((1, 6), dtype="float32")
    over = regime_overlay(base_, [stock_rs[-1:]], 1.0)
    assert (over[0, :3] > over[0, 3:]).all()

    # Case 11: duplicate listings -- B copies A's returns, C is independent;
    # B (alphabetically later) is blocked once 63 identical sessions are seen.
    rng_ = np.random.default_rng(1)
    ra = rng_.normal(0, 0.01, 120)
    rr = pd.DataFrame({"A": ra, "B": ra, "C": rng_.normal(0, 0.01, 120)})
    assert duplicate_listings(rr) == [("A", "B", "same series")]
    allowed = pd.DataFrame(True, index=rr.index, columns=rr.columns)
    blk = duplicate_block(rr, [("B", "A", "same series")], allowed)
    assert blk.B.iloc[-1] and not blk.A.any() and not blk.C.any() and not blk.B.iloc[5]
    assert duplicate_listings(rr[["C"]].assign(D=rng_.normal(0, 0.01, 120)),
                              pd.Series({"C": "Fox Corporation", "D": "Fox Corp"})) == [("C", "D", "same name")]

    # Case 9: RSI on a monotonically rising series is 100; falling -> 0.
    up = pd.DataFrame({"u": np.arange(1, 40, dtype=float), "d": np.arange(40, 1, -1, dtype=float)})
    r = calculate_rsi(up, 14)
    assert np.isclose(r.u.iloc[-1], 100) and np.isclose(r.d.iloc[-1], 0)

    # Case 12: 90-day review. A always ranks first -> with the review it is
    # sold at maturity and bought back at the SAME close (new 90-day lot);
    # without it, the buy-back waits one session. If A falls out of the top
    # renew_rank, it is sold and the slot goes to B at that close.
    days = pd.bdate_range("2021-01-04", periods=260)
    prices = pd.DataFrame({"A": np.linspace(10, 20, len(days)), "B": 10.0}, index=days)

    def ranked(order):
        return {d: pd.Series([2.0, 1.0], index=order(d)) for d in days}

    keep_a = ranked(lambda d: ["A", "B"])
    for renew, gap in ((True, 0), (False, 1)):
        res = Backtester(prices, renew=renew).run(TopKStrategy(keep_a, 1, max_weight=1.0, renew_rank=2), days[1], days[-1])
        f = res.fills
        sells, buys = f.loc[f.side.eq("SELL")], f.loc[f.side.eq("BUY")]
        a_buys = buys.loc[buys.symbol.eq("A")]
        assert len(sells) >= 2 and (buys.loc[buys.symbol.ne("A")].cash_change > -0.01 * INITIAL_CAPITAL).all()
        first_rebuy = a_buys.date.iloc[1]   # (B only gets small leftover cash)
        assert days.get_loc(first_rebuy) - days.get_loc(sells.date.iloc[0]) == gap
        audit_quiet = (np.isclose(INITIAL_CAPITAL + f.cash_change.sum(), res.history.cash.iloc[-1], atol=1e-4)
                       and ((sells.date - sells.entry_date).dt.days >= HOLDING_DAYS).all())
        assert audit_quiet
    drop_a = ranked(lambda d: ["A", "B"] if d < days[40] else ["B", "A"])
    res = Backtester(prices, renew=True).run(TopKStrategy(drop_a, 1, max_weight=1.0, renew_rank=1), days[1], days[-1])
    f = res.fills
    first_sell = f.loc[f.side.eq("SELL")].date.iloc[0]
    on_sell = f.loc[f.side.eq("BUY") & f.date.eq(first_sell)]
    assert on_sell.symbol.tolist() == ["B"] and on_sell.cash_change.sum() < -0.5 * INITIAL_CAPITAL   # A let go

    # Case 13: growth tilt. A low-vol episode starts when the trigger fires,
    # ends once growth stocks earned +5% over the index, and cannot restart
    # until the trigger has switched off once; tilted names get 1.5x budgets.
    days = pd.bdate_range("2020-01-01", periods=400)
    rng_ = np.random.default_rng(3)
    bench = pd.Series(np.cumprod(1 + np.r_[rng_.normal(0, 0.02, 300), np.full(100, 0.0)]) * 100, index=days)
    px_ = pd.DataFrame({"G": bench * np.r_[np.ones(330), np.linspace(1, 1.2, 70)], "S": bench}, index=days)

    from types import SimpleNamespace
    _MD = SimpleNamespace(calendar=days, px=px_, bench=bench,
                          info=pd.DataFrame({"sector": ["Tech", "Utilities"], "industry": ["x", "y"]}, index=["G", "S"]))
    el = pd.DataFrame(True, index=days, columns=["G", "S"])
    gr = np.tile([True, False], (len(days), 1))
    on = growth_tilt_episodes(_MD, el, gr, "lowvol", 0.05)
    calm_ = market_is_calm(bench, days)
    first_on = np.flatnonzero(on)[0]
    assert calm_[first_on] and not on[:first_on].any()
    g_cum = px_.G.iloc[first_on:] / px_.G.iloc[first_on] / (bench.iloc[first_on:] / bench.iloc[first_on]) - 1
    stop = first_on + int(np.argmax(g_cum.to_numpy() >= 0.05))
    assert on[stop - 1] and not on[stop] and not on[stop:].any()     # calm never switches off again
    Mt, size = apply_growth_tilt(np.ones((len(days), 2), "float32"), gr, np.ones_like(gr, float), on, 1.5, False)
    assert size[stop - 1, 0] == 1.5 and size[stop - 1, 1] == 1.0 and size[stop, 0] == 1.0

    # Case 14: dispersion weights. Strategy X earns +1% in high-dispersion
    # rows and -1% in low ones, Y the reverse; each regime must lean into
    # the strategy that worked there, using only finished outcomes.
    T_ = 400
    bk_ = np.where(np.arange(T_) % 2 == 0, 1, 0)
    ev_ = {"fcf": np.where(bk_ == 1, 0.01, -0.01), "y": np.where(bk_ == 1, -0.01, 0.01)}
    W_ = dispersion_weights(ev_, np.arange(T_) + 70, bk_, 2, [200], T_, 0, "best")
    assert W_["fcf"][250] == 1 and W_["y"][250] == 0 and W_["y"][251] == 1 and W_["fcf"][251] == 0
    W_ = dispersion_weights(ev_, np.arange(T_) + 70, bk_, 2, [200], T_, 0, "fcf+lean")
    assert W_["fcf"][251] == 1 and W_["y"][251] == 1 and W_["y"][250] == 0
    sc_ = combine_dynamic({"fcf": np.array([[1.0, np.nan]], "float32"), "y": np.array([[np.nan, 1.0]], "float32")},
                          {"fcf": np.array([1.0], "float32"), "y": np.array([0.0], "float32")})
    assert sc_[0, 0] > 0 and np.isnan(sc_[0, 1])       # zero-weight strategy's picks are not candidates

    # Case 15: factor search pieces.
    # (a) Sector revenue leaders pick the top grower of EACH sector, even
    #     when one sector grows faster overall.
    syms = [f"S{i}" for i in range(12)]
    idx_ = pd.bdate_range("2020-01-01", periods=3)
    rev = pd.DataFrame([[0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.05, 0.04, 0.03, 0.02, 0.01, 0.00]] * 3,
                       index=idx_, columns=syms)
    fund_ = {"rev_yoy": rev, "op_margin": pd.DataFrame(0.1, index=idx_, columns=syms)}
    sect = pd.Series(["Tech"] * 6 + ["Utilities"] * 6, index=syms)
    el_ = pd.DataFrame(True, index=idx_, columns=syms)
    lead = generate_sector_revenue_leaders(fund_, el_, sect, "rev_yoy", 0.1, True)
    assert set(np.array(syms)[~np.isnan(lead[0])]) == {"S0", "S6"}
    # (b) Strategy momentum only uses outcomes known at t: A pays +1%, B -1%
    #     until row 100, then the reverse; weights follow with a lag.
    T_ = 300
    evA = np.where(np.arange(T_) < 100, 0.01, -0.01)
    ends = np.arange(T_) + 65
    Wt = trailing_strategy_weights({"A": evA, "B": -evA}, ends, T_, 30, "best")
    assert Wt["A"][130] == 1 and Wt["B"][250] == 1
    assert Wt["A"][164] == 1          # row 100's outcome is only known at row 166
    # (c) Mega-cap tilt: only when mega caps lead, larger companies rank first.
    Mm = np.array([[1.0, 2.0], [1.0, 2.0]], "float32")       # name 1 has the better signal
    rk = np.array([[1.0, 300.0], [1.0, 300.0]], "float32")   # name 0 is the mega cap
    tilted = apply_mega_tilt(Mm, rk, np.ones((2, 2), bool), np.array([True, False]), 2.0, False)
    assert tilted[0, 0] > tilted[0, 1] and tilted[1, 1] > tilted[1, 0]

    # Case 16: price repair. A 10:1 unadjusted split and a +100%/-50% glitch
    # become 0% days; every other return and the latest price are unchanged.
    days = pd.bdate_range("2020-01-01", periods=30)
    true = 100 * np.cumprod(np.r_[1.0, np.full(29, 1.01)])
    bad = true.copy()
    bad[:15] *= 10                       # prices before day 15 not split-adjusted
    bad[20:23] *= 2                      # glitch: doubles on day 20, back on day 23
    pxb = pd.DataFrame({"X": bad}, index=days)
    fixed, _, reps = repair_price_errors(pxb, pxb.copy())
    rr_ = fixed.X.pct_change().to_numpy()
    assert set(reps.kind) == {"level shift", "round trip"} and len(reps) == 3
    assert np.allclose(rr_[[15, 20, 23]], 0) and np.isclose(fixed.X.iloc[-1], bad[-1])
    assert np.allclose(np.delete(rr_[1:], [14, 19, 22]), 0.01)

    print("\nAll self-checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default=str(DEFAULT_DB), help="DuckDB file (default: full HW4 database)")
    parser.add_argument("--check", action="store_true", help="run accounting/timing self-checks and exit")
    parser.add_argument("--dev-only", action="store_true", help="research + validation only; never touch the test period")
    args = parser.parse_args()
    if args.check:
        self_check()
    else:
        main(args.db, dev_only=args.dev_only)
