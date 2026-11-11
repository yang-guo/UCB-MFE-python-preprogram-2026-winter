"""Monthly return forecasts and an independently implemented HW4 backtest.

Run from the repository root with the full bCourses DuckDB file in
Lectures/modeling/data/equities_top2000.duckdb. Features and labels use the
course's audited helper functions; the portfolio ledger below is this
submission's own implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import hashlib
from html import escape

import duckdb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ROOT = Path(__file__).resolve().parents[3]
MODELING = ROOT / "Lectures" / "modeling"
DATABASE = MODELING / "data" / "equities_top2000.duckdb"
OUTPUT = Path(__file__).resolve().parent / "results"
CORPORATE_ACTIONS = Path(__file__).resolve().parent / "corporate_actions.csv"

TRAIN_START = pd.Timestamp("2010-01-01")
FIRST_TEST = pd.Timestamp("2016-04-01")
TEST_END = pd.Timestamp("2026-09-30")
INITIAL_CASH = 1_000_000.0
FEE_BPS = 5.0
HOLDING_DAYS = 90
TOP_N = 20
COHORT_FRACTION = 1 / 3
NAME_CAP = 0.05
SECTOR_CAP = 0.25
ALPHA_GRID = [1.0, 100.0, 1000.0]
FEATURE_BATCH_SIZE = 100


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _course_helpers():
    """Import the instructor's documented, reusable feature/data helpers."""
    import sys

    if str(MODELING) not in sys.path:
        sys.path.insert(0, str(MODELING))
    import course_data

    return course_data


def _read_reference_tables(database: Path):
    with duckdb.connect(str(database), read_only=True) as con:
        calendar = con.execute(
            "SELECT date FROM calendar WHERE date <= ? ORDER BY date", [TEST_END]
        ).fetchdf()["date"]
        benchmark = con.execute(
            "SELECT date, adj_close FROM benchmark WHERE date <= ? ORDER BY date", [TEST_END]
        ).fetchdf()
        symbols = con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchdf()["symbol"]
        membership = con.execute(
            """SELECT symbol, quarter_start, quarter_end, selection_date, universe_rank
               FROM universe_membership WHERE quarter_start <= ?
               ORDER BY quarter_start, universe_rank""", [TEST_END]
        ).fetchdf()
        universe = con.execute("SELECT symbol, sector FROM universe ORDER BY symbol").fetchdf()
    calendar = pd.to_datetime(calendar).astype("datetime64[ns]")
    benchmark["date"] = pd.to_datetime(benchmark["date"]).astype("datetime64[ns]")
    for col in ["quarter_start", "quarter_end", "selection_date"]:
        membership[col] = pd.to_datetime(membership[col])
    benchmark["adj_close"] = pd.to_numeric(benchmark.adj_close, errors="coerce")
    return pd.DatetimeIndex(calendar), benchmark, symbols.tolist(), membership, universe


def _month_end_signals(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = pd.Series(calendar, index=calendar).groupby(calendar.to_period("M")).max()
    return pd.DatetimeIndex(dates.to_numpy())


def _usable_price_rows(prices: pd.DataFrame, calendar: pd.DatetimeIndex):
    """Remove untraded quotes and flag price-basis or continuity breaks.

    Factor jumps above 50% and returns after gaps longer than five sessions
    are audit triggers, not corrections. Those rows and later signals in the
    ticker are quarantined because the data cannot establish continuity.
    """
    frame = prices.sort_values(["symbol", "date"]).copy()
    traded = np.isfinite(frame.volume) & frame.volume.gt(0)
    good_price = (np.isfinite(frame.close) & frame.close.gt(0)
                  & np.isfinite(frame.adj_close) & frame.adj_close.gt(0))
    factor = frame.adj_close / frame.close.where(frame.close.gt(0))
    factor = factor.where(traded & good_price)
    previous = factor.groupby(frame.symbol).ffill().groupby(frame.symbol).shift()
    change = factor / previous
    factor_pair_valid = np.isfinite(factor) & factor.gt(0) & np.isfinite(previous) & previous.gt(0)
    basis_break = factor_pair_valid & ((change > 1.5) | (change < (1 / 1.5)))
    usable_candidate = frame.loc[traded & good_price & ~basis_break].copy()
    session_positions = calendar.get_indexer(pd.DatetimeIndex(usable_candidate.date))
    session_series = pd.Series(session_positions, index=usable_candidate.index)
    prior_session = session_series.groupby(usable_candidate.symbol).shift()
    gap_break = (session_series - prior_session).gt(6).fillna(False)
    basis_events = frame.loc[basis_break, ["symbol", "date"]].copy()
    basis_events["event"] = "adjusted_price_basis_break"
    gap_events = usable_candidate.loc[gap_break, ["symbol", "date"]].copy()
    gap_events["event"] = "quote_return_after_long_gap"
    breaks = pd.concat([basis_events, gap_events], ignore_index=True)
    break_dates = {symbol: pd.DatetimeIndex(group.date)
                   for symbol, group in breaks.groupby("symbol", sort=False)}
    break_reasons = {(row.symbol, row.date): row.event for row in breaks.itertuples(index=False)}
    usable = usable_candidate.loc[~gap_break].copy()
    audit = {"input_rows": len(frame), "zero_or_missing_volume_rows": int((~traded).sum()),
             "invalid_price_rows": int((~good_price).sum()),
             "adjusted_basis_break_rows": int(basis_break.sum()),
             "long_gap_return_rows": int(gap_break.sum()),
             "symbols_with_price_breaks": int(breaks.symbol.nunique()),
             "usable_rows": len(usable)}
    return usable, break_dates, break_reasons, audit


def _add_monthly_targets(panel: pd.DataFrame, prices: pd.DataFrame,
                         calendar: pd.DatetimeIndex, break_dates) -> pd.DataFrame:
    """Align each month-end signal to next-session entry and a 90-day exit."""
    result = panel.copy()
    positions = calendar.get_indexer(result.date)
    next_pos = positions + 1
    entry = np.full(len(result), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid = (positions >= 0) & (next_pos < len(calendar))
    entry[valid] = calendar[next_pos[valid]].to_numpy(dtype="datetime64[ns]")
    result["entry_date"] = pd.to_datetime(entry)
    result["entry_price"] = np.nan
    result["label_end"] = pd.NaT
    result["target"] = np.nan
    result["target_status"] = "no_entry_quote"

    price_groups = {s: g.set_index("date").adj_close.sort_index()
                    for s, g in prices.groupby("symbol", sort=False)}
    for symbol, idx in result.groupby("symbol", sort=False).groups.items():
        if symbol not in price_groups:
            continue
        quote = price_groups[symbol]
        rows = result.loc[idx]
        entry_dates = pd.DatetimeIndex(rows.entry_date)
        entry_px = quote.reindex(entry_dates).to_numpy(dtype=float)
        result.loc[idx, "target_status"] = np.where(
            np.isfinite(entry_px) & (entry_px > 0), "no_exit_quote", "no_entry_quote"
        )
        scheduled = entry_dates + pd.Timedelta(days=HOLDING_DAYS)
        entry_valid = ~entry_dates.isna()
        quote_ix = np.full(len(rows), len(quote), dtype=int)
        quote_ix[entry_valid] = quote.index.searchsorted(scheduled[entry_valid])
        quote_valid = entry_valid & (quote_ix < len(quote))
        exit_dates = np.full(len(rows), np.datetime64("NaT"), dtype="datetime64[ns]")
        exit_px = np.full(len(rows), np.nan)
        if quote_valid.any():
            row_loc = np.flatnonzero(quote_valid)
            candidates = quote.index.to_numpy()[quote_ix[quote_valid]]
            session_ix = calendar.get_indexer(pd.DatetimeIndex(candidates))
            due_ix = calendar.searchsorted(scheduled[quote_valid])
            on_time = (session_ix >= 0) & ((session_ix - due_ix) <= 5)
            chosen = row_loc[on_time]
            exit_dates[chosen] = candidates[on_time]
            exit_px[chosen] = quote.iloc[quote_ix[chosen]].to_numpy(dtype=float)
        valid_rows = np.isfinite(entry_px) & (entry_px > 0) & np.isfinite(exit_px) & (exit_px > 0)
        crossed_break = np.zeros(len(rows), dtype=bool)
        events = break_dates.get(symbol, pd.DatetimeIndex([]))
        if len(events):
            for j in np.flatnonzero(valid_rows):
                crossed_break[j] = bool(((events >= entry_dates[j]) & (events <= exit_dates[j])).any())
        accepted_rows = valid_rows & ~crossed_break
        result.loc[idx, "entry_price"] = entry_px
        valid_end = exit_dates.copy()
        valid_end[~accepted_rows] = np.datetime64("NaT")
        result.loc[idx, "label_end"] = pd.to_datetime(valid_end)
        targets = np.full(len(rows), np.nan)
        targets[accepted_rows] = exit_px[accepted_rows] / entry_px[accepted_rows] - 1
        result.loc[idx, "target"] = targets
        statuses = np.asarray(result.loc[idx, "target_status"].copy())
        statuses[valid_rows & ~crossed_break] = "observed"
        statuses[crossed_break] = "censored_price_continuity_break"
        result.loc[idx, "target_status"] = statuses
    return result


def build_signal_panel(database: Path, calendar, symbols, membership):
    """Build monthly, dated cross-sections in bounded ticker batches."""
    course_data = _course_helpers()
    signal_dates = _month_end_signals(calendar)
    parts, quality_rows, all_break_dates = [], [], {}
    for first in range(0, len(symbols), FEATURE_BATCH_SIZE):
        batch = symbols[first:first + FEATURE_BATCH_SIZE]
        prices, fundamentals, universe, _ = course_data.load_data(
            database, end=TEST_END.strftime("%Y-%m-%d"), symbols=batch
        )
        if prices.empty:
            continue
        prices, break_dates, _break_reasons, quality = _usable_price_rows(prices, calendar)
        quality["symbols_in_batch"] = len(batch)
        quality_rows.append(quality)
        all_break_dates.update(break_dates)
        panel = course_data.price_features(prices, calendar)
        events = course_data.financial_events(fundamentals)
        panel = course_data.join_financials(panel, events, universe)
        panel = course_data.apply_membership(panel, membership, calendar)
        first_break = {symbol: dates.min() for symbol, dates in break_dates.items()}
        if first_break:
            threshold = panel.symbol.map(first_break)
            panel = panel.loc[threshold.isna() | panel.date.lt(threshold)].copy()
        panel = panel.loc[panel.date.isin(signal_dates) & panel.eligible].copy()
        panel = _add_monthly_targets(panel, prices, calendar, break_dates)
        parts.append(panel)
        print(f"Features: {min(first + len(batch), len(symbols)):,}/{len(symbols):,} securities", flush=True)
    if not parts:
        raise ValueError("No eligible monthly signal rows were built from the database")
    result = pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    if result.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate signal-date/ticker rows")
    return result, pd.DataFrame(quality_rows), all_break_dates


def select_model(panel, features):
    """Choose Ridge alpha using whole-date chronological folds in development."""
    from course_data import make_pipeline
    from course_data import date_folds

    dev = panel.loc[
        panel.date.ge(TRAIN_START) & panel.date.lt("2016-01-01")
        & panel.eligible & panel.target.notna()
    ].reset_index(drop=True)
    folds = date_folds(dev, validation_years=(2012, 2013, 2014, 2015),
                       train_start=TRAIN_START.strftime("%Y-%m-%d"))
    search = GridSearchCV(
        make_pipeline(Ridge()), {"model__alpha": ALPHA_GRID}, cv=folds,
        scoring="neg_root_mean_squared_error", n_jobs=1, error_score="raise",
        return_train_score=True,
    )
    search.fit(dev[features], dev.target)
    cv = pd.DataFrame(search.cv_results_)
    cv["mean_fold_RMSE"] = -cv["mean_test_score"]
    return search, dev, cv


def make_walk_forward_scores(panel, development, estimator, features, calendar):
    """Quarterly expanding fits; each fit sees only labels completed before it."""
    quarter_end_signals = _month_end_signals(calendar)
    quarter_end_signals = quarter_end_signals[
        (quarter_end_signals >= pd.Timestamp("2016-03-31"))
        & (quarter_end_signals <= TEST_END)
        & (quarter_end_signals.to_period("M").month % 3 == 0)
    ]
    # The initial fit is the final development signal immediately before the buffer.
    fit_dates = [pd.Timestamp("2016-03-31")]
    fit_dates += [pd.Timestamp(x) for x in quarter_end_signals if x > fit_dates[0]]
    fit_dates = sorted(set(fit_dates))
    all_scores, audits = [], []
    signal_dates = pd.DatetimeIndex(sorted(panel.loc[panel.eligible, "date"].unique()))
    test_signal_dates = signal_dates[(signal_dates >= fit_dates[0]) & (signal_dates <= TEST_END)]

    for i, asof in enumerate(fit_dates):
        next_asof = fit_dates[i + 1] if i + 1 < len(fit_dates) else TEST_END + pd.Timedelta(days=1)
        training = panel.loc[
            panel.date.ge(TRAIN_START) & panel.date.lt(asof)
            & panel.label_end.lt(asof) & panel.target.notna() & panel.eligible
        ]
        scoring = panel.loc[
            panel.date.ge(asof) & panel.date.lt(next_asof)
            & panel.date.isin(test_signal_dates) & panel.eligible
        ]
        if training.empty or scoring.empty:
            audits.append({"fit_date": asof, "training_rows": len(training),
                           "last_training_signal": training.date.max() if len(training) else pd.NaT,
                           "last_label_end": training.label_end.max() if len(training) else pd.NaT,
                           "scored_rows": len(scoring)})
            continue
        model = clone(estimator).fit(training[features], training.target)
        scores = scoring[["date", "symbol"]].copy()
        scores["score"] = model.predict(scoring[features])
        scores["fit_date"] = asof
        all_scores.append(scores)
        audits.append({"fit_date": asof, "training_rows": len(training),
                       "last_training_signal": training.date.max(),
                       "last_label_end": training.label_end.max(),
                       "first_scored_signal": scoring.date.min(),
                       "last_scored_signal": scoring.date.max(), "scored_rows": len(scoring)})
    if not all_scores:
        raise ValueError("Walk-forward schedule generated no scores")
    scores = pd.concat(all_scores, ignore_index=True).sort_values(["date", "symbol"])
    if scores.duplicated(["date", "symbol"]).any():
        raise ValueError("A ticker received multiple scores for one signal date")
    audit = pd.DataFrame(audits)
    mature = audit.dropna(subset=["last_label_end"])
    if not (mature.last_label_end < mature.fit_date).all():
        raise AssertionError("A model fit used an outcome that had not matured")
    if not (scores.fit_date <= scores.date).all():
        raise AssertionError("A score predates its model fit")
    return scores.reset_index(drop=True), audit


@dataclass
class Lot:
    lot_id: int
    symbol: str
    units: float
    entry_date: pd.Timestamp
    entry_price: float
    entry_fee: float
    due_date: pd.Timestamp
    signal_date: pd.Timestamp
    adj_factor_at_entry: float


class PortfolioLedger:
    """Own the cash, independent lots, trade fees, and realized PnL."""

    def __init__(self, initial_cash=INITIAL_CASH, fee_bps=FEE_BPS):
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee_rate = fee_bps / 10_000
        self.lots: dict[int, Lot] = {}
        self.fills = []
        self._next_id = 0
        self.total_fees = 0.0
        self.realized_pnl = 0.0

    def buy(self, symbol, budget, price, date, signal_date, adj_factor=1.0):
        units = budget / (price * (1 + self.fee_rate))
        notional = units * price
        fee = notional * self.fee_rate
        if notional + fee > self.cash + 1e-7:
            raise ValueError("Purchase would spend more than available cash")
        lot = Lot(self._next_id, symbol, units, date, price, fee,
                  date + pd.Timedelta(days=HOLDING_DAYS), signal_date, adj_factor)
        self.cash = max(0.0, self.cash - notional - fee)
        self.lots[lot.lot_id] = lot
        self._record(lot, "BUY", date, units, price, fee,
                     -(notional + fee), 0.0)
        self._next_id += 1

    def sell(self, lot, date, price):
        notional = lot.units * price
        fee = notional * self.fee_rate
        pnl = lot.units * (price - lot.entry_price) - lot.entry_fee - fee
        self.cash += notional - fee
        self.total_fees += fee
        self.realized_pnl += pnl
        self._record(lot, "SELL", date, lot.units, price, fee,
                     notional - fee, pnl, fee_already_counted=True)
        del self.lots[lot.lot_id]

    def _record(self, lot, side, date, units, price, fee, cash_change, pnl,
                fee_already_counted=False):
        if not fee_already_counted:
            self.total_fees += fee
        self.fills.append({"date": date, "signal_date": lot.signal_date,
                           "lot_id": lot.lot_id, "symbol": lot.symbol, "side": side,
                           "price": price, "units": units, "notional": units * price,
                           "fee": fee, "cash_change": cash_change,
                           "due_date": lot.due_date, "realized_pnl": pnl})


def verify_ledger_by_hand() -> str:
    """Reconcile a small round trip whose arithmetic can be checked by hand."""
    ledger = PortfolioLedger(initial_cash=1_000, fee_bps=10)
    signal = pd.Timestamp("2020-01-30")
    entry = pd.Timestamp("2020-01-31")
    ledger.buy("XYZ", budget=500, price=10, date=entry, signal_date=signal)
    lot = next(iter(ledger.lots.values()))
    cash_after_buy = ledger.cash
    expected_units = 500 / (10 * 1.001)
    expected_entry_fee = expected_units * 10 * 0.001
    expected_marked_nav = 500 + expected_units * 12
    marked_nav = ledger.cash + lot.units * 12
    ledger.sell(lot, pd.Timestamp("2020-04-30"), 12)
    expected_exit_fee = expected_units * 12 * 0.001
    expected_final_cash = expected_units * 12 * (1 - 0.001) + 500
    expected_pnl = expected_units * (12 - 10) - expected_entry_fee - expected_exit_fee
    checks = [
        np.isclose(lot.units, expected_units),
        np.isclose(lot.entry_fee, expected_entry_fee),
        np.isclose(cash_after_buy, ledger.initial_cash - 500),
        np.isclose(marked_nav, expected_marked_nav),
        lot.due_date == pd.Timestamp("2020-04-30"),
        np.isclose(ledger.cash, expected_final_cash),
        np.isclose(ledger.realized_pnl, expected_pnl),
        np.isclose(ledger.initial_cash + ledger.realized_pnl, ledger.cash),
        np.isclose(ledger.total_fees, expected_entry_fee + expected_exit_fee),
        not ledger.lots,
    ]
    if not all(checks):
        raise AssertionError("Hand-check round trip did not reconcile")
    return (
        "Manual round trip: $1,000 start; buy $500 inclusive of 10 bps at $10; "
        f"{expected_units:.8f} adjusted units; due 2020-04-30; sell at $12. "
        f"Entry fee ${expected_entry_fee:.6f}; exit fee ${expected_exit_fee:.6f}; "
        f"final cash ${ledger.cash:.6f}; realized PnL ${ledger.realized_pnl:.6f}. "
        "Cash, fees, holding date, and realized PnL reconcile."
    )


def allocate_budgets(scores, ledger, marks, sectors):
    """Rank scores and buy within cash, cohort, name, and sector limits."""
    nav = ledger.cash + sum(l.units * marks[l.symbol] for l in ledger.lots.values())
    holdings = {}
    sector_values = {}
    for lot in ledger.lots.values():
        value = lot.units * marks[lot.symbol]
        holdings[lot.symbol] = holdings.get(lot.symbol, 0.0) + value
        sector = sectors.get(lot.symbol, "Unknown")
        sector_values[sector] = sector_values.get(sector, 0.0) + value
    ranked = scores.sort_values(ascending=False, kind="stable").head(TOP_N)
    remaining = min(ledger.cash, nav * COHORT_FRACTION)
    if ranked.empty:
        return {}
    equal_budget = remaining / len(ranked)
    orders = {}
    for symbol in ranked.index:
        if symbol not in sectors:
            continue
        sector = sectors.get(symbol, "Unknown")
        name_room = max(0.0, NAME_CAP * nav - holdings.get(symbol, 0.0))
        sector_room = max(0.0, SECTOR_CAP * nav - sector_values.get(sector, 0.0))
        budget = min(equal_budget, remaining,
                     name_room * (1 + ledger.fee_rate),
                     sector_room * (1 + ledger.fee_rate))
        if budget > 1e-8:
            orders[symbol] = budget
            remaining -= budget
            notional = budget / (1 + ledger.fee_rate)
            holdings[symbol] = holdings.get(symbol, 0.0) + notional
            sector_values[sector] = sector_values.get(sector, 0.0) + notional
    return orders


def portfolio_metrics(nav, annual_rf=0.02):
    nav = nav.dropna().astype(float).sort_index()
    if len(nav) < 2 or (nav <= 0).any() or not np.isfinite(nav).all():
        raise ValueError("Performance metrics require positive finite daily values")
    returns = nav.pct_change(fill_method=None).dropna()
    vol = returns.std(ddof=1)
    test_start = nav.index[1] if len(nav) > 2 else nav.index[0]
    years = (nav.index[-1] - test_start).days / 365.25
    daily_rf = (1 + annual_rf) ** (1 / 252) - 1
    return pd.Series({
        "final_value": nav.iloc[-1],
        "CAGR": (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1,
        "max_drawdown": (nav / nav.cummax() - 1).min(),
        "Sharpe": np.sqrt(252) * (returns - daily_rf).mean() / vol if vol > 1e-12 else np.nan,
    })


def save_cumulative_chart(history: pd.DataFrame, path: Path, stale_lots: int):
    """Write a lightweight SVG of reported, zero-recovery stress, and benchmark."""
    width, height = 1120, 660
    left, right, top, bottom = 92, width - 38, 94, height - 82
    series = {
        "Reported last-quote marks": history.nav.astype(float) / INITIAL_CASH - 1,
        "Zero-recovery stress": history.zero_recovery_nav.astype(float) / INITIAL_CASH - 1,
        "S&P 500 total-return index": history.benchmark_nav.astype(float) / INITIAL_CASH - 1,
    }
    values = np.concatenate([value.to_numpy() for value in series.values()])
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    pad = max((high - low) * 0.06, 0.02)
    low, high = low - pad, high + pad
    count = len(history)

    def x_at(index):
        return left + (right - left) * index / max(count - 1, 1)

    def y_at(value):
        return bottom - (bottom - top) * (value - low) / (high - low)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#243040}.grid{stroke:#dce2e8;stroke-width:1}.axis{stroke:#52606d;stroke-width:1.3}</style>',
        '<text x="92" y="35" font-size="22" font-weight="bold">Cumulative return, April 2016–September 2026</text>',
        f'<text x="92" y="60" font-size="13" fill="#596579">Sensitivity writes unresolved holdings to zero after 5 missing sessions; {stale_lots} final lots affected</text>',
    ]
    for tick in np.linspace(low, high, 6):
        y = y_at(float(tick))
        out.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}"/>')
        out.append(f'<text x="{left - 12}" y="{y + 5:.1f}" text-anchor="end" font-size="12">{tick:.0%}</text>')
    years = list(range(2016, 2027, 2))
    for year in years:
        matches = np.flatnonzero(pd.DatetimeIndex(history.index).year == year)
        if not len(matches):
            continue
        x = x_at(int(matches[0]))
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>')
        out.append(f'<text x="{x:.1f}" y="{bottom + 25}" text-anchor="middle" font-size="12">{year}</text>')
    out.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>')
    out.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
    colors = {"Reported last-quote marks": "#1769aa", "Zero-recovery stress": "#8c5a9e", "S&P 500 total-return index": "#d1495b"}
    for name, values_series in series.items():
        coords = [f'{x_at(i):.1f},{y_at(float(value)):.1f}'
                  for i, value in enumerate(values_series.to_numpy())]
        out.append(f'<polyline fill="none" stroke="{colors[name]}" stroke-width="2.2" points="{" ".join(coords)}"/>')
    legend_x = left
    for name, color in colors.items():
        out.append(f'<line x1="{legend_x}" y1="{height - 28}" x2="{legend_x + 26}" y2="{height - 28}" stroke="{color}" stroke-width="3"/>')
        out.append(f'<text x="{legend_x + 34}" y="{height - 23}" font-size="13">{escape(name)}</text>')
        legend_x += 310
    out.append('</svg>')
    path.write_text("\n".join(out), encoding="utf-8")


def run_portfolio(database, scores, universe, start=FIRST_TEST, end=TEST_END):
    """Daily close-to-close simulation, with prior-close decisions and 90-day lots."""
    with duckdb.connect(str(database), read_only=True) as con:
        prices = con.execute(
            """SELECT symbol, date, close, adj_close, volume FROM prices
               WHERE date >= ? AND date <= ? ORDER BY date, symbol""",
            [pd.Timestamp(start) - pd.Timedelta(days=10), end]
        ).fetchdf()
        benchmark = con.execute(
            """SELECT date, adj_close FROM benchmark
               WHERE date >= ? AND date <= ? ORDER BY date""", [start, end]
        ).fetchdf()
        calendar = con.execute(
            """SELECT date FROM calendar WHERE date >= ? AND date <= ? ORDER BY date""",
            [pd.Timestamp(start) - pd.Timedelta(days=10), end]
        ).fetchdf().date
    prices["date"] = pd.to_datetime(prices.date)
    source_prices = prices.copy()
    benchmark["date"] = pd.to_datetime(benchmark.date)
    prices, _test_break_dates, test_break_reasons, _ = _usable_price_rows(
        prices, pd.DatetimeIndex(pd.to_datetime(calendar))
    )
    test_breaks_by_date = {}
    for (symbol, date), reason in test_break_reasons.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            test_breaks_by_date.setdefault(date, {})[symbol] = reason
    all_dates = pd.DatetimeIndex(pd.to_datetime(calendar))
    dates = all_dates[all_dates >= pd.Timestamp(start)]
    quotes = prices.loc[prices.date >= pd.Timestamp(start)].pivot(
        index="date", columns="symbol", values="adj_close"
    ).reindex(dates)
    benchmark = benchmark.set_index("date").adj_close.reindex(dates)
    sectors = universe.set_index("symbol").sector.fillna("Unknown").to_dict()
    actions = pd.read_csv(CORPORATE_ACTIONS, parse_dates=["effective_date"])
    cash_actions = {
        row.symbol: (row.effective_date, float(row.cash_per_share))
        for row in actions.itertuples(index=False)
        if row.event_type == "cash_acquisition" and pd.notna(row.cash_per_share)
    }
    reorg = pd.read_csv(
        Path(__file__).resolve().parent / "reorganization_exits.csv",
        parse_dates=["effective_date"],
    ).set_index("symbol")
    external_exits = pd.read_csv(
        Path(__file__).resolve().parent / "supplemental_exit_quotes.csv",
        parse_dates=["due_date"],
    ).set_index(["symbol", "due_date"])
    supplemental_prices = source_prices.loc[
        source_prices.volume.gt(0) & source_prices.close.gt(0) & source_prices.adj_close.gt(0)
    ]
    raw_close = supplemental_prices.pivot(
        index="date", columns="symbol", values="close"
    ).reindex(dates)
    adjusted_close = supplemental_prices.pivot(
        index="date", columns="symbol", values="adj_close"
    ).reindex(dates)
    ledger = PortfolioLedger()
    last_price, last_seen = {}, {}
    unresolved_lots = set()
    blocked_symbols = set()
    previous_session = all_dates[all_dates.get_loc(dates[0]) - 1]
    history = [{"date": previous_session, "cash": ledger.cash,
                "holdings_value": 0.0, "nav": ledger.cash,
                "zero_recovery_nav": ledger.cash,
                "unresolved_writeoff_value": 0.0, "open_lots": 0,
                "fees": 0.0, "stale_marks": 0}]
    diagnostics = []
    for day in dates:
        calendar_pos = all_dates.get_loc(day)
        signal_date = all_dates[calendar_pos - 1]
        # Fix budgets from the prior close before observing today's prices.
        planned_orders = {}
        if signal_date.to_period("M") != day.to_period("M"):
            scores_today = scores.loc[scores.date.eq(signal_date)].set_index("symbol").score
            if not scores_today.empty:
                prior_marks = {s: last_price[s] for s in last_price}
                planned_orders = allocate_budgets(scores_today, ledger, prior_marks, sectors)
                retired_symbols = {
                    symbol for symbol, (event_date, _cash) in cash_actions.items()
                    if event_date <= day
                }
                retired_symbols.update(
                    symbol for symbol, row in reorg.iterrows()
                    if row.effective_date <= day
                )
                for symbol in set(planned_orders) & retired_symbols:
                    diagnostics.append({"date": day, "symbol": symbol,
                                        "event": "rejected_entry_after_corporate_action"})
                    planned_orders.pop(symbol)
                if sum(planned_orders.values()) > ledger.cash + 1e-6:
                    raise ValueError("Order budgets exceeded signal-date cash")
        quoted = quotes.loc[day].dropna()
        # A quote after a long gap may belong to a different price or issuer
        # basis. Freeze held positions at their last valid mark after five
        # missing sessions; do not apply a later quote to that old lot.
        for lot in ledger.lots.values():
            if lot.symbol in quoted and lot.symbol in last_seen:
                age_before_quote = calendar_pos - last_seen[lot.symbol]
                if age_before_quote > 5 and lot.symbol not in blocked_symbols:
                    blocked_symbols.add(lot.symbol)
                    diagnostics.append({"date": day, "symbol": lot.symbol,
                                        "event": "quote_return_after_long_gap",
                                        "last_quote_date": all_dates[last_seen[lot.symbol]],
                                        "sessions_since_quote": age_before_quote})
        for symbol, reason in sorted(test_breaks_by_date.get(day, {}).items()):
            if symbol not in blocked_symbols:
                blocked_symbols.add(symbol)
                diagnostics.append({"date": day, "symbol": symbol, "event": reason})
        observed = quoted.drop(labels=list(blocked_symbols), errors="ignore")
        for symbol, price in observed.items():
            if np.isfinite(price) and price > 0:
                last_price[symbol] = float(price)
                last_seen[symbol] = calendar_pos
        # A verified cash merger redeems the old shares on its effective date.
        # Convert per-share cash into this simulation's adjusted-price units
        # using the actual entry-date adjusted/raw close ratio; this keeps
        # historical dividend adjustment from distorting share count.
        for lot in list(ledger.lots.values()):
            action = cash_actions.get(lot.symbol)
            if action and action[0] == day:
                ledger.sell(lot, day, action[1] * lot.adj_factor_at_entry)
                diagnostics.append({"date": day, "symbol": lot.symbol,
                                   "event": "verified_cash_merger_settlement",
                                   "cash_per_share": action[1]})
        # Preserve an old lot through a documented spin-off, stock merger, or
        # confirmed ticker alias, then settle it using the successor quote at
        # its 90-calendar-day horizon (next session when maturity is closed).
        for lot in list(ledger.lots.values()):
            if lot.symbol not in reorg.index or day < lot.due_date:
                continue
            term = reorg.loc[lot.symbol]
            if term.effective_date > day or calendar_pos < all_dates.searchsorted(lot.due_date):
                continue
            supplemental_key = (lot.symbol, lot.due_date)
            external = external_exits.loc[supplemental_key] if supplemental_key in external_exits.index else None
            quote_value = external.close if external is not None else term.successor_adj_close_on_exit
            if pd.isna(quote_value):
                successor = term.successor_symbol
                if successor not in adjusted_close.columns:
                    continue
                successor_quote = adjusted_close.at[day, successor]
                if not np.isfinite(successor_quote) or successor_quote <= 0:
                    continue
                quote_value = float(successor_quote)
            per_share_value = (
                float(term.consideration_cash)
                + float(term.successor_shares) * float(quote_value)
            )
            ledger.sell(lot, day, per_share_value * lot.adj_factor_at_entry)
            diagnostics.append({"date": day, "symbol": lot.symbol,
                               "event": "verified_successor_or_ticker_mapping_exit",
                               "successor": term.successor_symbol,
                               "consideration_per_old_share": per_share_value,
                               "source_url": external.source_url if external is not None else term.source_url})
        stale = 0
        for lot in list(ledger.lots.values()):
            if day >= lot.due_date and lot.symbol in observed:
                ledger.sell(lot, day, float(observed[lot.symbol]))
        for lot in ledger.lots.values():
            age = calendar_pos - last_seen.get(lot.symbol, calendar_pos)
            if lot.symbol not in observed:
                stale += 1
                if age > 5 and lot.lot_id not in unresolved_lots:
                    unresolved_lots.add(lot.lot_id)
                    blocked_symbols.add(lot.symbol)
                    diagnostics.append({"date": day, "symbol": lot.symbol,
                                        "event": "unresolved_gap_over_five_sessions",
                                        "last_quote_date": all_dates[last_seen[lot.symbol]],
                                        "sessions_since_quote": age})

        # Execute previously planned orders at today's close.
        for symbol, budget in planned_orders.items():
            if symbol not in observed:
                diagnostics.append({"date": day, "symbol": symbol, "event": "rejected_missing_quote"})
                continue
            adj_px = float(observed[symbol])
            raw_px = float(raw_close.loc[day, symbol])
            ledger.buy(symbol, budget, adj_px, day, signal_date,
                       adj_factor=adj_px / raw_px)

        holding_values = sum(lot.units * last_price[lot.symbol] for lot in ledger.lots.values())
        nav = ledger.cash + holding_values
        # The source has no dependable delisting proceeds. Report a separate,
        # conservative sensitivity that assigns zero value once a held name has
        # gone without a valid quote for more than five sessions. This is a
        # non-cash impairment scenario, not an invented fill or base-case mark.
        unresolved_value = sum(
            lot.units * last_price[lot.symbol]
            for lot in ledger.lots.values() if lot.lot_id in unresolved_lots
        )
        zero_recovery_nav = nav - unresolved_value
        if not np.isclose(zero_recovery_nav + unresolved_value, nav, atol=1e-7):
            raise AssertionError(f"Zero-recovery sensitivity does not reconcile on {day.date()}")
        cash_from_fills = ledger.initial_cash + sum(fill["cash_change"] for fill in ledger.fills)
        if ledger.cash < -1e-7 or not np.isclose(ledger.cash, cash_from_fills, atol=1e-6):
            raise AssertionError(f"Cash does not reconcile to fills on {day.date()}")
        if not np.isclose(nav, ledger.cash + holding_values, atol=1e-7):
            raise AssertionError(f"NAV does not reconcile to cash and holdings on {day.date()}")
        day_fills = [f for f in ledger.fills if f["date"] == day]
        history.append({"date": day, "cash": ledger.cash, "holdings_value": holding_values,
                        "nav": nav, "zero_recovery_nav": zero_recovery_nav,
                        "unresolved_writeoff_value": unresolved_value,
                        "open_lots": len(ledger.lots),
                        "fees": sum(f["fee"] for f in day_fills), "stale_marks": stale})
    history = pd.DataFrame(history).set_index("date")
    initial = ledger.initial_cash
    first_close = benchmark.iloc[0]
    history["benchmark_nav"] = initial
    history.loc[dates, "benchmark_nav"] = initial * benchmark / first_close
    fills = pd.DataFrame(ledger.fills)
    open_rows = []
    final_calendar_pos = all_dates.get_loc(dates[-1])
    for lot in ledger.lots.values():
        last_quote_pos = last_seen.get(lot.symbol, final_calendar_pos)
        open_rows.append({**vars(lot), "last_observed_price": last_price.get(lot.symbol, np.nan),
                          "last_quote_date": all_dates[last_quote_pos],
                          "mark_age_sessions": final_calendar_pos - last_quote_pos,
                          "unresolved_price_gap": final_calendar_pos - last_quote_pos > 5})
    open_lots = pd.DataFrame(open_rows)
    ending_marks = {symbol: last_price[symbol] for symbol in last_price}
    unrealized = sum(lot.units * (ending_marks[lot.symbol] - lot.entry_price) - lot.entry_fee
                     for lot in ledger.lots.values())
    total_pnl = history.nav.iloc[-1] - ledger.initial_cash
    if not np.isclose(total_pnl, ledger.realized_pnl + unrealized, atol=1e-5):
        raise AssertionError("Ending NAV does not reconcile to realized plus unrealized PnL")
    if not np.isclose(ledger.total_fees, fills.fee.sum() if len(fills) else 0.0, atol=1e-7):
        raise AssertionError("Fee total does not reconcile to the fill ledger")
    return history, fills, open_lots, pd.DataFrame(diagnostics), ledger


def run():
    if not DATABASE.is_file():
        raise FileNotFoundError(
            f"Download equities_top2000.duckdb from bCourses to {DATABASE}"
        )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hand_check = verify_ledger_by_hand()
    (OUTPUT / "accounting_check.txt").write_text(hand_check + "\n")
    course_data = _course_helpers()
    features = course_data.FEATURES
    calendar, benchmark_all, symbols, membership, universe = _read_reference_tables(DATABASE)
    # Static source metadata omits instrument type: BWNB is a senior note,
    # and SOFIW is a public warrant. Neither belongs in a common-stock strategy.
    non_equity_symbols = {"BWNB", "SOFIW"}
    symbols = [symbol for symbol in symbols if symbol not in non_equity_symbols]
    membership = membership.loc[~membership.symbol.isin(non_equity_symbols)].copy()
    universe = universe.loc[~universe.symbol.isin(non_equity_symbols)].copy()
    if calendar.max() < TEST_END:
        raise ValueError("Database calendar does not extend through September 30, 2026")
    print(f"Universe securities: {len(symbols):,}; calendar sessions: {len(calendar):,}")
    panel, price_quality, _break_dates = build_signal_panel(
        DATABASE, calendar, symbols, membership
    )
    price_quality.to_csv(OUTPUT / "price_quality_audit.csv", index=False)
    (panel.target_status.value_counts(dropna=False).rename_axis("status")
     .rename("signal_rows").to_csv(OUTPUT / "label_censoring.csv"))
    search, development, cv = select_model(panel, features)
    cv.to_csv(OUTPUT / "validation.csv", index=False)
    search.best_estimator_.fit(development.loc[
        development.label_end.lt("2016-03-31"), features
    ], development.loc[development.label_end.lt("2016-03-31"), "target"])
    scores, fit_audit = make_walk_forward_scores(
        panel, development, search.best_estimator_, features, calendar
    )
    fit_audit.to_csv(OUTPUT / "fit_audit.csv", index=False)
    history, fills, open_lots, diagnostics, ledger = run_portfolio(
        DATABASE, scores, universe
    )
    history.to_csv(OUTPUT / "daily_portfolio.csv")
    fills.to_csv(OUTPUT / "fills.csv", index=False)
    open_lots.to_csv(OUTPUT / "open_lots.csv", index=False)
    if len(open_lots):
        unresolved = open_lots.loc[open_lots.unresolved_price_gap].copy()
        unresolved["last_quote_mark"] = unresolved.units * unresolved.last_observed_price
        (unresolved.groupby("symbol", as_index=False)
         .agg(open_lots=("lot_id", "count"), mark_age_sessions=("mark_age_sessions", "max"),
              oldest_last_quote=("last_quote_date", "min"), latest_due_date=("due_date", "max"),
              last_quote_mark=("last_quote_mark", "sum"))
         .sort_values("last_quote_mark", ascending=False)
         .to_csv(OUTPUT / "unresolved_positions.csv", index=False))
    diagnostics.to_csv(OUTPUT / "data_diagnostics.csv", index=False)
    metrics = pd.DataFrame({
        "Strategy — last-quote valuation": portfolio_metrics(history.nav),
        "Strategy — zero-recovery stress": portfolio_metrics(history.zero_recovery_nav),
        "S&P 500 total-return index": portfolio_metrics(history.benchmark_nav),
    })
    metrics.to_csv(OUTPUT / "performance.csv")
    stale_lots = int(open_lots.unresolved_price_gap.sum()) if len(open_lots) else 0
    save_cumulative_chart(history, OUTPUT / "cumulative_returns.svg", stale_lots)
    config = {
        "database": DATABASE.name,
        "database_sha256": _file_sha256(DATABASE),
        "features": features, "training_start": str(TRAIN_START.date()),
        "development_validation_years": [2012, 2013, 2014, 2015],
        "alpha_grid": ALPHA_GRID, "selected_alpha": search.best_params_["model__alpha"],
        "final_test_start": str(FIRST_TEST.date()), "final_test_end": str(TEST_END.date()),
        "initial_capital": INITIAL_CASH, "top_n": TOP_N,
        "monthly_cohort_fraction": COHORT_FRACTION, "holding_calendar_days": HOLDING_DAYS,
        "fee_bps_per_side": FEE_BPS, "name_cap": NAME_CAP, "sector_cap": SECTOR_CAP,
        "risk_free_rate": 0.02, "cash_interest": 0,
        "universe_policy": "dated quarterly membership at next-session entry; retain held positions after membership exit",
        "non_equity_exclusions": "BWNB senior notes and SOFIW public warrants excluded from the common-stock strategy",
        "corporate_action_inputs": ["corporate_actions.csv", "reorganization_exits.csv", "supplemental_exit_quotes.csv"],
        "retraining": "quarterly expanding window; fit only on outcomes completed before fit date",
        "price_policy": "adjusted closes, fractional adjusted-price units; no separately added dividends",
        "price_quality_policy": "remove nonpositive-volume quote rows; quarantine rows where adj_close/close changes by over 50% or a quote returns after more than five missing sessions; stop scoring that symbol after its first continuity break; censor targets crossing a break; do not apply guessed split corrections",
        "data_gaps": "missing entry quote skips buy; documented cash deals settle on effective date; successor shares and ticker aliases settle at 90-day maturity when sourced successor closes are available; BODY May 30 2024 and DISH February 29 2024 still lack verified closes; after five sessions without a quote, freeze unmatched old lots at their last marks and report zero recovery as a stress only",
    }
    (OUTPUT / "experiment.json").write_text(json.dumps(config, indent=2))
    print("Selected alpha:", search.best_params_["model__alpha"])
    print("\nPerformance comparison\n", metrics)
    print("\nOpen positions at test end:", len(open_lots))
    print("Output directory:", OUTPUT)


if __name__ == "__main__":
    run()
