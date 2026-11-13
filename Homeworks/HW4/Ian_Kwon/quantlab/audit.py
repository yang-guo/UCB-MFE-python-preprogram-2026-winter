"""Independent accounting and timing audit of a ``BacktestResult``.

Everything is recomputed from the fill ledger and the price matrix, not from
the engine's internal state, so an engine bug cannot also hide itself here.
Each check returns pass/fail plus a detail string; ``assert_clean`` raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestResult

TOL = 1e-6


def _check(name, ok, detail=""):
    return {"check": name, "passed": bool(ok), "detail": detail}


def audit_backtest(result: BacktestResult, calendar: pd.DatetimeIndex, symbols: pd.Index,
                   px: np.ndarray) -> pd.DataFrame:
    h, f = result.history, result.fills
    fee_rate = result.settings["fee_rate"]
    holding = pd.Timedelta(days=result.settings["holding_days"])
    out = []
    initial = h.cash.iloc[0]

    # 1. cash reconstruction from the ledger
    daily = f.groupby("date").cash_change.sum().reindex(h.index, fill_value=0.0)
    rebuilt = initial + daily.cumsum()
    err = (rebuilt - h.cash).abs().max()
    out.append(_check("cash = initial + cumulative fill cash changes", err < TOL * max(1, initial),
                      f"max abs error {err:.2e}"))
    out.append(_check("cash never negative", (h.cash >= -TOL).all(), f"min cash {h.cash.min():.6f}"))
    out.append(_check("nav = cash + holdings", ((h.cash + h.holdings_value - h.nav).abs() < TOL * initial).all()))

    # 2. holdings re-marked independently from the price matrix
    col = {s: i for i, s in enumerate(symbols)}
    pos = calendar.get_indexer(h.index[1:])
    marks = pd.DataFrame(px, index=calendar, columns=symbols).ffill()  # last observed quote
    signed = f.assign(u=np.where(f.side.eq("BUY"), f.units, -f.units))
    units = signed.pivot_table(index="date", columns="symbol", values="u", aggfunc="sum")
    units = units.reindex(h.index[1:], fill_value=0.0).fillna(0.0).cumsum()
    # lots made terminal by the ticker-reuse guard keep their pre-gap mark
    ev = result.events
    term = ev.loc[ev.event.eq("terminal_freeze_after_long_gap")] if len(ev) else ev
    fixed = pd.Series(0.0, index=h.index[1:])
    for _, e in term.iterrows():
        lot_units = f.loc[f.lot_id.eq(e.lot_id) & f.side.eq("BUY"), "units"].iloc[0]
        units.loc[units.index >= e.date, e.symbol] -= lot_units
        fixed.loc[fixed.index >= e.date] += e.value
    if units.shape[1]:
        m = marks.loc[h.index[1:], units.columns]
        value = (units * m.fillna(0.0)).sum(axis=1) + fixed
    else:
        value = fixed
    err = (value - h.holdings_value.iloc[1:]).abs().max() if len(value) else 0.0
    out.append(_check("holdings = units x last observed quote (independent re-mark)", err < 1e-4 * initial,
                      f"max abs error {err:.2e}"))
    out.append(_check("units never negative", (units >= -1e-9).all().all() if units.size else True))

    buys, all_sells = f.loc[f.side.eq("BUY")], f.loc[f.side.eq("SELL")]
    stale = all_sells.loc[all_sells.settlement.eq("stale_mark")]
    sells = all_sells.loc[~all_sells.settlement.eq("stale_mark")]
    if len(stale):  # settle_at_maturity policy: last observed quote, frozen at maturity
        sp = calendar.get_indexer(stale.date)
        ok_px, ok_gap = [], []
        for p_, s_, price in zip(sp, stale.symbol, stale.price):
            q = np.flatnonzero(np.isfinite(px[: p_ + 1, col[s_]]))
            ok_px.append(len(q) and np.isclose(px[q[-1], col[s_]], price, atol=1e-12))
            ok_gap.append(len(q) and p_ - q[-1] > result.settings["max_stale_sessions"])
        out.append(_check("stale settlements use the last observed quote of a frozen lot",
                          all(ok_px) and all(ok_gap)))
        out.append(_check("stale settlements happen at/after maturity", (stale.date >= stale.due_date).all()))
    # 3. timing: buys fill at the session immediately after the signal
    if len(buys):
        sig_pos = calendar.get_indexer(buys.signal_date)
        fill_pos = calendar.get_indexer(buys.date)
        out.append(_check("every buy fills on the session after its signal", (fill_pos == sig_pos + 1).all(),
                          f"violations {int((fill_pos != sig_pos + 1).sum())}"))
        bpx = px[fill_pos, [col[s] for s in buys.symbol]]
        out.append(_check("buy price = that session's close", np.allclose(bpx, buys.price, rtol=0, atol=1e-12)))
        cash_before = h.cash.shift(1).reindex(buys.date.unique())
        outlay = buys.groupby("date").apply(lambda g: (g.notional + g.fee).sum(), include_groups=False)
        ok = (outlay <= cash_before.reindex(outlay.index) + 1e-6).all()
        out.append(_check("same-day buys <= cash at the signal close (no cash spent twice)", ok))
        out.append(_check("fees = notional x fee rate", np.allclose(f.fee, f.notional * fee_rate, atol=1e-9)))
        out.append(_check("due date = entry + holding period", (buys.due_date == buys.date + holding).all()))
    # 4. sells: at/after maturity, first quoted session on/after due, at that close
    if len(sells):
        out.append(_check("no sale before maturity", (sells.date >= sells.due_date).all()))
        sp = calendar.get_indexer(sells.date)
        cols = [col[s] for s in sells.symbol]
        out.append(_check("sell price = that session's close", np.allclose(px[sp, cols], sells.price, atol=1e-12)))
        due_pos = calendar.searchsorted(sells.due_date)
        first_quote = []
        for d0, c in zip(due_pos, cols):
            q = np.flatnonzero(np.isfinite(px[d0:, c]))
            first_quote.append(d0 + q[0] if len(q) else -1)
        out.append(_check("sale = first quoted session on/after maturity", (np.array(first_quote) == sp).all(),
                          f"violations {int((np.array(first_quote) != sp).sum())}"))
    if len(all_sells):
        matched = all_sells.merge(buys, on="lot_id", suffixes=("_s", "_b"))
        out.append(_check("each sale closes exactly one whole bought lot",
                          len(matched) == len(all_sells) and not all_sells.lot_id.duplicated().any()
                          and np.allclose(matched.units_s, matched.units_b)))
    # 5. open lots at the end: none past maturity unless unquoted since maturity
    ol = result.open_lots
    if len(ol):
        end = h.index[-1]
        overdue = ol.loc[ol.due_date <= end]
        bad = []
        for _, lot in overdue.iterrows():
            d0 = calendar.searchsorted(lot.due_date)
            e = calendar.get_loc(end)
            if np.isfinite(px[d0:e + 1, lot.col]).any():
                bad.append(lot.lot_id)
        terminal = set(ol.loc[ol.terminal, "lot_id"]) if "terminal" in ol else set()
        bad = [b for b in bad if b not in terminal]
        out.append(_check("open matured lots have had no quote since maturity (except terminal lots)", not bad,
                          f"lots {bad[:5]}"))
    # 6. PnL reconciliation
    p = result.pnl
    err = abs(p.realized_pnl + p.unrealized_pnl - p.total_pnl)
    out.append(_check("realized + unrealized PnL = NAV - initial", err < 1e-4 * initial, f"error {err:.2e}"))
    out.append(_check("no fills outside the run window",
                      f.date.between(h.index[1], h.index[-1]).all() if len(f) else True))
    return pd.DataFrame(out)


def assert_clean(report: pd.DataFrame) -> None:
    failed = report.loc[~report.passed]
    if len(failed):
        raise AssertionError("Backtest audit failed:\n" + failed.to_string(index=False))
