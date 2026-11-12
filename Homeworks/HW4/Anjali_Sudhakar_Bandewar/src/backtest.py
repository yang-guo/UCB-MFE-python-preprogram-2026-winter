"""HW4 backtester: 12-month momentum, top 50, overlapping 90-day tranches.

Each close t: rank eligible universe members by momentum_12m (data <= t),
set aside min(cash, NAV_t / 62) and split it equally across the top 50.
Orders fill at the next session's close. Each purchase is held 90 calendar
days and sold at the first executable close in the scheduled exit session or
the next five sessions.

Two price types are kept strictly apart:
- executable fill price: adj_close on a market session with adj_close > 0
  AND volume > 0. Only these are used for buys and sells.
- valuation mark: the latest adj_close > 0 on/before a session, any volume.
  Used only to value open positions; never a trade price.
"""
import json
import time
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from exclusions import EXCLUDED_SYMBOLS, excluded_symbols_sql

DB = Path(
    "Homeworks/HW4/Anjali_Sudhakar_Bandewar/data/equities_top2000.duckdb"
)
OUTPUT_DIR = Path("Homeworks/HW4/Anjali_Sudhakar_Bandewar/results/backtest")

FIRST_SIGNAL = pd.Timestamp("2016-03-31")  # first entry fills at 2016-04-01
TEST_START = pd.Timestamp("2016-04-01")
TEST_END = pd.Timestamp("2026-09-30")      # no price after this date is read

INITIAL_CAPITAL = 1_000_000.0
TOP_N = 50
HOLDING_DAYS = 90
TRANCHE_SESSIONS = 62          # median sessions in 90 calendar days
MAX_EXIT_DELAY_SESSIONS = 5
MIN_MEDIAN_DOLLAR_VOLUME = 5_000_000.0
LIQUIDITY_WINDOW = 63
MIN_TRADED_SESSIONS = 57       # >= 90% of the 63-session window
RISK_FREE = 0.02

PRIMARY = {"cost_bps": 10.0, "exit_rule": "write_off"}
RUNS = {
    "primary_10bps": PRIMARY,
    "cost_0bps": {"cost_bps": 0.0, "exit_rule": "write_off"},
    "cost_25bps": {"cost_bps": 25.0, "exit_rule": "write_off"},
    "next_valid_exit_10bps": {"cost_bps": 10.0, "exit_rule": "next_valid"},
}

# Verified ticker continuity: the same listed shares under a new symbol.
# After link_date (the old ticker's last row in the database) a position in
# the old ticker is priced from the successor, scaled by the adj_close ratio
# on link_date so returns are continuous. No other mappings are assumed.
# - GPS -> GAP: Gap Inc. renamed its NYSE symbol effective 2024-08-22, same
#   CUSIP, no shareholder action (company release, OCC memo). The database
#   repeats identical closes under both tickers until GPS ends on 2024-09-18.
# - DWAC -> DJT: Digital World renamed itself Trump Media & Technology Group
#   on 2024-03-25 and its Class A shares were redesignated as the new common
#   stock (SEC 8-K, CIK 1849635); DJT trading began 2024-03-26.
TICKER_CONTINUITY = {
    "GPS": {"successor": "GAP", "link_date": "2024-09-18"},
    "DWAC": {"successor": "DJT", "link_date": "2024-03-25"},
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_signals(con):
    """Top-50 selections per signal date, using only data on/before that date."""
    return con.execute(f"""
    WITH px AS (
        SELECT symbol, date, close, adj_close, volume
        FROM prices
        WHERE date <= DATE '{TEST_END.date()}'
    ),
    sessions AS (
        SELECT date, ROW_NUMBER() OVER (ORDER BY date) AS sn
        FROM calendar
        WHERE date <= DATE '{TEST_END.date()}'
    ),
    next_session AS (
        SELECT date, LEAD(date) OVER (ORDER BY date) AS entry_date FROM sessions
    ),
    momentum AS (
        -- Same definition as build_price_features.py: 252 positive-price rows.
        SELECT symbol, date,
               adj_close / LAG(adj_close, 252) OVER (
                   PARTITION BY symbol ORDER BY date) - 1 AS momentum_12m
        FROM px
        WHERE adj_close > 0
    ),
    liquidity AS (
        -- Trailing 63 market sessions ending at the signal date. close and
        -- volume are both split-adjusted, so close * volume is split-invariant.
        SELECT p.symbol, p.date,
               p.adj_close > 0 AND p.volume > 0 AS traded_today,
               MEDIAN(p.close * p.volume) OVER w AS median_dollar_volume,
               SUM(CASE WHEN p.volume > 0 AND p.adj_close > 0 THEN 1 ELSE 0 END)
                   OVER w AS traded_sessions
        FROM px p
        JOIN sessions s ON p.date = s.date
        WINDOW w AS (PARTITION BY p.symbol ORDER BY s.sn
                     RANGE BETWEEN {LIQUIDITY_WINDOW - 1} PRECEDING AND CURRENT ROW)
    ),
    eligible AS (
        SELECT
            l.symbol,
            l.date AS signal_date,
            n.entry_date,
            m.momentum_12m,
            l.median_dollar_volume,
            l.traded_sessions,
            u.universe_rank,
            u.selection_date,
            COUNT(*) OVER (PARTITION BY l.date) AS eligible_count
        FROM liquidity l
        JOIN momentum m ON m.symbol = l.symbol AND m.date = l.date
        JOIN next_session n ON n.date = l.date
        -- Membership on the entry date; selection_date precedes effective_start,
        -- so it is known at the signal close.
        JOIN universe_membership u
            ON u.symbol = l.symbol
           AND n.entry_date BETWEEN u.effective_start AND u.quarter_end
        WHERE l.date BETWEEN DATE '{FIRST_SIGNAL.date()}' AND DATE '{TEST_END.date()}'
          AND n.entry_date IS NOT NULL
          AND l.symbol NOT IN {excluded_symbols_sql()}
          AND l.traded_today
          AND m.momentum_12m IS NOT NULL
          AND l.median_dollar_volume >= {MIN_MEDIAN_DOLLAR_VOLUME}
          AND l.traded_sessions >= {MIN_TRADED_SESSIONS}
    )
    SELECT *
    FROM eligible
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY signal_date ORDER BY momentum_12m DESC, symbol
    ) <= {TOP_N}
    ORDER BY signal_date, momentum_12m DESC, symbol
    """).df()


def load_price_grids(con, symbols, sessions):
    """Session x symbol grids of executable prices and valuation marks."""
    con.register("sel", pd.DataFrame({"symbol": symbols}))
    px = con.execute(f"""
        SELECT p.symbol, p.date, p.adj_close, p.volume
        FROM prices p
        SEMI JOIN sel ON sel.symbol = p.symbol
        SEMI JOIN calendar c ON c.date = p.date
        WHERE p.date BETWEEN DATE '{FIRST_SIGNAL.date()}' AND DATE '{TEST_END.date()}'
          AND p.adj_close > 0
    """).df()
    px["date"] = pd.to_datetime(px["date"])
    px["source_symbol"] = px["symbol"]
    px["source_ratio"] = 1.0

    # Extend continuity-mapped tickers with their successor's prices.
    for old, link in TICKER_CONTINUITY.items():
        if old not in symbols:
            continue
        new, link_date = link["successor"], link["link_date"]
        old_adj, new_adj = con.execute(f"""
            SELECT o.adj_close, n.adj_close FROM prices o
            JOIN prices n ON n.date = o.date AND n.symbol = '{new}'
            WHERE o.symbol = '{old}' AND o.date = DATE '{link_date}'
        """).fetchall()[0]
        assert con.execute(f"SELECT MAX(date) FROM prices WHERE symbol = '{old}'"
                           ).fetchall()[0][0] == pd.Timestamp(link_date).date()
        ext = con.execute(f"""
            SELECT p.date, p.adj_close, p.volume
            FROM prices p
            SEMI JOIN calendar c ON c.date = p.date
            WHERE p.symbol = '{new}'
              AND p.date > DATE '{link_date}'
              AND p.date <= DATE '{TEST_END.date()}'
              AND p.adj_close > 0
        """).df()
        ext["date"] = pd.to_datetime(ext["date"])
        ratio = old_adj / new_adj
        ext = ext.assign(symbol=old, adj_close=ext["adj_close"] * ratio,
                         source_symbol=new, source_ratio=ratio)
        px = pd.concat([px, ext[px.columns]], ignore_index=True)
    assert px["date"].max() <= TEST_END
    assert not px.duplicated(["symbol", "date"]).any()

    raw = px.pivot(index="date", columns="symbol", values="adj_close").reindex(
        index=sessions, columns=symbols)
    vol = px.pivot(index="date", columns="symbol", values="volume").reindex(
        index=sessions, columns=symbols)
    executable = raw.where(vol.fillna(0) > 0)          # fills only
    marks = raw.ffill()                                 # valuation only
    session_idx = pd.DataFrame(
        np.where(raw.notna(), np.arange(len(sessions))[:, None], np.nan),
        index=sessions, columns=symbols).ffill()        # session of last mark
    return executable.to_numpy(), marks.to_numpy(), session_idx.to_numpy(), px


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def simulate(signals, sessions, executable, marks, mark_session, sym_index,
             cost_bps, exit_rule):
    cost = cost_bps / 1e4
    n_sess = len(sessions)
    by_signal = {d: g for d, g in signals.groupby("signal_date")}

    lots = []          # list of dicts, one per filled purchase
    open_lots = []     # indices into lots
    cash = INITIAL_CAPITAL
    daily = []
    pending = []       # orders planned at the previous close
    pending_cash = 0.0

    for i, d in enumerate(sessions):
        # 1) Exits due at this close.
        still_open = []
        for k in open_lots:
            lot = lots[k]
            s = lot["sym_idx"]
            if i < lot["sched_exit_idx"]:
                still_open.append(k)
                continue
            delay = i - lot["sched_exit_idx"]
            px = executable[i, s]
            if not np.isnan(px) and (delay <= MAX_EXIT_DELAY_SESSIONS or exit_rule == "next_valid"):
                gross = lot["units"] * px
                fee = gross * cost
                cash += gross - fee
                lot.update(exit_date=d, exit_price=px, exit_gross=gross, exit_fee=fee,
                           exit_delay_sessions=delay,
                           status="closed" if delay <= MAX_EXIT_DELAY_SESSIONS else "closed_late")
            elif delay >= MAX_EXIT_DELAY_SESSIONS and exit_rule == "write_off":
                lot.update(exit_date=d, exit_price=0.0, exit_gross=0.0, exit_fee=0.0,
                           exit_delay_sessions=delay, status="written_off")
            else:
                still_open.append(k)
        open_lots = still_open

        # 2) Fill orders planned at the previous close (cash already set aside).
        bought = 0.0
        for sym, s, alloc, signal_date in pending:
            px = executable[i, s]
            if np.isnan(px):
                cash += alloc   # no executable price: budget returns to cash
                lots.append(dict(symbol=sym, sym_idx=s, signal_date=signal_date,
                                 entry_date=d, status="entry_unfilled",
                                 budget=alloc))
                continue
            gross = alloc / (1 + cost)
            fee = gross * cost
            sched = sessions.searchsorted(d + pd.Timedelta(days=HOLDING_DAYS))
            lots.append(dict(symbol=sym, sym_idx=s, signal_date=signal_date,
                             entry_date=d, entry_price=px, units=gross / px,
                             entry_gross=gross, entry_fee=fee, budget=alloc,
                             maturity_date=d + pd.Timedelta(days=HOLDING_DAYS),
                             sched_exit_idx=sched,
                             scheduled_exit_date=sessions[sched] if sched < n_sess else pd.NaT,
                             status="open"))
            open_lots.append(len(lots) - 1)
            bought += alloc
        pending, pending_cash = [], 0.0

        # 3) Mark open positions (valuation only).
        idx = np.array([lots[k]["sym_idx"] for k in open_lots], dtype=int)
        units = np.array([lots[k]["units"] for k in open_lots])
        lot_marks = marks[i, idx] if len(idx) else np.array([])
        stale = (i - mark_session[i, idx]) if len(idx) else np.array([])
        positions_value = float(np.nansum(units * lot_marks))
        nav = cash + positions_value

        # 4) Signal at this close; orders fill at the next close.
        n_selected = 0
        if i + 1 < n_sess and d in by_signal:
            budget = min(cash, nav / TRANCHE_SESSIONS)
            per_stock = budget / TOP_N
            for row in by_signal[d].itertuples():
                pending.append((row.symbol, sym_index[row.symbol], per_stock, d))
            n_selected = len(pending)
            pending_cash = per_stock * n_selected
            cash -= pending_cash
        assert cash >= -1e-6, (d, cash)

        weights = {}
        if len(idx):
            vals = pd.Series(units * lot_marks).groupby(idx).sum()
            weights = vals / nav
        daily.append(dict(
            date=d, cash=cash + pending_cash, positions_value=positions_value, nav=nav,
            open_lots=len(open_lots), bought_today=bought, n_selected=n_selected,
            stale_marks_gt5=int((stale > MAX_EXIT_DELAY_SESSIONS).sum()) if len(idx) else 0,
            max_stock_weight=float(weights.max()) if len(idx) else 0.0,
        ))

    trades = pd.DataFrame(lots).drop(columns=["sym_idx", "sched_exit_idx"])
    final = daily[-1]
    for k in open_lots:
        lot = lots[k]
        lot_px = marks[n_sess - 1, lot["sym_idx"]]
        trades.loc[k, "end_mark"] = lot_px
        trades.loc[k, "end_value"] = lot["units"] * lot_px
    return trades, pd.DataFrame(daily), cash


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def performance(values, start_value=INITIAL_CAPITAL):
    """values: Series indexed by date from the first test close."""
    path = pd.concat([pd.Series([start_value]), values.reset_index(drop=True)])
    rets = path.pct_change().dropna()
    years = (values.index[-1] - values.index[0]).days / 365.25
    rf_daily = (1 + RISK_FREE) ** (1 / 252) - 1
    excess = rets - rf_daily
    return {
        "final_value": float(values.iloc[-1]),
        "total_return": float(values.iloc[-1] / start_value - 1),
        "cagr": float((values.iloc[-1] / start_value) ** (1 / years) - 1),
        "max_drawdown": float((path / path.cummax() - 1).min()),
        "sharpe": float(excess.mean() / excess.std() * np.sqrt(252)),
        "ann_volatility": float(rets.std() * np.sqrt(252)),
    }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def attach_price_sources(trades, px_used):
    """Record which database ticker and scale factor each fill came from."""
    src = px_used[["symbol", "date", "source_symbol", "source_ratio"]]
    for side in ("entry", "exit"):
        trades = trades.merge(
            src.rename(columns={"date": f"{side}_date",
                                "source_symbol": f"{side}_source_symbol",
                                "source_ratio": f"{side}_source_ratio"}),
            on=["symbol", f"{side}_date"], how="left")
    return trades


def validate(con, signals, trades, daily, final_cash, sessions, cost_bps, px_used):
    cost = cost_bps / 1e4
    checks = {}
    filled = trades[trades["status"] != "entry_unfilled"]
    exited = filled[filled["status"].isin(["closed", "closed_late"])]

    # Look-ahead
    checks["entry_not_after_signal"] = int((trades["entry_date"] <= trades["signal_date"]).sum())
    nxt = pd.Series(sessions[1:], index=sessions[:-1])
    checks["entry_not_next_session"] = int((trades["entry_date"] != trades["signal_date"].map(nxt)).sum())
    checks["selection_after_signal"] = int((pd.to_datetime(signals["selection_date"]) > signals["signal_date"]).sum())
    checks["prices_after_test_end"] = int((px_used["date"] > TEST_END).sum())
    checks["trade_dates_after_test_end"] = int(
        (filled["entry_date"] > TEST_END).sum() + (exited["exit_date"] > TEST_END).sum())
    checks["excluded_symbols_traded"] = int(filled["symbol"].isin(EXCLUDED_SYMBOLS).sum())

    # Independent recomputation of momentum and liquidity for a sample.
    sample = signals.sample(min(300, len(signals)), random_state=0)
    bad_mom = bad_liq = 0
    for r in sample.itertuples():
        hist = con.execute("""
            SELECT adj_close FROM prices
            WHERE symbol = ? AND date <= ? AND adj_close > 0 ORDER BY date DESC LIMIT 253
        """, [r.symbol, r.signal_date.date()]).fetchnumpy()["adj_close"]
        if abs(hist[0] / hist[252] - 1 - r.momentum_12m) > 1e-9:
            bad_mom += 1
        liq = con.execute("""
            SELECT MEDIAN(p.close * p.volume), SUM(CASE WHEN p.volume > 0 AND p.adj_close > 0 THEN 1 ELSE 0 END)
            FROM prices p
            WHERE p.symbol = ? AND p.date IN (
                SELECT date FROM calendar WHERE date <= ? ORDER BY date DESC LIMIT 63)
        """, [r.symbol, r.signal_date.date()]).fetchall()[0]
        if abs(liq[0] - r.median_dollar_volume) > 1e-6 * liq[0] or liq[1] != r.traded_sessions:
            bad_liq += 1
    checks["momentum_recompute_mismatch_of_300"] = bad_mom
    checks["liquidity_recompute_mismatch_of_300"] = bad_liq

    # Executable fills only, checked against the database row actually used
    # (the successor ticker's row for continuity-mapped fills).
    con.register("t_entries", filled[["entry_source_symbol", "entry_source_ratio",
                                      "entry_date", "entry_price"]])
    con.register("t_exits", exited[["exit_source_symbol", "exit_source_ratio",
                                    "exit_date", "exit_price"]])
    checks["entry_fill_not_executable"] = con.execute("""
        SELECT COUNT(*) FROM t_entries t LEFT JOIN prices p
          ON p.symbol = t.entry_source_symbol AND p.date = CAST(t.entry_date AS DATE)
        WHERE p.volume IS NULL OR p.volume <= 0
           OR abs(p.adj_close * t.entry_source_ratio - t.entry_price) > 1e-9 * t.entry_price
    """).fetchall()[0][0]
    checks["exit_fill_not_executable"] = con.execute("""
        SELECT COUNT(*) FROM t_exits t LEFT JOIN prices p
          ON p.symbol = t.exit_source_symbol AND p.date = CAST(t.exit_date AS DATE)
        WHERE p.volume IS NULL OR p.volume <= 0
           OR abs(p.adj_close * t.exit_source_ratio - t.exit_price) > 1e-9 * t.exit_price
    """).fetchall()[0][0]
    # Successor prices only for mapped tickers, and only after the link date.
    link = pd.to_datetime(pd.Series({k: v["link_date"] for k, v in TICKER_CONTINUITY.items()}))
    outside = 0
    for frame, side in ((filled, "entry"), (exited, "exit")):
        mapped = frame[f"{side}_source_symbol"] != frame["symbol"]
        after_link = frame[f"{side}_date"] > frame["symbol"].map(link)
        outside += int((mapped & ~after_link.fillna(False)).sum())   # too early / unmapped
        outside += int((~mapped & after_link.fillna(False)).sum())   # old ticker after link
    checks["mapped_fill_outside_link_window"] = outside
    checks["exit_before_maturity"] = int((exited["exit_date"] < exited["maturity_date"]).sum())
    checks["exit_delay_over_5_closed"] = int(
        (filled.loc[filled["status"] == "closed", "exit_delay_sessions"] > MAX_EXIT_DELAY_SESSIONS).sum())

    # Sizing and cash
    per_day = trades.groupby("signal_date")["budget"].agg(["sum", "count", "max"])
    nav_at_signal = daily.set_index("date")["nav"]
    checks["orders_per_day_over_50"] = int((per_day["count"] > TOP_N).sum())
    checks["day_budget_over_nav_over_62"] = int(
        (per_day["sum"] > nav_at_signal.reindex(per_day.index) / TRANCHE_SESSIONS + 1e-6).sum())
    checks["duplicate_symbol_same_signal"] = int(trades.duplicated(["signal_date", "symbol"]).sum())
    checks["fee_mismatch"] = int((abs(filled["entry_fee"] - filled["entry_gross"] * cost) > 1e-6).sum()
                                 + (abs(exited["exit_fee"] - exited["exit_gross"] * cost) > 1e-6).sum())
    checks["negative_cash_days"] = int((daily["cash"] < -1e-6).sum())
    checks["max_single_stock_weight_over_5pct"] = int((daily["max_stock_weight"] > 0.05).sum())

    # Reconciliation
    cash_recon = (INITIAL_CAPITAL - (filled["entry_gross"] + filled["entry_fee"]).sum()
                  + (exited["exit_gross"] - exited["exit_fee"]).sum())
    checks["cash_reconciliation_error"] = round(abs(cash_recon - final_cash), 6)
    open_value = trades.loc[trades["status"] == "open", "end_value"].sum()
    checks["nav_reconciliation_error"] = round(abs(final_cash + open_value - daily["nav"].iloc[-1]), 6)
    realized = (exited["exit_gross"] - exited["entry_gross"]).sum() \
        - filled.loc[filled["status"] == "written_off", "entry_gross"].sum()
    unrealized = (trades.loc[trades["status"] == "open", "end_value"]
                  - trades.loc[trades["status"] == "open", "entry_gross"]).sum()
    fees = filled["entry_fee"].sum() + exited["exit_fee"].sum()
    checks["pnl_reconciliation_error"] = round(
        abs(realized + unrealized - fees - (daily["nav"].iloc[-1] - INITIAL_CAPITAL)), 6)
    return checks


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB), read_only=True)

    sessions = pd.DatetimeIndex(pd.to_datetime(con.execute(f"""
        SELECT date FROM calendar
        WHERE date BETWEEN DATE '{FIRST_SIGNAL.date()}' AND DATE '{TEST_END.date()}'
        ORDER BY date
    """).df()["date"]))
    assert sessions[0] == FIRST_SIGNAL and sessions[1] == TEST_START and sessions[-1] == TEST_END

    signals = load_signals(con)
    signals["signal_date"] = pd.to_datetime(signals["signal_date"])
    signals["entry_date"] = pd.to_datetime(signals["entry_date"])
    print(f"Signals: {len(signals):,} selections over {signals['signal_date'].nunique()} dates "
          f"({time.perf_counter() - t0:.0f}s)")

    symbols = sorted(signals["symbol"].unique())
    sym_index = {s: j for j, s in enumerate(symbols)}
    executable, marks, mark_session, px_used = load_price_grids(con, symbols, sessions)

    bench = con.execute(f"""
        SELECT date, adj_close FROM benchmark
        WHERE symbol = '^SP500TR'
          AND date BETWEEN DATE '{TEST_START.date()}' AND DATE '{TEST_END.date()}'
        ORDER BY date
    """).df()
    bench["date"] = pd.to_datetime(bench["date"])
    bench = bench.set_index("date")["adj_close"]
    benchmark_value = INITIAL_CAPITAL / bench.iloc[0] * bench
    assert bench.index[0] == TEST_START and bench.index[-1] == TEST_END

    results, all_checks, outputs = {}, {}, {}
    for name, cfg in RUNS.items():
        trades, daily, final_cash = simulate(signals, sessions, executable, marks,
                                             mark_session, sym_index, **cfg)
        trades = attach_price_sources(trades, px_used)
        nav = daily.set_index("date")["nav"].loc[TEST_START:]
        results[name] = performance(nav)
        all_checks[name] = validate(con, signals, trades, daily, final_cash, sessions,
                                    cfg["cost_bps"], px_used)
        outputs[name] = (trades, daily)
        print(f"  {name}: final ${results[name]['final_value']:,.0f} "
              f"({time.perf_counter() - t0:.0f}s)")
    results["benchmark_SP500TR"] = performance(benchmark_value)

    trades, daily = outputs["primary_10bps"]
    daily = daily.set_index("date")
    daily["benchmark_value"] = benchmark_value
    daily["cash_weight"] = daily["cash"] / daily["nav"]
    trades.to_csv(OUTPUT_DIR / "trades.csv", index=False)
    daily.to_csv(OUTPUT_DIR / "daily.csv")
    signals.to_csv(OUTPUT_DIR / "signals.csv", index=False)

    # Ramp-up and cash drag
    test_daily = daily.loc[TEST_START:]
    ramp_end = test_daily.index[TRANCHE_SESSIONS - 1]
    post = test_daily.loc[ramp_end:]
    ramp = {
        "ramp_end_date": str(ramp_end.date()),
        "cash_weight_day1": float(test_daily["cash_weight"].iloc[0]),
        "cash_weight_at_ramp_end": float(test_daily["cash_weight"].loc[ramp_end]),
        "avg_cash_weight_ramp": float(test_daily["cash_weight"].loc[:ramp_end].mean()),
        "avg_cash_weight_after_ramp": float(post["cash_weight"].mean()),
        "strategy_return_during_ramp": float(test_daily["nav"].loc[ramp_end] / INITIAL_CAPITAL - 1),
        "benchmark_return_during_ramp": float(benchmark_value.loc[ramp_end] / INITIAL_CAPITAL - 1),
        "post_ramp_strategy": performance(post["nav"].iloc[1:], post["nav"].iloc[0]),
        "post_ramp_benchmark": performance(benchmark_value.loc[ramp_end:].iloc[1:],
                                           benchmark_value.loc[ramp_end]),
    }

    # Write-offs and their next-valid-exit sensitivity
    wo = trades[trades["status"] == "written_off"].copy()
    alt_trades = outputs["next_valid_exit_10bps"][0]
    alt = alt_trades.set_index(["symbol", "signal_date"])
    for col in ("exit_date", "exit_price", "exit_delay_sessions", "status", "end_value"):
        if col in alt:
            wo["alt_" + col] = [alt[col].get((r.symbol, r.signal_date)) for r in wo.itertuples()]
    wo.to_csv(OUTPUT_DIR / "write_offs.csv", index=False)

    status_counts = trades["status"].value_counts().to_dict()
    metrics = {
        "config": {"first_signal": str(FIRST_SIGNAL.date()), "test_start": str(TEST_START.date()),
                   "test_end": str(TEST_END.date()), "top_n": TOP_N, "holding_days": HOLDING_DAYS,
                   "tranche_sessions": TRANCHE_SESSIONS, "min_median_dollar_volume": MIN_MEDIAN_DOLLAR_VOLUME,
                   "min_traded_sessions_of_63": MIN_TRADED_SESSIONS, "excluded": list(EXCLUDED_SYMBOLS),
                   "cash_interest": 0.0, "risk_free": RISK_FREE},
        "performance": results, "ramp_up": ramp, "trade_status_primary": status_counts,
        "validation": all_checks,
        "selection_stats": {
            "signal_dates": int(signals["signal_date"].nunique()),
            "days_with_fewer_than_50": int((signals.groupby("signal_date").size() < TOP_N).sum()),
            "min_eligible_count": int(signals["eligible_count"].min()),
            "median_eligible_count": float(signals["eligible_count"].median()),
            "distinct_symbols_selected": len(symbols),
        },
        "diagnostics_primary": {
            "max_stock_weight": float(daily["max_stock_weight"].max()),
            "days_with_stale_marks_gt5": int((daily["stale_marks_gt5"] > 0).sum()),
            "max_stale_lots_one_day": int(daily["stale_marks_gt5"].max()),
            "open_lots_at_end": int((trades["status"] == "open").sum()),
            "open_value_at_end": float(trades.loc[trades["status"] == "open", "end_value"].sum()),
            "liquidation_fee_at_end": float(trades.loc[trades["status"] == "open", "end_value"].sum()
                                            * PRIMARY["cost_bps"] / 1e4),
            "total_fees": float(trades["entry_fee"].sum() + trades["exit_fee"].fillna(0).sum()),
            "annual_turnover_x_nav": float(
                (trades["entry_gross"].sum() + trades["exit_gross"].fillna(0).sum())
                / test_daily["nav"].mean() / ((TEST_END - TEST_START).days / 365.25)),
        },
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    pd.DataFrame(results).T.to_csv(OUTPUT_DIR / "metrics.csv")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name, style in (("primary_10bps", "-"), ("cost_25bps", ":")):
        s = outputs[name][1].set_index("date")["nav"].loc[TEST_START:]
        ax.plot(s.index, s / INITIAL_CAPITAL - 1, style, label=f"Momentum top-50 ({name})")
    ax.plot(benchmark_value.index, benchmark_value / INITIAL_CAPITAL - 1, "-", color="black",
            label="^SP500TR buy & hold")
    ax.axvline(ramp_end, color="grey", lw=0.8, ls="--", label="end of 62-session ramp-up")
    ax.set_title("Cumulative return, Apr 1 2016 - Sep 30 2026 ($1M start)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cumulative_returns.png", dpi=150)

    print("\nPERFORMANCE")
    print(pd.DataFrame(results).T.to_string(float_format=lambda x: f"{x:,.4f}"))
    print("\nRAMP-UP"); print(json.dumps(ramp, indent=2))
    print("\nPRIMARY TRADE STATUS", status_counts)
    print("\nSELECTION", metrics["selection_stats"])
    print("\nDIAGNOSTICS", json.dumps(metrics["diagnostics_primary"], indent=2))
    print("\nVALIDATION (all counts/errors should be 0)")
    print(pd.DataFrame(all_checks).to_string())
    con.close()
    print(f"\nDone in {time.perf_counter() - t0:.0f}s; outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
