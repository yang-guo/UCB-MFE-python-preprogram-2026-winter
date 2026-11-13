"""POST-HOC DIAGNOSTICS of the frozen final run. Not used to change the strategy.

Read-only: reads outputs/final/ (saved results of the one frozen run), the full
panel and the database. Re-runs no backtest and fits no model. Writes tables to
outputs/final_diagnostics/.

D1 yearly returns and drawdowns        D5 test-period rank IC and top-5% gap
D2 stale-rule capital and cash by year D6 universe vs purchased 90-day returns
D3 ORLY / KLAC split-like jumps         D7 names of the 34 stale-rule stocks
D4 large one-day moves on held lots     D8 fees, turnover, distinct holdings

Run from the repository root:
    .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/final_diagnostics.py

Credits: no lecture code reused here (the inputs come from course_data.py-based
outputs). Written with AI assistance (Claude Code) and reviewed by the author.
"""

import sys
import warnings
from pathlib import Path

sys.dont_write_bytecode = True

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage4_dev_grid as g  # noqa: E402  (prediction_quality, same definitions as development)

FINAL = HERE / "outputs" / "final"
OUT = HERE / "outputs" / "final_diagnostics"
PANEL = HERE / "outputs" / "panel_full"
DATABASE = HERE.parents[2] / "Lectures" / "modeling" / "data" / "equities_top2000.duckdb"
START, END = pd.Timestamp("2016-03-31"), pd.Timestamp("2026-09-30")
LABEL = "POST-HOC DIAGNOSTIC, not used to change the strategy"


def save(df, name):
    df.to_csv(OUT / f"{name}.csv")
    return df


def drawdown_info(nav):
    peak = nav.cummax()
    dd = nav / peak - 1
    trough = dd.idxmin()
    peak_date = nav.loc[:trough].idxmax()
    return dd.min(), peak_date, trough


def load_lots():
    fills = pd.read_csv(FINAL / "fills.csv", parse_dates=["date", "signal_date", "due_date"])
    buys = fills.loc[fills.side.eq("buy")].set_index("lot_id")
    sells = fills.loc[fills.side.eq("sell")].set_index("lot_id")
    lots = buys[["symbol", "date", "units", "price", "gross", "signal_date"]].rename(
        columns={"date": "entry_date", "price": "entry_price", "gross": "entry_gross"})
    lots["exit_date"] = sells.date.reindex(lots.index)
    lots["held_to"] = lots.exit_date.fillna(END)
    return fills, lots


def d1(h):
    rows = []
    years = sorted(h.index.year.unique())
    for y in years:
        prev = h.loc[h.index < pd.Timestamp(f"{y}-01-01")]
        cur = h.loc[h.index.year == y]
        base = prev.iloc[-1] if len(prev) else cur.iloc[0]
        window = pd.concat([prev.iloc[-1:], cur]) if len(prev) else cur
        row = {"year": y, "from": window.index[0].date(), "to": cur.index[-1].date()}
        for col, name in [("nav", "strategy"), ("benchmark_nav", "benchmark")]:
            row[f"{name}_return"] = cur[col].iloc[-1] / base[col] - 1
            dd, _, trough = drawdown_info(window[col])
            row[f"{name}_max_dd"], row[f"{name}_trough"] = dd, trough.date()
        rows.append(row)
    total = {"year": "2016-03-31 to 2026-09-30", "from": h.index[0].date(), "to": h.index[-1].date()}
    for col, name in [("nav", "strategy"), ("benchmark_nav", "benchmark")]:
        total[f"{name}_return"] = h[col].iloc[-1] / h[col].iloc[0] - 1
        dd, peak, trough = drawdown_info(h[col])
        total[f"{name}_max_dd"], total[f"{name}_trough"] = dd, f"{trough.date()} (peak {peak.date()})"
    rows.append(total)
    return save(pd.DataFrame(rows).set_index("year"), "D1_yearly_returns_drawdowns")


def d2(h, lots):
    lg = pd.read_csv(FINAL / "long_gaps.csv", parse_dates=["last_quote_date", "flagged_on"])
    stale_value = pd.Series(0.0, index=h.index)
    for _, r in lg.iterrows():          # frozen from the day after its last quote
        stale_value[h.index > r.last_quote_date] += r.units * r.last_mark
    df = pd.DataFrame({"stale_share": stale_value / h.nav, "cash_share": h.cash / h.nav,
                       "invested_share": h.holdings / h.nav,
                       "active_invested_share": (h.holdings - stale_value) / h.nav})
    out = df.groupby(df.index.year).agg(stale_avg=("stale_share", "mean"), stale_max=("stale_share", "max"),
                                        cash_avg=("cash_share", "mean"), invested_avg=("invested_share", "mean"),
                                        active_invested_avg=("active_invested_share", "mean"))
    out.loc["whole period"] = [df.stale_share.mean(), df.stale_share.max(), df.cash_share.mean(),
                               df.invested_share.mean(), df.active_invested_share.mean()]
    out.loc["last day (2026-09-30)"] = [df.stale_share.iloc[-1], df.stale_share.iloc[-1], df.cash_share.iloc[-1],
                                        df.invested_share.iloc[-1], df.active_invested_share.iloc[-1]]
    return save(out, "D2_stale_capital_and_cash")


def quotes(symbols):
    q = pd.read_parquet(PANEL, columns=["date", "symbol", "close", "adj_close"],
                        filters=[("date", ">=", START), ("date", "<=", END), ("symbol", "in", sorted(symbols))])
    return q.sort_values(["symbol", "date"]).reset_index(drop=True)


def d3(h, lots):
    jumps = pd.read_csv(FINAL / "jump_flags.csv", parse_dates=["date", "previous_mark_date"])
    q = quotes(jumps.symbol.unique())
    rows, summary = [], []
    for _, j in jumps.iterrows():
        open_across = lots.loc[lots.symbol.eq(j.symbol) & lots.entry_date.lt(j.date) & lots.held_to.ge(j.date)]
        s = q.loc[q.symbol.eq(j.symbol)].set_index("date")
        before, after = s.loc[j.previous_mark_date], s.loc[j.date]
        nav_before = h.nav.loc[j.previous_mark_date]
        ratio_adj, ratio_close = after.adj_close / before.adj_close, after.close / before.close
        k = round(1 / ratio_adj)
        for lot_id, l in open_across.iterrows():
            effect = l.units * (after.adj_close - before.adj_close)
            rows.append({"symbol": j.symbol, "jump_date": j.date.date(), "lot_id": lot_id, "units": l.units,
                         "entry_date": l.entry_date.date(), "entry_price": l.entry_price,
                         "mark_before": before.adj_close, "mark_after": after.adj_close,
                         "value_effect": effect, "effect_share_of_nav": effect / nav_before,
                         "exit_date": l.exit_date.date() if pd.notna(l.exit_date) else "open at end"})
        effect = (open_across.units * (after.adj_close - before.adj_close)).sum()
        summary.append({"symbol": j.symbol, "jump_date": j.date.date(), "lots_open_across": len(open_across),
                        "adj_close_ratio": ratio_adj, "close_ratio": ratio_close,
                        "nearest_split_ratio": f"1-for-{k}" if k > 1 else "none",
                        "value_effect_on_jump_day": effect, "share_of_nav": effect / nav_before,
                        "nav_before": nav_before,
                        # If it was an unadjusted k-for-1 split, the held value should have been unchanged:
                        "assumed_split_true_value_lost": -effect if k > 1 else 0.0})
    return save(pd.DataFrame(summary).set_index("symbol"), "D3_split_like_jumps"), \
        save(pd.DataFrame(rows), "D3_lots_open_across_jumps")


def d4(h, lots):
    q = quotes(lots.symbol.unique())
    q["adj_ratio"] = q.adj_close / q.groupby("symbol").adj_close.shift()
    q["close_ratio"] = q.close / q.groupby("symbol").close.shift()
    q["prev_adj"] = q.groupby("symbol").adj_close.shift()
    big = q.loc[q.adj_ratio.le(0.60) | q.adj_ratio.ge(1.60)].copy()
    rows = []
    for _, m in big.iterrows():
        held = lots.loc[lots.symbol.eq(m.symbol) & lots.entry_date.lt(m.date) & lots.held_to.ge(m.date)]
        if not len(held):
            continue
        effect = held.units.sum() * (m.adj_close - m.prev_adj)
        inv = 1 / m.adj_ratio if m.adj_ratio < 1 else m.adj_ratio
        k = round(inv)
        rows.append({"symbol": m.symbol, "date": m.date.date(), "adj_move": m.adj_ratio - 1,
                     "close_move": m.close_ratio - 1,
                     "close_moved_same_ratio": bool(abs(m.close_ratio / m.adj_ratio - 1) < 0.01),
                     "split_like_ratio": f"{'1-for-' if m.adj_ratio < 1 else ''}{k}" if k >= 2 and abs(inv / k - 1) < 0.03 else "",
                     "lots_held": len(held), "nav_effect": effect,
                     "share_of_nav": effect / h.nav.loc[h.index[h.index < m.date][-1]]})
    out = pd.DataFrame(rows).sort_values("nav_effect")
    return save(out.set_index(["symbol", "date"]), "D4_large_moves_on_held_lots")


def d5():
    scores = pd.read_parquet(FINAL / "scores.parquet", columns=["date", "symbol", "score"])
    targets = pd.read_parquet(PANEL, columns=["date", "symbol", "target", "label_end"],
                              filters=[("date", ">=", START), ("date", "<=", END)])
    targets = targets.loc[targets.target.notna() & targets.label_end.le(END)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test = g.prediction_quality(targets, scores)
    test = test.rename(index={"2012-2015": "2016-2026"})
    dev = pd.read_csv(HERE / "outputs" / "stage4_prediction_quality.csv", header=[0, 1], index_col=0)["annual"]
    dev.index = [f"dev {i}" for i in dev.index]
    out = pd.concat([test, dev])
    out.attrs["rows_with_targets"] = len(scores.merge(targets[["date", "symbol"]], on=["date", "symbol"]))
    return save(out, "D5_test_rank_ic_top5_gap"), scores, targets


def d6(scores, targets, fills, lots):
    universe = scores.merge(targets[["date", "symbol", "target"]], on=["date", "symbol"], how="left")
    buys = fills.loc[fills.side.eq("buy")]
    decision_dates = buys.signal_date.unique()
    u = universe.loc[universe.date.isin(decision_dates)]
    u_daily = u.groupby("date").target.mean()
    bought = buys.merge(targets[["date", "symbol", "target"]], left_on=["signal_date", "symbol"],
                        right_on=["date", "symbol"], how="left", suffixes=("", "_t"))
    closed = pd.read_csv(FINAL / "closed_lots.csv", parse_dates=["entry_date"])
    closed = closed.merge(buys[["lot_id", "signal_date"]], on="lot_id")
    rows = []
    for y in sorted(pd.DatetimeIndex(decision_dates).year.unique()):
        uy = u_daily[u_daily.index.year == y]
        by = bought.loc[bought.signal_date.dt.year.eq(y)]
        w = by.loc[by.target.notna()]
        cy = closed.loc[closed.signal_date.dt.year.eq(y)]
        rows.append({"signal_year": y, "decision_dates": len(uy),
                     "universe_ew_mean_90d": uy.mean(),
                     "bought_dollar_weighted_90d": np.average(w.target, weights=w.gross) if len(w) else np.nan,
                     "bought_lots": len(by), "bought_lots_without_target": int(by.target.isna().sum()),
                     "closed_lots_net_return_dollar_weighted": (cy.proceeds.sum() / cy.cost.sum() - 1) if len(cy) else np.nan})
    out = pd.DataFrame(rows).set_index("signal_year")
    out["bought_minus_universe"] = out.bought_dollar_weighted_90d - out.universe_ew_mean_90d
    w = bought.loc[bought.target.notna()]
    out.loc["all"] = [len(u_daily), u_daily.mean(), np.average(w.target, weights=w.gross), len(bought),
                      int(bought.target.isna().sum()), closed.proceeds.sum() / closed.cost.sum() - 1,
                      np.average(w.target, weights=w.gross) - u_daily.mean()]
    return save(out, "D6_universe_vs_bought_returns")


def d7():
    lg = pd.read_csv(FINAL / "long_gaps.csv")
    per = lg.groupby("symbol").agg(lots=("lot_id", "size"), last_quote=("last_quote_date", "first"),
                                   last_price=("last_mark", "first"), value=("value_at_end", "sum"))
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        u = con.execute("""SELECT symbol, name, sector, industry, exchange, is_active_snapshot, delisted_at_snapshot
                           FROM universe WHERE symbol IN (SELECT unnest(?))""", [list(per.index)]).fetchdf()
    out = per.join(u.set_index("symbol"))
    # Debt/preferred wording, including vendor abbreviations seen here ("SR NT", "6.50% NT 24", "7.00% Seni").
    words = (r"(?i)(?:\bnotes?\b|\bnt\b|\bsr\b|\bseni|\bpreferred\b|\bpfd\b|depositary|\btrust pref|\bunits?\b|"
             r"\bwarrants?\b|debenture|subordinated|%|\betn\b|exchange traded|\bcall\b|\bl\.p\.)")
    out["name_flag"] = out.name.fillna("").str.contains(words, regex=True)
    out["par_price_flag"] = [any(abs(p / par - 1) < 0.02 for par in (25.0, 50.0)) for p in out.last_price]
    out["possible_non_common"] = out.name_flag | out.par_price_flag
    return save(out.sort_values("last_quote"), "D7_stale_rule_stocks")


def d8(h, fills, lots):
    years = (h.index[-1] - h.index[0]).days / 365.25
    avg_nav = h.nav.mean()
    buys, sells = fills.loc[fills.side.eq("buy")], fills.loc[fills.side.eq("sell")]
    open_lots = pd.read_csv(FINAL / "open_lots.csv")
    lg = pd.read_csv(FINAL / "long_gaps.csv")
    distinct_daily = []
    out = pd.Series({
        "fees_total": fills.fee.sum(), "average_nav": avg_nav,
        "fees_share_of_average_nav_total": fills.fee.sum() / avg_nav,
        "fees_share_of_average_nav_per_year": fills.fee.sum() / avg_nav / years,
        "buys_dollars": buys.gross.sum(), "sells_dollars": sells.gross.sum(),
        "turnover_per_year_(min(buys,sells)/avg_nav)": min(buys.gross.sum(), sells.gross.sum()) / avg_nav / years,
        "buys_per_year_share_of_avg_nav": buys.gross.sum() / avg_nav / years,
        "distinct_stocks_held_at_end": open_lots.symbol.nunique(),
        "of_which_stale_rule": lg.loc[lg.still_open_at_end, "symbol"].nunique(),
        "open_lots_at_end": len(open_lots), "years": years})
    return save(out.to_frame("value"), "D8_fees_turnover_holdings")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.txt").write_text(LABEL + ". Read-only analysis of outputs/final/ (frozen run).\n", encoding="utf-8")
    h = pd.read_csv(FINAL / "history.csv", parse_dates=["date"], index_col="date")
    fills, lots = load_lots()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    fmt = lambda x: f"{x:,.4f}"  # noqa: E731
    print(f"=== {LABEL} ===")
    print("\nD1 yearly returns and drawdowns\n", d1(h).to_string(float_format=fmt))
    print("\nD2 stale-rule capital and cash (shares of NAV)\n", d2(h, lots).to_string(float_format=fmt))
    s3, l3 = d3(h, lots)
    print("\nD3 split-like jumps\n", s3.to_string(float_format=fmt))
    print(l3.to_string(index=False, float_format=fmt))
    d4_out = d4(h, lots)
    print(f"\nD4 one-day moves <= -40% or >= +60% while held: {len(d4_out)} events; "
          f"total NAV effect {d4_out.nav_effect.sum():,.0f}")
    print(d4_out.to_string(float_format=fmt))
    d5_out, scores, targets = d5()
    print(f"\nD5 rank IC and top-5% gap (test rows with finished outcomes: {d5_out.attrs['rows_with_targets']:,})\n",
          d5_out.to_string(float_format=lambda x: f"{x:+.4f}"))
    print("\nD6 universe vs purchased 90-day returns\n", d6(scores, targets, fills, lots).to_string(float_format=fmt))
    print("\nD7 stale-rule stocks\n", d7().to_string(float_format=lambda x: f"{x:,.2f}"))
    print("\nD8 fees, turnover, holdings\n", d8(h, fills, lots).to_string(float_format=lambda x: f"{x:,.4f}"))


if __name__ == "__main__":
    main()
