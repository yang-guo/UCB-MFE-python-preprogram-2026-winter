"""Final test (Apr 1, 2016 - Sep 30, 2026) with the configuration frozen in development.

Run:  python run_final.py          (about 3-5 minutes on a laptop)

Outputs (``results/``, small enough to commit):
  final_summary.csv, cumulative_returns.png, yearly_returns.csv, accounting_checks.csv,
  timing_audit.csv, model_fits.csv, test_prediction_metrics.csv, extreme_lots.csv,
  open_lots_at_end.csv, sensitivity.csv
Large ledgers (every fill, daily history) go to ``outputs/`` (git-ignored).
"""

from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtester import Backtester, TopNStrategy, performance
from data import FULL_DB, NUMERIC_FEATURES, build_panel
from model import add_rank_features, prediction_metrics, walk_forward_predictions

HERE = Path(__file__).parent
RESULTS, OUTPUTS = HERE / "results", HERE / "outputs"
RESULTS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

# ---- Frozen in run_development.py (highest dev Sharpe that also beat the benchmark) ----
CONFIG = dict(model="ridge", target="rank", refit="QS", top_n=25, cohort_fraction=1 / 12,
              name_cap=0.05, sector_cap=0.30, fee_bps=10.0)
TEST_START, TEST_END = "2016-04-01", "2026-09-30"
INITIAL = 1_000_000.0


def timing_audit(full_panel):
    """Rebuild the panel with the database truncated at a cut date: every feature on
    or before the cut must be identical.  If any feature used later data, it would differ."""
    rows = []
    for cut in ["2015-12-31", "2020-06-30"]:
        short, _ = build_panel(FULL_DB, end=cut)
        a = full_panel.loc[full_panel.date <= cut].set_index(["date", "symbol"]).sort_index()
        b = short.set_index(["date", "symbol"]).sort_index()
        # Expected boundary: the cut date itself cannot be a signal in the short file
        # (it has no next session there), so compare the rows both panels contain.
        common = a.index.intersection(b.index)
        only_full = a.index.difference(b.index)
        cols = NUMERIC_FEATURES + ["tradable"]
        a, b = a.loc[common, cols].astype(float), b.loc[common, cols].astype(float)
        rows.append(dict(cut=cut, common_rows=len(common), rows_only_in_full=len(only_full),
                         dates_only_in_full=", ".join(sorted({str(d.date()) for d in only_full.get_level_values(0)})),
                         rows_only_in_truncated=len(b.index.difference(a.index)),
                         max_abs_feature_diff=float(np.nanmax((a - b).abs().to_numpy())),
                         identical_missing_pattern=bool((a.isna() == b.isna()).all().all())))
    return pd.DataFrame(rows)


def accounting_checks(res, market, start, end):
    f, h, cal = res.fills, res.history, market.calendar
    buys, sells = f[f.side == "BUY"], f[f.side == "SELL"]
    px = market.adj_close
    # Entry fill used that day's close; decision date was the previous session.
    prev_ok = (buys.signal_date.map(cal.get_loc) + 1 == buys.date.map(cal.get_loc)).all()
    buy_px_ok = np.allclose(buys.price.to_numpy(), [px.at[d, s] for d, s in zip(buys.date, buys.symbol)])
    # A sale happens at the first *quoted* session on/after maturity.
    def first_quote(sym, due):
        col = px[sym].loc[due:]
        return col.first_valid_index()
    exit_ok = all(first_quote(s, d) == x for s, d, x in zip(sells.symbol, sells.due_date, sells.date))
    final_marks = {l.symbol: px[l.symbol].loc[:end].dropna().iloc[-1] for l in res.account.lots.values()}
    open_value = sum(l.units * final_marks[l.symbol] for l in res.account.lots.values())
    unreal = sum(l.units * (final_marks[l.symbol] - l.entry_price) - l.entry_fee for l in res.account.lots.values())
    checks = {
        "ledger: initial cash + sum(cash changes) == final cash":
            np.isclose(INITIAL + f.cash_change.sum(), res.account.cash),
        "NAV(end) == cash + open lots x last observed close":
            np.isclose(h.nav.iloc[-1], res.account.cash + open_value),
        "NAV(end) - initial == realized PnL + unrealized PnL":
            np.isclose(h.nav.iloc[-1] - INITIAL, sells.pnl.sum() + unreal),
        "cash never negative": bool((h.cash >= -1e-6).all()),
        "every buy decided at the previous session's close": bool(prev_ok),
        "every buy filled at that day's adjusted close": bool(buy_px_ok),
        "every sale at first quoted close on/after entry + 90 days": bool(exit_ok),
        "min days held (sold lots)": int(sells.days_held.min()),
        "max days held (sold lots)": int(sells.days_held.max()),
        "no fill after test end": bool(f.date.max() <= pd.Timestamp(end)),
        "no buy before test start": bool(buys.date.min() >= pd.Timestamp(start)),
        "buys / sells / open lots": f"{len(buys)} / {len(sells)} / {len(res.account.lots)}",
        "buy orders cancelled (no quote at fill)": int((res.events.event == "buy_cancelled_no_quote").sum()),
        "lots never exited because quotes stopped (open, stale at end)":
            int(sum(px[l.symbol].loc[l.due_date:end].isna().all() for l in res.account.lots.values()
                    if l.due_date <= pd.Timestamp(end))),
        "total fees paid ($)": round(f.fee.sum(), 2),
        "average cash fraction of NAV": round((h.cash / h.nav).mean(), 4),
    }
    return pd.Series(checks, name="value")


def concentration(res, market):
    """Largest name and sector weights at each month end (incl. lots, marked)."""
    px = market.adj_close.ffill()
    f = res.fills
    rows = []
    for d in res.history.resample("ME").last().index:
        d = res.history.index[res.history.index <= d][-1]
        live = f[(f.side == "BUY") & (f.date <= d)]
        sold = set(f[(f.side == "SELL") & (f.date <= d)].lot_id)
        live = live[~live.lot_id.isin(sold)]
        if live.empty:
            continue
        v = live.units * px.loc[d, live.symbol].to_numpy()
        by_name = v.groupby(live.symbol.to_numpy()).sum()
        by_sector = v.groupby(market.sectors.reindex(live.symbol).to_numpy()).sum()
        nav = res.history.nav.loc[d]
        rows.append(dict(date=d, max_name=by_name.max() / nav, max_sector=by_sector.max() / nav,
                         names=len(by_name)))
    return pd.DataFrame(rows)


def held_price_jumps(res, market, threshold=0.40):
    """Lots held across a one-day adjusted-price move beyond +/-40% that did not
    revert within 5 quoted sessions: the signature of a split the vendor failed to
    adjust (e.g. KLAC -89% on 2026-06-08).  Reported, not silently corrected."""
    px = market.adj_close
    lots = res.fills[res.fills.side == "BUY"][["lot_id", "symbol", "date"]].rename(columns={"date": "entry"})
    exits = res.fills[res.fills.side == "SELL"].set_index("lot_id").date
    lots["exit"] = lots.lot_id.map(exits).fillna(pd.Timestamp(TEST_END))
    pnl = res.fills.groupby("lot_id").pnl.sum()
    rows = []
    for sym, g in lots.groupby("symbol"):
        q = px[sym].dropna()
        r = q.pct_change()
        jumps = r[r.abs() > threshold]
        for day, move in jumps.items():
            before = q.loc[:day].iloc[-2]
            after = q.loc[day:].iloc[1:6]
            if (np.abs(np.log(after / before)) < np.log(1.25)).any():
                continue                       # returned to the old level: one bad tick
            hit = g[(g.entry < day) & (g.exit >= day)]
            for lot in hit.itertuples():
                # Ratio close to 1/k for a common split factor k -> likely unadjusted split.
                k = min((2, 3, 4, 5, 10, 20), key=lambda k: abs(np.log((1 + move) * k)))
                likely_split = move < 0 and abs(np.log((1 + move) * k)) < 0.12
                rows.append(dict(symbol=sym, lot_id=lot.lot_id, entry=lot.entry, exit=lot.exit,
                                 jump_date=day, one_day_move=move, likely_split=likely_split,
                                 split_factor=k if likely_split else np.nan, lot_pnl=pnl.get(lot.lot_id, np.nan)))
    return pd.DataFrame(rows, columns=["symbol", "lot_id", "entry", "exit", "jump_date", "one_day_move",
                                       "likely_split", "split_factor", "lot_pnl"])


def run_backtest(pred, market, fee_bps=CONFIG["fee_bps"], top_n=CONFIG["top_n"], settle=None):
    strat = TopNStrategy(pred, market.sectors, top_n=top_n, cohort_fraction=CONFIG["cohort_fraction"],
                         name_cap=CONFIG["name_cap"], sector_cap=CONFIG["sector_cap"])
    return Backtester(market.adj_close, strat, INITIAL, fee_bps, settle_stale_after=settle).run(TEST_START, TEST_END)


def plot(nav, bench, path):
    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1.2], hspace=0.08))
    for a in (ax, ax2):
        a.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.spines[["left", "bottom"]].set_color("#c9c8c2")
        a.grid(axis="y", color="#e6e5e0", linewidth=0.8)
        a.tick_params(colors=muted)
    s, b = nav / INITIAL - 1, bench / INITIAL - 1
    ax.plot(s.index, s * 100, color=blue, lw=2, label="Strategy (Ridge top-25, weekly cohorts)")
    ax.plot(b.index, b * 100, color=orange, lw=2, label="^SP500TR buy-and-hold")
    for series, color in ((s, blue), (b, orange)):
        ax.annotate(f"{series.iloc[-1]:+.0%}", (series.index[-1], series.iloc[-1] * 100), xytext=(6, 0),
                    textcoords="offset points", va="center", color=ink, fontsize=10)
    ax.set_ylabel("Cumulative return (%)", color=muted)
    ax.set_title("Cumulative return, $1M invested at the Apr 1, 2016 close (through Sep 30, 2026)",
                 loc="left", color=ink, fontsize=12)
    ax.legend(frameon=False, loc="upper left")
    ax2.plot(s.index, (nav / nav.cummax() - 1) * 100, color=blue, lw=1.5)
    ax2.plot(b.index, (bench / bench.cummax() - 1) * 100, color=orange, lw=1.5)
    ax2.set_ylabel("Drawdown (%)", color=muted)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)


def main():
    t0 = time.time()
    panel, market = build_panel(FULL_DB, end=TEST_END)
    print(f"panel {panel.shape} ({time.time() - t0:.0f}s); quarantined: {market.quarantine}")
    audit = timing_audit(panel)
    audit.to_csv(RESULTS / "timing_audit.csv", index=False)
    print(audit.to_string())

    ranked = add_rank_features(panel)
    pred, fits = walk_forward_predictions(ranked, TEST_START, TEST_END, kind=CONFIG["model"],
                                          target=CONFIG["target"], refit=CONFIG["refit"], verbose=False)
    fits.to_csv(RESULTS / "model_fits.csv", index=False)
    assert (fits.max_label_end < fits.fit_date).all()
    print(f"{len(fits)} quarterly fits; predictions for {pred.date.nunique()} weekly signals")

    pm = prediction_metrics(pred)
    for year, g in pred.groupby(pred.date.dt.year):
        pm[f"IC_{year}"] = prediction_metrics(g).mean_rank_IC
    pm.to_frame("ridge_rank").to_csv(RESULTS / "test_prediction_metrics.csv", float_format="%.4f")

    res = run_backtest(pred, market)
    nav = res.history.nav
    bench = INITIAL * market.benchmark.loc[nav.index] / market.benchmark.loc[nav.index[0]]
    summary = pd.DataFrame({"Strategy": performance(nav, INITIAL), "^SP500TR": performance(bench, INITIAL)})
    summary["Difference"] = summary.Strategy - summary["^SP500TR"]
    beta = np.cov(nav.pct_change().dropna(), bench.pct_change().dropna())
    summary.loc["beta_to_SP500TR", "Strategy"] = beta[0, 1] / beta[1, 1]
    summary.to_csv(RESULTS / "final_summary.csv", float_format="%.4f")
    print(summary.to_string(float_format=lambda x: f"{x:,.4f}"))

    plot(nav, bench, RESULTS / "cumulative_returns.png")
    yearly = pd.DataFrame({"Strategy": nav.resample("YE").last(), "^SP500TR": bench.resample("YE").last()})
    yearly = pd.concat([pd.DataFrame({"Strategy": [INITIAL], "^SP500TR": [INITIAL]},
                                     index=[nav.index[0] - pd.Timedelta(days=1)]), yearly]).pct_change().iloc[1:]
    yearly.index = yearly.index.year
    yearly["Excess"] = yearly.Strategy - yearly["^SP500TR"]
    yearly.to_csv(RESULTS / "yearly_returns.csv", float_format="%.4f")

    checks = accounting_checks(res, market, TEST_START, TEST_END)
    conc = concentration(res, market)
    checks["max single-name weight at a month end"] = round(conc.max_name.max(), 4)
    checks["max sector weight at a month end"] = round(conc.max_sector.max(), 4)
    checks.to_csv(RESULTS / "accounting_checks.csv")
    print(checks.to_string())

    sells = res.fills[res.fills.side == "SELL"].assign(lot_return=lambda x: x.price / x.entry_price - 1)
    extreme = sells.loc[(sells.lot_return > 1.5) | (sells.lot_return < -0.6),
                        ["symbol", "entry_date", "date", "entry_price", "price", "lot_return", "pnl"]]
    extreme.sort_values("pnl").to_csv(RESULTS / "extreme_lots.csv", index=False, float_format="%.4f")
    jumps = held_price_jumps(res, market)
    jumps.to_csv(RESULTS / "held_price_jumps.csv", index=False, float_format="%.4f")
    print(jumps.groupby(["symbol", "jump_date"]).agg(lots=("lot_id", "size"), move=("one_day_move", "first"),
                                                     split=("likely_split", "first"),
                                                     realized_pnl=("lot_pnl", "sum")).to_string())
    ol = res.open_lots
    ol.to_csv(RESULTS / "open_lots_at_end.csv", index=False, float_format="%.4f")
    stale = ol[ol.last_quote_date < pd.Timestamp(TEST_END)] if len(ol) else ol
    print(f"open lots at end {len(ol)} (value ${ol.value.sum():,.0f}); stale (no quote on last day) {len(stale)}")

    # Sensitivity runs (NOT used for any choice): costs, and the no-model momentum baseline.
    sens = {"chosen (10 bps)": summary.Strategy}
    for bps in (0.0, 25.0):
        sens[f"fees {bps:.0f} bps"] = performance(run_backtest(pred, market, fee_bps=bps).history.nav, INITIAL)
    stale_value = stale.value.sum() if len(stale) else 0.0
    sens["assumed: stuck lots settled at last close"] = performance(
        run_backtest(pred, market, settle=10).history.nav, INITIAL)
    split_pnl = jumps.loc[jumps.likely_split].drop_duplicates("lot_id").lot_pnl.sum()
    sens["approx: PnL of split-hit lots removed"] = performance(
        nav - np.where(nav.index == nav.index[-1], split_pnl, 0), INITIAL)
    sens["stale open lots written to $0"] = performance(nav - np.where(nav.index == nav.index[-1], stale_value, 0), INITIAL)
    mom = ranked.loc[ranked.date >= TEST_START, ["date", "symbol", "sector"]].assign(
        score=ranked.loc[ranked.date >= TEST_START, "mom_252_21"].fillna(0).to_numpy())
    sens["baseline: 12-1 momentum, same rules"] = performance(run_backtest(mom, market).history.nav, INITIAL)
    sens["baseline: 12-1 momentum, stuck lots settled"] = performance(
        run_backtest(mom, market, settle=10).history.nav, INITIAL)
    # Random picks from the same tradable universe, same rules: separates what the
    # model adds from what an equal-weight mid/small-cap book does by itself.
    rand = [performance(run_backtest(mom.assign(score=np.random.default_rng(seed).random(len(mom))),
                                     market).history.nav, INITIAL) for seed in range(5)]
    sens["baseline: random 25 names (mean of 5 seeds)"] = pd.DataFrame(rand).mean()
    sens["^SP500TR"] = summary["^SP500TR"]
    sens = pd.DataFrame(sens).T
    sens.to_csv(RESULTS / "sensitivity.csv", float_format="%.4f")
    print(sens[["final_value", "CAGR", "max_drawdown", "Sharpe"]].to_string(float_format=lambda x: f"{x:,.4f}"))

    res.fills.to_csv(OUTPUTS / "fills.csv.gz", index=False)
    res.history.assign(benchmark_nav=bench).to_csv(OUTPUTS / "daily_history.csv.gz")
    pred.to_csv(OUTPUTS / "test_predictions.csv.gz", index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
