"""HW4: freeze a 90-day long-only rule in 2010–2015, then run Apr 2016–Sep 2026.

    python test_accounting.py
    python hw4.py --develop     # 2013–2015 comparison; writes results/freeze.json
    python hw4.py --final       # frozen rule on the homework test window
    python hw4.py --smoke       # 30-stock class database; writes freeze_smoke.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge

COURSE_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file())
MODELING = COURSE_ROOT / "Lectures" / "modeling"
if str(MODELING) not in sys.path:
    sys.path.insert(0, str(MODELING))

from course_data import date_folds, load_data, make_pipeline, prediction_metrics  # noqa: E402

from backtester import Backtester, MonthlyTopNAllocator, performance  # noqa: E402
from features import (  # noqa: E402
    FACTOR_SPECS,
    MODEL_FEATURES,
    RANK_CAP,
    batched_month_end_panel,
    entry_screen,
    factor_scores,
    in_fold_keep,
)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CLASS_DB = MODELING / "data" / "equities_class.duckdb"
FULL_DB = MODELING / "data" / "equities_top2000.duckdb"
TOP_N = 15
COHORT = 1.0 / 3.0
FEE_BPS = 5


def freeze_path(tag: str) -> Path:
    return RESULTS / ("freeze_smoke.json" if tag.startswith("smoke") else "freeze.json")


def make_ridge():
    return make_pipeline(Ridge(alpha=100), numeric_features=MODEL_FEATURES, include_sector=False)


def choose_rank_cap(panel: pd.DataFrame) -> int | None:
    screened = panel.loc[entry_screen(panel, RANK_CAP)]
    if screened.symbol.nunique() < 40:
        return None
    return RANK_CAP


def labeled_development(panel: pd.DataFrame, development_end: str, rank_cap, label_end_before=None):
    mask = (
        entry_screen(panel, rank_cap)
        & panel.date.ge("2010-01-01")
        & panel.date.le(development_end)
        & panel.target.notna()
        & panel.rank_target.notna()
    )
    if label_end_before is not None:
        mask = mask & panel.label_end.lt(label_end_before)
    return panel.loc[mask].reset_index(drop=True)


def _next_session_year(date_series: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    pos = calendar.get_indexer(pd.DatetimeIndex(date_series))
    years = np.full(len(date_series), np.nan)
    ok = (pos >= 0) & (pos + 1 < len(calendar))
    years[ok] = calendar[pos[ok] + 1].year
    return pd.Series(years, index=date_series.index)


def _score_frame(rows: pd.DataFrame, values) -> pd.DataFrame:
    out = rows[["date", "symbol"]].copy()
    out["score"] = np.asarray(values, dtype=float)
    return out.loc[np.isfinite(out.score)].reset_index(drop=True)


def walk_forward_ridge(panel, labeled, validation_years, rank_cap, calendar):
    folds = date_folds(labeled, validation_years=validation_years)
    ics, parts = [], []
    screened = panel.loc[entry_screen(panel, rank_cap)].copy()
    exec_year = _next_session_year(screened.date, calendar)
    for year, (_, va) in zip(validation_years, folds):
        to_score = screened.loc[exec_year.eq(year)]
        if to_score.empty:
            continue
        # Fit before the first scored signal. That date can be the prior December,
        # so the cutoff is earlier than the first validation row inside `year`.
        cutoff = pd.Timestamp(to_score.date.min())
        train = labeled.loc[labeled.date.lt(cutoff) & labeled.label_end.lt(cutoff)]
        keep = in_fold_keep(train.target)
        if keep.sum() < 50:
            keep = pd.Series(True, index=train.index)
        model = clone(make_ridge()).fit(train.loc[keep, MODEL_FEATURES], train.loc[keep, "rank_target"])
        assert train.loc[keep, "label_end"].max() < cutoff
        valid = labeled.iloc[va]
        ics.append(float(prediction_metrics(valid, model.predict(valid[MODEL_FEATURES]))["mean_daily_rank_IC"]))
        chunk = _score_frame(to_score, model.predict(to_score[MODEL_FEATURES]))
        chunk["fit_date"] = cutoff
        parts.append(chunk)
    return float(np.nanmean(ics)), pd.concat(parts, ignore_index=True), ics


def walk_factor(panel, labeled, spec, validation_years, rank_cap, calendar):
    folds = date_folds(labeled, validation_years=validation_years)
    ics, parts = [], []
    screened = panel.loc[entry_screen(panel, rank_cap)].copy()
    exec_year = _next_session_year(screened.date, calendar)
    for year, (_, va) in zip(validation_years, folds):
        valid = labeled.iloc[va]
        pred = factor_scores(valid, spec)
        ok = pred.notna() & valid.target.notna()
        ics.append(
            float("nan") if ok.sum() < 5
            else float(prediction_metrics(valid.loc[ok], pred.loc[ok])["mean_daily_rank_IC"])
        )
        to_score = screened.loc[exec_year.eq(year)]
        chunk = _score_frame(to_score, factor_scores(to_score, spec))
        chunk["fit_date"] = valid.date.min()
        parts.append(chunk)
    return float(np.nanmean(ics)), pd.concat(parts, ignore_index=True), ics


def paper_backtest(scores, prices, benchmark, start, end, top_n=TOP_N):
    allocator = MonthlyTopNAllocator(scores, top_n=top_n, cohort_fraction=COHORT, holding_days=90)
    traded = set(
        scores.sort_values(["date", "score", "symbol"], ascending=[True, False, True])
        .groupby("date").head(top_n).symbol
    )
    px = prices.loc[prices.symbol.isin(traded), ["date", "symbol", "adj_close"]]
    result = Backtester(px, benchmark[["date", "adj_close"]], allocator,
                        initial_cash=1_000_000, fee_bps=FEE_BPS).run(start, end)
    stats = performance(result.history.iloc[1:].nav)
    bench = performance(result.history.iloc[1:].benchmark_nav)
    return result, stats, bench


def freeze_rule(rows: pd.DataFrame, top_n: int) -> dict:
    """Positive-IC candidates, then highest development CAGR (Sharpe, IC as ties)."""
    work = rows.copy()
    contenders = work.loc[work.ic.gt(0)] if work.ic.gt(0).any() else work
    pick = contenders.sort_values(["cagr", "sharpe", "ic"], ascending=False).iloc[0]
    return {
        "candidate": str(pick.candidate),
        "flip_score": False,
        "top_n": int(top_n),
        "rank_cap": None if pd.isna(pick.rank_cap) else int(pick.rank_cap),
        "development_ic": float(pick.ic),
        "development_cagr": float(pick.cagr),
        "development_sharpe": float(pick.sharpe),
    }


def save_chart(history: pd.DataFrame, path: Path, title: str) -> None:
    test = history.iloc[1:]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(test.index, test.nav / test.nav.iloc[0], label="Strategy")
    ax.plot(test.index, test.benchmark_nav / test.benchmark_nav.iloc[0], label="S&P 500 total return")
    ax.set_ylabel("Cumulative growth of $1")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def compare_candidates(panel, benchmark, prices, development_end, validation_years, rank_cap, tag,
                       label_end_before=None):
    calendar = pd.DatetimeIndex(benchmark.date)
    first_valid = calendar[calendar >= pd.Timestamp(f"{validation_years[0]}-01-01")][0]
    last_valid = calendar[calendar <= pd.Timestamp(development_end)][-1]
    labeled = labeled_development(panel, development_end, rank_cap, label_end_before=label_end_before)
    print(f"Development labeled rows: {len(labeled):,}; rank_cap={rank_cap}")
    rows, score_map = [], {}

    ic, scores, yearly = walk_forward_ridge(panel, labeled, validation_years, rank_cap, calendar)
    _, stats, bench = paper_backtest(scores, prices, benchmark, first_valid, last_valid)
    score_map["ridge_rank"] = scores
    rows.append(_row("ridge_rank", ic, stats, bench, rank_cap, yearly))
    print(f"  ridge_rank: IC={ic:.4f}  CAGR={stats.CAGR:.2%}  Sharpe={stats.Sharpe:.2f}")

    for name, spec in FACTOR_SPECS.items():
        ic, scores, yearly = walk_factor(panel, labeled, spec, validation_years, rank_cap, calendar)
        _, stats, bench = paper_backtest(scores, prices, benchmark, first_valid, last_valid)
        score_map[name] = scores
        rows.append(_row(name, ic, stats, bench, rank_cap, yearly))
        print(f"  {name}: IC={ic:.4f}  CAGR={stats.CAGR:.2%}  Sharpe={stats.Sharpe:.2f}")

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / f"development_{tag}.csv", index=False)
    freeze = freeze_rule(table, TOP_N)
    freeze_path(tag).write_text(json.dumps(freeze, indent=2))
    print("Frozen:", json.dumps(freeze))
    return freeze, score_map, table


def _row(name, ic, stats, bench, rank_cap, yearly):
    return {
        "candidate": name, "ic": ic, "cagr": float(stats.CAGR), "sharpe": float(stats.Sharpe),
        "final_value": float(stats.final_value), "bench_cagr": float(bench.CAGR),
        "rank_cap": rank_cap, "yearly_ic": ",".join(f"{x:.4f}" for x in yearly),
    }


def fit_ridge(panel, signal_end, label_end_before, rank_cap):
    labeled = labeled_development(panel, signal_end, rank_cap, label_end_before=label_end_before)
    keep = in_fold_keep(labeled.target)
    model = clone(make_ridge())
    model.fit(labeled.loc[keep, MODEL_FEATURES], labeled.loc[keep, "rank_target"])
    assert labeled.label_end.max() < pd.Timestamp(label_end_before)
    return model


def score_test(panel, freeze, first_signal, test_end, rank_cap, calendar):
    rows = panel.loc[entry_screen(panel, rank_cap) & panel.date.ge(first_signal) & panel.date.le(test_end)].copy()
    candidate = freeze["candidate"]
    if candidate in FACTOR_SPECS:
        scores = _score_frame(rows, factor_scores(rows, FACTOR_SPECS[candidate]))
        scores["fit_date"] = first_signal
        return scores
    exec_year = _next_session_year(rows.date, calendar)
    parts = []
    for year in sorted(int(y) for y in np.unique(exec_year.dropna())):
        year_days = calendar[calendar.year == year]
        first_fill = year_days[year_days.ge(pd.Timestamp(first_signal) + pd.Timedelta(days=1))]
        if first_fill.empty:
            continue
        label_cutoff = calendar[calendar.get_loc(first_fill[0]) - 1]
        model = fit_ridge(panel, test_end, label_cutoff, rank_cap)
        chunk_rows = rows.loc[exec_year.eq(year)]
        if chunk_rows.empty:
            continue
        chunk = _score_frame(chunk_rows, model.predict(chunk_rows[MODEL_FEATURES]))
        chunk["fit_date"] = label_cutoff
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True)


def write_run(result, tag, freeze, extra: dict):
    summary = pd.DataFrame({
        "strategy": performance(result.history.iloc[1:].nav),
        "S&P 500 total return": performance(result.history.iloc[1:].benchmark_nav),
    })
    print(summary.to_string())
    print(result.pnl.to_string())
    diag = result.diagnostics
    summary.to_csv(RESULTS / f"performance_{tag}.csv")
    result.pnl.to_csv(RESULTS / f"pnl_{tag}.csv")
    if not result.fills.empty:
        result.fills.head(20).to_csv(RESULTS / f"fills_head_{tag}.csv", index=False)
    save_chart(result.history, RESULTS / f"cumulative_returns_{tag}.png", f"Cumulative returns: {tag}")
    result.history[["cash", "holdings_value", "nav", "benchmark_nav", "open_lots", "fees"]].to_csv(
        RESULTS / f"daily_nav_{tag}.csv"
    )
    meta = {
        **freeze,
        **extra,
        "strategy_final": float(summary.loc["final_value", "strategy"]),
        "benchmark_final": float(summary.loc["final_value", "S&P 500 total return"]),
        "beat_benchmark_dollars": float(summary.loc["final_value", "strategy"]
                                        - summary.loc["final_value", "S&P 500 total return"]),
        "n_fills": int(len(result.fills)),
        "open_lots_end": int(len(result.open_lots)),
        "unresolved_held_gaps": int((diag.event == "unresolved_held_gap").sum()) if len(diag) else 0,
        "stale_marks": int((diag.event == "stale_mark").sum()) if len(diag) else 0,
    }
    (RESULTS / f"run_{tag}.json").write_text(json.dumps(meta, indent=2, default=str))
    return summary


def load_prices(database, end, symbols=None):
    prices, _, _, _ = load_data(database, end=end, symbols=symbols)
    return prices[["date", "symbol", "adj_close"]]


def run_develop(database, tag, development_end, validation_years, panel_end, label_end_before=None):
    RESULTS.mkdir(exist_ok=True)
    print(f"Development panel {database.name} through {panel_end}")
    panel, benchmark = batched_month_end_panel(database, end=panel_end)
    rank_cap = choose_rank_cap(panel.loc[panel.date.le(development_end)])
    prices = load_prices(database, panel_end)
    freeze, score_map, _ = compare_candidates(
        panel, benchmark, prices, development_end, validation_years, rank_cap, tag,
        label_end_before=label_end_before,
    )
    picked = score_map[freeze["candidate"]].copy()
    calendar = pd.DatetimeIndex(benchmark.date)
    start = calendar[calendar >= pd.Timestamp(f"{validation_years[0]}-01-01")][0]
    end = calendar[calendar <= pd.Timestamp(development_end)][-1]
    result, _, _ = paper_backtest(picked, prices, benchmark, start, end, top_n=int(freeze["top_n"]))
    save_chart(result.history, RESULTS / f"development_returns_{tag}.png",
               f"Development {freeze['candidate']} ({start.date()}–{end.date()})")
    return freeze


def run_final(database, tag, test_start, test_end, development_end, validation_years):
    RESULTS.mkdir(exist_ok=True)
    print(f"Final panel {database.name} through {test_end}")
    panel, benchmark = batched_month_end_panel(database, end=test_end)
    calendar = pd.DatetimeIndex(benchmark.date)
    first_test = calendar[calendar >= pd.Timestamp(test_start)][0]
    first_signal = calendar[calendar.get_loc(first_test) - 1]
    path = freeze_path("hw4")
    rank_cap = choose_rank_cap(panel.loc[panel.date.le(development_end)])
    if path.is_file():
        freeze = json.loads(path.read_text())
        print("Loaded freeze", freeze)
    else:
        prices_dev = load_prices(database, "2016-04-15")
        freeze, _, _ = compare_candidates(
            panel, benchmark, prices_dev, development_end, validation_years, rank_cap, "inline",
            label_end_before=str(pd.Timestamp(test_start).date()),
        )
    rank_cap = None if freeze["rank_cap"] is None else int(freeze["rank_cap"])
    scores = score_test(panel, freeze, first_signal, test_end, rank_cap, calendar)
    print(f"Scored test rows: {len(scores):,}")
    top_n = int(freeze.get("top_n", TOP_N))
    traded = set(
        scores.sort_values(["date", "score", "symbol"], ascending=[True, False, True])
        .groupby("date").head(top_n).symbol
    )
    prices = load_prices(database, test_end, symbols=sorted(traded))
    result, _, _ = paper_backtest(scores, prices, benchmark, first_test, pd.Timestamp(test_end), top_n=top_n)
    write_run(result, tag, freeze, {
        "test_start": str(first_test.date()),
        "test_end": test_end,
        "first_signal": str(first_signal.date()),
        "scored_rows": int(len(scores)),
        "n_traded_symbols": len(traded),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--develop", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        run_develop(CLASS_DB, "smoke", "2014-12-31", (2012, 2013, 2014), "2015-12-31")
        return
    if args.final:
        run_final(FULL_DB, "hw4", "2016-04-01", "2026-09-30", "2015-12-31", (2013, 2014, 2015))
        return
    run_develop(FULL_DB, "dev", "2015-12-31", (2013, 2014, 2015), "2016-04-15",
                label_end_before="2016-04-01")


if __name__ == "__main__":
    main()
