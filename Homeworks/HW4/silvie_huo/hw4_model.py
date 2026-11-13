"""HW4 stock-return model: features, 90-day labels, validation, walk-forward scores.

Reuses course_data.py unchanged (load_data, price_features, financial_events,
join_financials, apply_membership, make_labels, date_folds, make_pipeline,
prediction_metrics, FEATURES).
"""
from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from course_data import (FEATURES, NUMERIC_FEATURES, apply_membership, date_folds,
                         financial_events, join_financials, load_data, load_membership,
                         make_labels, make_pipeline, prediction_metrics, price_features)

HERE = Path(__file__).parent
DB = HERE / "data/equities_top2000.duckdb"
OUT = HERE / "outputs/hw4"
DEV_START, DEV_END = "2010-01-01", "2015-12-31"
TEST_START, TEST_END = "2016-04-01", "2026-09-30"
HOLDING_DAYS = 90
REBALANCE_MONTHS = (3, 6, 9, 12)
KEEP = (["date", "symbol", "sector", "universe_rank", "eligible", "entry_date", "entry_price",
         "scheduled_exit", "label_end", "exit_price", "target"] + NUMERIC_FEATURES)

CANDIDATES = {
    **{f"ridge_a{a}": dict(kind="ridge", alpha=a, winsor=False) for a in (10, 100, 1000, 10000)},
    **{f"ridge_a{a}_w": dict(kind="ridge", alpha=a, winsor=True) for a in (10, 100, 1000, 10000)},
    "hgb_w": dict(kind="hgb", winsor=True),
}


def load_calendar(database=DB, end=TEST_END):
    with duckdb.connect(str(database), read_only=True) as con:
        d = con.execute("SELECT date FROM benchmark WHERE date<=? ORDER BY date", [end]).fetchdf().date
    return pd.DatetimeIndex(pd.to_datetime(d)).astype("datetime64[ns]")


def decision_dates(calendar, start, end, months=REBALANCE_MONTHS):
    """First market session of each rebalance month (entry) and the session before it (signal)."""
    cal = pd.DatetimeIndex(calendar)
    first = pd.Series(cal, index=cal).groupby([cal.year, cal.month]).min()
    first = first[first.index.get_level_values(1).isin(months)]
    entries = pd.DatetimeIndex(first.values)
    entries = entries[(entries >= pd.Timestamp(start)) & (entries <= pd.Timestamp(end))]
    pos = cal.get_indexer(entries)
    ok = pos > 0
    return pd.DataFrame({"signal_date": cal[pos[ok] - 1], "entry_date": entries[ok]})


def build_panel(database=DB, end=TEST_END, batch_size=250, stride=5, signals=()):
    """Same steps as course_data.prepare_panel, but batched by symbol and thinned to every
    `stride`-th session plus the decision signals. Per-symbol functions make batching safe."""
    cal = load_calendar(database, end)
    sample = cal[::stride].union(pd.DatetimeIndex(signals))
    membership = load_membership(database, end=end)
    with duckdb.connect(str(database), read_only=True) as con:
        symbols = [r[0] for r in con.execute("SELECT symbol FROM universe ORDER BY symbol").fetchall()]
    parts = []
    for i in range(0, len(symbols), batch_size):
        prices, fundamentals, universe, benchmark = load_data(database, end=end,
                                                              symbols=symbols[i:i + batch_size])
        if prices.empty:
            continue
        panel = price_features(prices, benchmark.date)
        panel = panel[panel.date.isin(sample)]
        panel = join_financials(panel, financial_events(fundamentals), universe)
        panel = apply_membership(panel, membership, benchmark.date)
        panel = panel[panel.eligible]
        if panel.empty:
            continue
        panel = make_labels(panel, prices, benchmark.date, holding_days=HOLDING_DAYS)
        parts.append(panel[KEEP].copy())
        print(f"batch {i // batch_size + 1}: {len(panel):,} rows")
    return pd.concat(parts, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)


def fit_spec(spec, train):
    """Fit preprocessing + model on `train` only (clipping, imputation, scaling are in the pipeline)."""
    y = train.target
    if spec["winsor"]:  # training-target winsorization, bounds learned from the training rows
        y = y.clip(*y.quantile([.01, .99]))
    model = (Ridge(alpha=spec["alpha"]) if spec["kind"] == "ridge"
             else HistGradientBoostingRegressor(max_depth=3, learning_rate=.05, max_iter=200,
                                                random_state=0))
    return make_pipeline(model).fit(train[FEATURES], y)


def validate(dev, years=(2013, 2014, 2015)):
    """Expanding-window folds: each fold trains on labels that ended before the year's first signal."""
    folds = date_folds(dev, validation_years=years, train_start=DEV_START)
    rows = []
    for name, spec in CANDIDATES.items():
        for year, (tr, va) in zip(years, folds):
            model = fit_spec(spec, dev.iloc[tr])
            m = prediction_metrics(dev.iloc[va], model.predict(dev.iloc[va][FEATURES]))
            rows.append({"candidate": name, "valid_year": year, **m.to_dict()})
    folds_df = pd.DataFrame(rows)
    summary = folds_df.groupby("candidate")[["RMSE", "MAE", "R2", "mean_daily_rank_IC"]].mean()
    return folds_df, summary.sort_values("mean_daily_rank_IC", ascending=False)


def walk_forward_scores(panel, signal_dates, spec, min_rows=20_000):
    """Refit at each decision signal using only rows whose labels ended strictly before it
    (adapted from the fit loop in 2_research_workflow.ipynb)."""
    base = panel.loc[panel.date.ge(DEV_START) & panel.target.notna()]
    parts, audit = [], []
    for d in pd.DatetimeIndex(signal_dates):
        train = base.loc[base.date.lt(d) & base.label_end.lt(d)]
        to_score = panel.loc[panel.date.eq(d)]
        if len(train) < min_rows or to_score.empty:
            continue
        assert train.label_end.max() < d, "training label not finished before fit date"
        model = fit_spec(spec, train)
        out = to_score[["date", "symbol"]].copy()
        out["score"] = model.predict(to_score[FEATURES])
        out["fit_date"] = d
        parts.append(out)
        audit.append({"fit_date": d, "train_rows": len(train), "last_train_signal": train.date.max(),
                      "last_label_end": train.label_end.max(), "scored_rows": len(out)})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(audit)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cal = load_calendar()
    dec_dev = decision_dates(cal, DEV_START, DEV_END)
    dec_test = decision_dates(cal, TEST_START, TEST_END)
    signals = pd.DatetimeIndex(dec_dev.signal_date).union(pd.DatetimeIndex(dec_test.signal_date))

    panel = build_panel(signals=signals)
    panel.to_pickle(OUT / "panel.pkl")

    dev = panel[panel.date.between(DEV_START, DEV_END)].reset_index(drop=True)
    folds_df, summary = validate(dev)
    folds_df.to_csv(OUT / "validation_by_fold.csv", index=False)
    summary.to_csv(OUT / "validation_summary.csv")
    print(summary)
    best = summary.mean_daily_rank_IC.idxmax()
    spec = CANDIDATES[best]
    (OUT / "chosen_model.json").write_text(json.dumps({"name": best, **spec}, indent=2))

    dev_scores, dev_audit = walk_forward_scores(panel, dec_dev.signal_date, spec)
    test_scores, test_audit = walk_forward_scores(panel, dec_test.signal_date, spec)
    dev_scores.to_csv(OUT / "dev_scores.csv", index=False)
    test_scores.to_csv(OUT / "test_scores.csv", index=False)
    pd.concat([dev_audit.assign(period="dev"), test_audit.assign(period="test")]).to_csv(
        OUT / "fit_audit.csv", index=False)
    print(test_audit)


if __name__ == "__main__":
    main()