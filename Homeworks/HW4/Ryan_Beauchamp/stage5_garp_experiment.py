"""HW4 Task C: pre-registered GARP experiment + fee sensitivity (development data only).

Data: development panel and GARP features built from data through 2016-03-31;
signals 2012-2015; backtests 2012-01-03 to 2015-12-31.

Three versions, same annual walk-forward rank-target Ridge (alpha 1000), same folds:
  baseline        = course FEATURES (11 numeric + filing_age_days + sector)
  baseline + GARP = baseline + the 8 GARP-set columns
  GARP only       = the 8 GARP-set columns (no sector)
GARP set = fcf_yield, ebitda_yield, earnings_yield, rev_growth_ttm,
           eps_change_yield, op_margin_ttm, sue_q, garp_composite.

ADOPTION RULE (fixed before running; not to be changed afterwards)
  Adopt a GARP version only if, versus baseline,
    (1) the 2012-2015 mean daily rank IC is higher, AND
    (2) the top-5%-minus-rest gap is higher in at least 3 of the 4 years.
  The gap used for (2) is the WINSORIZED gap (per-date 1/99% clip of the
  realized targets, evaluation only); the raw gap is reported alongside.
  The rule is checked for "baseline + GARP" (the candidate for adoption);
  "GARP only" is reported for information.

Backtest: draft config (top 50, 2% total cap, every 5 sessions, cohort 5/63,
10 bps). Fee sensitivity: baseline draft config at 0, 10 and 25 bps.

Run from the repository root:
    .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/stage5_garp_experiment.py

Credits: make_pipeline/FEATURES from Lectures/modeling/course_data.py (via
walkforward.py); backtester structure modeled on Lectures/modeling/backtest.py.
Written with AI assistance (Claude Code) and reviewed by the author.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import walkforward as wf  # noqa: E402
import stage4_dev_grid as g  # noqa: E402
from garp_features import GARP_SET  # noqa: E402
from backtester import Backtester, ScoreStrategy, accounting_checks, performance  # noqa: E402

OUT = HERE / "outputs"
GARP_PATH = OUT / "garp_dev" / "garp_features.parquet"
DRAFT = dict(top_n=50, per_name_cap=0.02, entry_every=5)
VERSIONS = {
    "baseline": dict(numeric_features=list(wf.cd.NUMERIC_FEATURES), include_sector=True),
    "baseline + GARP": dict(numeric_features=list(wf.cd.NUMERIC_FEATURES) + GARP_SET, include_sector=True),
    "GARP only": dict(numeric_features=list(GARP_SET), include_sector=False),
}
LOG = OUT / "stage5_run_log.csv"


def backtest(scores_map, prices, fee_bps, label):
    s = ScoreStrategy(scores_map, top_n=DRAFT["top_n"], cohort_fraction=DRAFT["entry_every"] / g.SESSIONS_PER_HOLD,
                      per_name_cap=DRAFT["per_name_cap"])
    r = Backtester(prices, s, fee_bps=fee_bps, entry_every=DRAFT["entry_every"],
                   max_stale_sessions=g.MAX_STALE, long_gap_policy="hold").run()
    checks = accounting_checks(r)
    m = performance(r.history.nav)
    row = {"logged_at": datetime.now().isoformat(timespec="seconds"), "run": label, "fee_bps": fee_bps, **DRAFT,
           "final_value": m.final_value, "CAGR": m.CAGR, "max_drawdown": m.max_drawdown, "sharpe": m.sharpe,
           "avg_invested": (r.history.holdings / r.history.nav).mean(), "fees": r.fills.fee.sum(),
           "fills": len(r.fills), "long_gaps": len(r.long_gaps), "jumps": len(r.jumps),
           "checks_ok": all(checks.values())}
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    return row


def main():
    start = time.perf_counter()
    frame, _ = wf.load_model_rows(g.PANEL, g.MEASURES, "2010-01-01", "2015-12-31",
                                  extra_path=GARP_PATH, extra_columns=GARP_SET)
    assert frame.date.max() <= g.BT_END and frame.label_end.max() <= g.DATA_END
    print(f"Model rows: {len(frame):,}; GARP coverage on them: "
          + ", ".join(f"{c} {frame[c].notna().mean():.0%}" for c in GARP_SET))
    fits = wf.fit_dates(frame.date.unique(), g.BT_START, g.BT_END, "annual")

    scores, quality = {}, {}
    for name, spec in VERSIONS.items():
        print(f"\n{name}:")
        scores[name], fit_log = wf.walk_forward_scores(frame, fits, g.BT_END, **spec)
        scores[name].to_parquet(OUT / f"stage5_scores_{name.replace(' + ', '_plus_').replace(' ', '_')}.parquet",
                                index=False)
        quality[name] = g.prediction_quality(frame, scores[name])

    # The refactored code path must reproduce the stage 4 baseline scores exactly.
    cached = pd.read_parquet(OUT / "stage4_scores_annual.parquet")
    both = cached.merge(scores["baseline"], on=["date", "symbol"], suffixes=("_s4", "_s5"))
    assert len(both) == len(cached) == len(scores["baseline"])
    print(f"\nBaseline reproduces stage 4 annual scores: max |diff| {(both.score_s4 - both.score_s5).abs().max():.1e}")

    q = pd.concat(quality, axis=1)
    print("\nPrediction quality per fold (rank IC; top-5% minus rest, per 90 days):")
    print(q.to_string(float_format=lambda x: f"{x:+.4f}"))
    q.to_csv(OUT / "stage5_prediction_quality.csv")

    # Adoption rule, exactly as pre-registered.
    years = [2012, 2013, 2014, 2015]
    base, cand = quality["baseline"], quality["baseline + GARP"]
    ic_better = cand.loc["2012-2015", "rank_IC"] > base.loc["2012-2015", "rank_IC"]
    gap_years = int((cand.loc[years, "top5_gap_wins"] > base.loc[years, "top5_gap_wins"]).sum())
    adopt = bool(ic_better and gap_years >= 3)
    only = quality["GARP only"]
    only_years = int((only.loc[years, "top5_gap_wins"] > base.loc[years, "top5_gap_wins"]).sum())
    print(f"\nAdoption rule, baseline + GARP vs baseline: mean IC {cand.loc['2012-2015', 'rank_IC']:+.4f} vs "
          f"{base.loc['2012-2015', 'rank_IC']:+.4f} ({'higher' if ic_better else 'not higher'}); winsorized gap "
          f"higher in {gap_years}/4 years -> {'ADOPT' if adopt else 'DO NOT ADOPT'}")
    print(f"(info) GARP only: mean IC {only.loc['2012-2015', 'rank_IC']:+.4f}; winsorized gap higher in "
          f"{only_years}/4 years")

    # Backtests on the dev window.
    import duckdb
    with duckdb.connect(str(g.DATABASE), read_only=True) as con:
        bench = con.execute("SELECT date, adj_close FROM benchmark WHERE date BETWEEN ? AND ? ORDER BY date",
                            [g.BT_START.date(), g.BT_END.date()]).fetchdf()
    bench = bench.assign(date=pd.to_datetime(bench.date)).set_index("date").adj_close
    long = pd.read_parquet(g.PANEL, columns=["date", "symbol", "adj_close"],
                           filters=[("date", ">=", g.BT_START), ("date", "<=", g.BT_END)])
    prices = long.pivot(index="date", columns="symbol", values="adj_close").reindex(bench.index)

    rows = [backtest(wf.scores_by_date(scores[name]), prices, 10, name) for name in VERSIONS]
    rows += [backtest(wf.scores_by_date(scores["baseline"]), prices, fee, f"baseline fee {fee} bps") for fee in (0, 25)]
    bt = pd.DataFrame(rows).set_index("run")
    print("\nBacktests 2012-01-03 to 2015-12-31 (draft config; benchmark CAGR 14.92%, Sharpe 1.00, maxDD -12.04%):")
    print(bt[["fee_bps", "final_value", "CAGR", "max_drawdown", "sharpe", "avg_invested", "fees", "long_gaps",
              "jumps", "checks_ok"]].to_string(float_format=lambda x: f"{x:,.4f}"))
    bt.to_csv(OUT / "stage5_backtests.csv")
    pd.Series({"ic_better": ic_better, "gap_years_better": gap_years, "adopt": adopt}).to_csv(
        OUT / "stage5_adoption_rule.csv")
    print(f"\nTotal {time.perf_counter() - start:.0f}s; peak mem {g.peak_gb():.1f} GB")


if __name__ == "__main__":
    main()
