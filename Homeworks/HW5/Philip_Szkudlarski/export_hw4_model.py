"""Export my HW4 final model (run 10) as a fitted scikit-learn Pipeline.

This is an offline step, NOT part of the service or the Docker image. It needs
the HW4 code, the repository's uv environment (duckdb) and the course database:

    # from the repository root
    .venv/bin/python Homeworks/HW5/Philip_Szkudlarski/export_hw4_model.py

It reuses the HW4 functions unchanged (imported from ../../HW4/Philip_Szkudlarski/hw4.py)
to rebuild the four signal sleeves of the frozen final model
"pairs+fcf+revenue+momentum [trailing 126d]", exactly as HW4's main() does:

* pairs / revenue / momentum: fixed parameters chosen on 2010-2012, shifted by
  their waiting period; fcf: the walk-forward parameter choice.
* trailing-126-day weights from finished 90-day outcomes (min 20 signal days).

The model state on the last HW4 signal date (2026-09-30, end of the test data)
is frozen into the artifact. Nothing is re-tuned: no parameter, weight rule or
signal choice differs from HW4. Checks written to artifacts/export_checks.json:

1. pipeline.predict == HW4 combine_weighted score for every candidate that day;
2. yearly mean weights == HW4 outputs/test_dynamic_weights_by_year.csv;
3. every stock HW4 bought in the test had a finite HW4 score on its signal day.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
HW4_DIR = HERE.parents[1] / "HW4" / "Philip_Szkudlarski"
ARTIFACTS = HERE / "artifacts"
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HW4_DIR))

import hw4  # noqa: E402  (HW4 code, imported after the path is set)

from hw4_model import (  # noqa: E402
    FEATURES,
    FlaggedPercentileTransformer,
    SignalScoreRegressor,
)

# HW4 sleeve name -> serving feature name (same order as hw4_model.FEATURES).
SLEEVES = {
    "pairs": "pairs_divergence_z",
    "fcf": "fcf_quality_score",
    "revenue": "revenue_growth_vs_industry",
    "momentum": "momentum_percentile",
}
LOOKBACK, MODE = 126, "prop"  # final14 weighting method "trailing 126d"
MODEL_VERSION = "hw4-run10-trailing126d-2026-09-30-v2"


def build_hw4_state(database):
    """Repeat the parts of hw4.main() that define the final model's inputs."""
    hw4.np.seterr(all="ignore")
    kept, _ = hw4.load_universe(database)
    md = hw4.load_market_data(database, kept)
    cal, cols = md.calendar, md.px.columns
    end = min(cal[-1], pd.Timestamp(hw4.TEST_END))
    elig = hw4.eligibility(md)
    feat = hw4.Features(md)
    fund = hw4.fundamentals_wide(hw4.fundamental_events(md.fundamentals), md)
    labels = hw4.Labels(md, elig)
    pairs = hw4.PairsEngine(md, feat, elig)
    makers = {
        "pairs": lambda p: pairs.signals(**p, eps_ok=fund["eps_ok"]),
        "fcf": lambda p: hw4.generate_fcf_signals(fund, elig, **p),
        "revenue": lambda p: hw4.generate_revenue_momentum_signals(
            fund, elig, md.info.industry, **p
        ),
        "momentum": lambda p: hw4.generate_momentum_signals(feat, elig, **p),
    }
    grids = {
        "pairs": hw4.PAIRS_GRID,
        "fcf": hw4.FCF_GRID,
        "revenue": hw4.REVENUE_GRID,
        "momentum": hw4.MOMENTUM_GRID,
    }
    # Fit rows exactly as in hw4.main().
    valid_fits = [cal.searchsorted(pd.Timestamp(f"{y}-01-01")) for y in range(2013, 2016)]
    first_entry = hw4.first_session_on_or_after(cal, hw4.TEST_START)
    first_signal_row = cal.get_loc(first_entry) - 1
    test_fits = [first_signal_row] + [
        cal.searchsorted(pd.Timestamp(f"{y}-04-01")) - 1
        for y in range(first_entry.year + 1, end.year + 1)
        if pd.Timestamp(f"{y}-04-01") <= end
    ]
    all_fits = valid_fits + test_fits

    sleeves, params, cache = {}, {}, {}
    for name in SLEEVES:
        spec = hw4.StrategySpec(name, grids[name], makers[name])
        spec.columns = cols
        _, choices, _ = hw4.research_strategy(spec, labels, cal, all_fits)
        i, w, _ = choices[valid_fits[0]]
        if name == "fcf":  # HW4 final uses the walk-forward FCF sleeve
            sleeves[name] = hw4.piecewise_matrix(spec, choices, all_fits, len(cal), cache)
            last_fit = max(f for f in all_fits if f <= cal.get_loc(end))
            i_l, w_l, _ = choices[last_fit]
            params[name] = dict(params=spec.grid[i_l], wait=w_l, rule="walk-forward")
        else:
            sleeves[name] = hw4.shift_rows(spec.make(spec.grid[i]), w)
            params[name] = dict(params=spec.grid[i], wait=w, rule="fixed (2010-2012)")
        hw4.log(f"  {name}: {params[name]}")

    P = {n: hw4.pct_among_flagged(M) for n, M in sleeves.items()}
    F = {n: ~np.isnan(P[n]) for n in sleeves}
    events = {n: hw4.event_series(M, labels, waits=(0,))[0][0] for n, M in sleeves.items()}
    end0 = labels.end_row(0)
    W = hw4.trailing_strategy_weights(events, end0, len(cal), LOOKBACK, MODE)
    score = hw4.combine_weighted(P, F, W)
    return dict(
        md=md,
        cal=cal,
        cols=cols,
        end=end,
        first_entry=first_entry,
        sleeves=sleeves,
        params=params,
        events=events,
        end0=end0,
        W=W,
        score=score,
    )


def trailing_means(events, end0, row, lookback=LOOKBACK, min_obs=20):
    """Expected 90-day excess return per signal at ``row`` (HW4's mu-hat)."""
    known = np.searchsorted(end0, row, side="left")  # signal rows finished before row
    out = {}
    for n, v in events.items():
        window = np.asarray(v[max(known - lookback, 0) : known], dtype=float)
        window = window[np.isfinite(window)]
        out[n] = float(window.mean()) if window.size >= min_obs else None
    return out


def main(database=str(hw4.DEFAULT_DB)):
    st = build_hw4_state(database)
    cal, cols, end = st["cal"], st["cols"], st["end"]
    row = cal.get_loc(end)

    # ---- Snapshot cross-section and the fitted pipeline ----------------------
    X_all = pd.DataFrame(
        {SLEEVES[n]: st["sleeves"][n][row].astype(np.float32) for n in SLEEVES}, index=cols
    )[FEATURES]
    weights = {SLEEVES[n]: float(st["W"][n][row]) for n in SLEEVES}
    candidates = X_all.index[np.isfinite(st["score"][row])]
    X_ref = X_all.loc[X_all.notna().any(axis=1)]  # any flag (reference distributions)
    mu = trailing_means(st["events"], st["end0"], row)
    expected = {SLEEVES[n]: mu[n] for n in SLEEVES}
    # Check 0: my mu-hat recomputation implies exactly HW4's weights.
    pos_mu = {c: max(v or 0.0, 0.0) for c, v in expected.items()}
    implied = {c: v / sum(pos_mu.values()) for c, v in pos_mu.items()}
    mu_gap = max(abs(implied[c] - weights[c]) for c in FEATURES)
    assert mu_gap < 1e-5, f"mu-hat does not reproduce HW4 weights ({mu_gap})"
    pipeline = Pipeline(
        [
            ("percentiles", FlaggedPercentileTransformer()),
            ("score", SignalScoreRegressor(weights=weights, expected_returns=expected)),
        ]
    )
    pipeline.fit(X_ref)

    # ---- Check 1: exact parity with HW4's own score --------------------------
    hw4_score = pd.Series(st["score"][row], index=cols).loc[candidates].astype(float)
    served = pd.Series(pipeline.predict(X_all.loc[candidates]), index=candidates)
    parity = float((served - hw4_score).abs().max())
    assert parity < 1e-5, f"pipeline differs from HW4 score by {parity}"
    non_cand = X_all.index.difference(candidates)
    assert (pipeline.predict(X_all.loc[non_cand]) == 0.0).all()

    # ---- Check 2: weights match the HW4 output table --------------------------
    first_entry = st["first_entry"]
    rows = (cal >= first_entry) & (cal <= end)
    yearly = (
        pd.DataFrame({n: pd.Series(st["W"][n], index=cal)[rows] for n in SLEEVES})
        .groupby(cal[rows].year)
        .mean()
    )
    hw4_tab = pd.read_csv(HW4_DIR / "outputs/test_dynamic_weights_by_year.csv", index_col=0)
    weight_gap = float((yearly - hw4_tab[list(SLEEVES)]).abs().max().max())
    assert weight_gap < 1e-5, f"weights differ from HW4 output by {weight_gap}"

    # ---- Check 3: every HW4 test purchase had a finite score ------------------
    fills = pd.read_csv(HW4_DIR / "outputs/test_fills.csv", parse_dates=["signal_date"])
    buys = fills.loc[fills.side.eq("BUY") & fills.units.gt(1e-9)]
    pos = {s: j for j, s in enumerate(cols)}
    has_score = [
        np.isfinite(st["score"][cal.get_loc(d), pos[s]])
        for d, s in zip(buys.signal_date, buys.symbol)
    ]
    assert all(has_score), "a HW4 purchase had no score in the rebuilt matrix"

    # ---- Save the artifact, metadata, reference rows and example request -----
    ARTIFACTS.mkdir(exist_ok=True)
    bundle = {
        "pipeline": pipeline,
        "model_version": MODEL_VERSION,
        "features": FEATURES,
        "snapshot_date": f"{end:%Y-%m-%d}",
    }
    joblib.dump(bundle, ARTIFACTS / "hw4_model.joblib")

    info = st["md"].info
    ref = X_all.loc[candidates].copy()
    ref.insert(0, "name", info.name.reindex(candidates).to_numpy())
    ref["hw4_score"] = hw4_score
    ref["hw4_rank"] = hw4_score.rank(ascending=False, method="min").astype(int)
    # Expected excess return straight from HW4's arrays (not via the pipeline):
    # weight-averaged mu-hat over the positive-weight signals flagging the stock.
    w_live = pd.DataFrame(
        {
            SLEEVES[n]: np.where(st["sleeves"][n][row] == st["sleeves"][n][row], 1.0, 0.0)
            * float(st["W"][n][row])
            for n in SLEEVES
        },
        index=cols,
    ).loc[candidates][FEATURES]
    mu_vec = pd.Series({c: expected[c] or 0.0 for c in FEATURES})
    ref["hw4_expected_excess_return_90d"] = (w_live * mu_vec).sum(axis=1) / w_live.sum(axis=1)
    ref.sort_values("hw4_score", ascending=False).to_csv(
        ARTIFACTS / "hw4_reference_scores.csv", index_label="symbol", float_format="%.9g"
    )
    metadata = {
        "model_version": MODEL_VERSION,
        "hw4_final_model": "pairs+fcf+revenue+momentum [trailing 126d] | 90-day review top 40",
        "hw4_run": "run 10 (2026-11-12), see Homeworks/HW4/Philip_Szkudlarski/README.md",
        "snapshot_date": f"{end:%Y-%m-%d}",
        "decision": "score at the 2026-09-30 close; HW4 would buy at the next session's close",
        "features": FEATURES,
        "signal_parameters": {SLEEVES[n]: st["params"][n] for n in SLEEVES},
        "weights_trailing_126d": weights,
        "expected_90d_excess_return_trailing_126d": {SLEEVES[n]: mu[n] for n in SLEEVES},
        "reference_counts": {c: int(X_ref[c].notna().sum()) for c in FEATURES},
        "reference_ranges": {
            c: [float(X_ref[c].min()), float(X_ref[c].max())] for c in FEATURES
        },
        "n_candidates": int(len(candidates)),
        "checks": {
            "max_abs_diff_vs_hw4_score": parity,
            "max_abs_diff_weights_implied_by_mu_hat": mu_gap,
            "max_abs_diff_yearly_weights_vs_hw4_csv": weight_gap,
            "hw4_test_buys_with_finite_score": f"{sum(has_score)}/{len(has_score)}",
        },
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "joblib": joblib.__version__,
        },
        "training_data": "Lectures/modeling/data/equities_top2000.duckdb (bCourses)",
    }
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    top = ref.sort_values("hw4_score", ascending=False).iloc[0]
    example = {c: (None if pd.isna(top[c]) else float(top[c])) for c in FEATURES}
    (HERE / "example_request.json").write_text(json.dumps(example, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print("Top 10 candidates on", f"{end:%Y-%m-%d}")
    print(ref.sort_values("hw4_score", ascending=False).head(10).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
