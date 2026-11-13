"""HW4 task 2d (optional, time-boxed): Ridge vs HistGradientBoostingRegressor on the rank target.

Same annual walk-forward folds as stage4_dev_grid.py (fits 2012-2015, label_end
< fit date, all clean eligible rows scored). HGB uses sklearn defaults with
early stopping OFF (its built-in early stopping holds out a random, not
chronological, slice of the overlapping training labels) and random_state=0.
No tuning. Reports prediction quality only; Ridge numbers come from the cached
annual scores of stage4_dev_grid.py.

Run from the repository root:
    .venv/Scripts/python -B Homeworks/HW4/Ryan_Beauchamp/stage4d_hgb.py

Credits: make_pipeline/FEATURES from Lectures/modeling/course_data.py.
Written with AI assistance (Claude Code) and reviewed by the author.
"""

import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import walkforward as wf  # noqa: E402
import stage4_dev_grid as g  # noqa: E402

TIME_BOX_SECONDS = 30 * 60


def main():
    start = time.perf_counter()
    frame, _ = wf.load_model_rows(g.PANEL, g.MEASURES, "2010-01-01", "2015-12-31")
    fits = wf.fit_dates(frame.date.unique(), g.BT_START, g.BT_END, "annual")
    pieces = []
    for k, fit in enumerate(fits):
        if time.perf_counter() - start > TIME_BOX_SECONDS:
            print("Time box reached; stopping early.")
            break
        t = time.perf_counter()
        next_fit = fits[k + 1] if k + 1 < len(fits) else g.BT_END + pd.Timedelta(days=1)
        train = frame.loc[frame.date.ge("2010-01-01") & frame.target.notna() & frame.label_end.lt(fit)]
        assert train.label_end.max() < fit
        model = wf.cd.make_pipeline(HistGradientBoostingRegressor(early_stopping=False, random_state=0))
        model.fit(train[wf.cd.FEATURES], wf.date_rank(train.date, train.target.to_numpy()))
        rows = frame.loc[frame.date.ge(fit) & frame.date.lt(next_fit)]
        pieces.append(rows[["date", "symbol"]].assign(score=model.predict(rows[wf.cd.FEATURES]), fit_date=fit))
        print(f"  HGB fit {fit.date()}: {len(train):,} rows, {time.perf_counter() - t:.0f}s")
    hgb = pd.concat(pieces, ignore_index=True)
    hgb.to_parquet(g.OUT / "stage4d_scores_hgb_annual.parquet", index=False)
    ridge = pd.read_parquet(g.OUT / "stage4_scores_annual.parquet")
    quality = pd.concat({"Ridge (annual)": g.prediction_quality(frame, ridge),
                         "HGB (annual)": g.prediction_quality(frame, hgb)}, axis=1)
    print("\nPrediction quality, annual refits (rows with known targets):")
    print(quality.to_string(float_format=lambda x: f"{x:+.4f}"))
    quality.to_csv(g.OUT / "stage4d_prediction_quality.csv")
    print(f"\nTotal {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
