"""Export the last annual HW4 fit (fit date 2026-01-02) as a joblib pipeline.

HW4's walkforward.py fitted each model in a local variable and kept only the
scores, so this script refits the 2026-01-02 model with exactly the same rows,
target and pipeline, proves it reproduces HW4's recorded scores, and saves it.

Run with the HW4 environment (same library versions as training), from this folder:
    <repo>/.venv/Scripts/python.exe -B export_model.py --hw4-dir <repo>/Homeworks/HW4/Ryan_Beauchamp

Outputs (all in this folder): model.joblib, model_metadata.json, feature_ranges.csv,
sector_counts.csv, example_request.json, hw4_reference.json.

Written with AI assistance (Claude Code) and reviewed by the author.
"""

import sys

# Never write __pycache__ into the read-only HW4 checkout or Lectures/ when importing.
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
import threadpoolctl  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from quantile_clipper import QuantileClipper  # noqa: E402  (our shipped copy)

HERE = Path(__file__).resolve().parent  # the HW5 submission folder; outputs go here

# Frozen HW4 configuration for the last annual fit.
FIT_DATE = pd.Timestamp("2026-01-02")
TRAIN_START = "2010-01-01"
SCORE_END = "2026-09-29"
ALPHA = 1000.0
MODEL_VERSION = "hw4-ridge-rank-a1000-fit-2026-01-02"
EXPECTED_TRAIN_ROWS = 7_774_241
EXPECTED_LAST_LABEL_END = pd.Timestamp("2025-12-31")
EXPECTED_SCORED_ROWS = 365_938
MAX_ABS_DIFF = 1e-9          # required agreement with HW4's recorded scores
REFERENCE_DATE = pd.Timestamp("2026-06-30")  # example row: last session on/before this
PREFERRED_SYMBOL = "AAPL"
N_EXTRA_REFERENCES = 5
TARGET_DEFINITION = ("Per signal date, the cross-sectional percentile rank (pandas rank pct=True) "
                     "of the 90-day forward return, minus 0.5 (range about -0.5..0.5).")


def stop(message):
    """Abort with a clear message (used for every check the user asked for)."""
    sys.exit(f"STOP: {message}")


def git(hw4_dir, *args):
    """Read-only git query against the HW4 checkout."""
    return subprocess.run(["git", "-C", str(hw4_dir), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def build_pipeline(numeric):
    """Same structure and step names as course_data.make_pipeline(Ridge(alpha=1000)),
    but using quantile_clipper.QuantileClipper so the pickle does not need Lectures/."""
    numeric_steps = Pipeline([
        ("clip", QuantileClipper()),  # clip tails at train 1st/99th percentiles
        ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                 keep_empty_features=True)),  # median fill + missing flags
        ("scale", StandardScaler()),  # zero mean, unit variance
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_steps, numeric),
        ("sector", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sector"]),
    ], remainder="drop")
    return Pipeline([("preprocess", preprocess), ("model", Ridge(alpha=ALPHA))])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hw4-dir", required=True, type=Path,
                        help="HW4 submission folder (contains walkforward.py, outputs/, results/)")
    hw4 = parser.parse_args().hw4_dir.resolve()
    repo = Path(git(hw4, "rev-parse", "--show-toplevel")).resolve()

    def rel(path):
        """Repo-root-relative POSIX path, so no local absolute path is written to disk."""
        return Path(path).resolve().relative_to(repo).as_posix()

    panel_dir = hw4 / "outputs" / "panel_full"
    measures_path = hw4 / "outputs" / "clean_measures_full" / "measures.parquet"
    scores_path = hw4 / "outputs" / "final" / "scores.parquet"
    fit_log_path = hw4 / "results" / "fit_log.csv"

    # a. Load rows with HW4's own loader (clean flag, eligible rows, model columns).
    sys.path.insert(0, str(hw4))
    import walkforward as wf  # also imports course_data (only used here for feature names)

    numeric = list(wf.cd.NUMERIC_FEATURES)
    columns = numeric + ["sector"]  # feature order: 12 numeric, then sector
    print("Loading HW4 model rows ...", flush=True)
    frame, n_broken = wf.load_model_rows(panel_dir, measures_path, TRAIN_START, SCORE_END)
    print(f"  frame rows {len(frame):,} (clean flag removed {n_broken:,})")

    # b. Training rows: identical mask to wf.walk_forward_scores for this fit date.
    train = frame.loc[frame.date.ge(pd.Timestamp(TRAIN_START)) & frame.target.notna()
                      & frame.label_end.lt(FIT_DATE)]
    last_label_end = train.label_end.max()
    if len(train) != EXPECTED_TRAIN_ROWS:
        stop(f"train rows {len(train):,} != expected {EXPECTED_TRAIN_ROWS:,}")
    if last_label_end != EXPECTED_LAST_LABEL_END:
        stop(f"last train label_end {last_label_end.date()} != {EXPECTED_LAST_LABEL_END.date()}")
    # Cross-check against the row HW4 logged for this fit.
    fit_log = pd.read_csv(fit_log_path, parse_dates=["fit_date", "last_train_label_end"])
    logged = fit_log.loc[fit_log.fit_date.eq(FIT_DATE)]
    if len(logged) != 1:
        stop(f"fit_log.csv has {len(logged)} rows for {FIT_DATE.date()}")
    logged = logged.iloc[0]
    if (logged.train_rows != len(train) or logged.last_train_label_end != last_label_end
            or logged.scored_rows != EXPECTED_SCORED_ROWS):
        stop(f"fit_log.csv disagrees: {logged.to_dict()}")
    print(f"  train rows {len(train):,}, last label_end {last_label_end.date()} (matches fit_log)")

    # c. Fit: same y as walkforward.py (date-wise percentile rank of target, centred).
    print("Fitting Ridge pipeline ...", flush=True)
    model = build_pipeline(numeric)
    model.fit(train[columns], wf.date_rank(train.date, train.target.to_numpy()))

    # d. Verify against HW4's recorded scores for this fit.
    rows = frame.loc[frame.date.ge(FIT_DATE) & frame.date.le(pd.Timestamp(SCORE_END))]
    if len(rows) != EXPECTED_SCORED_ROWS:
        stop(f"scored rows {len(rows):,} != expected {EXPECTED_SCORED_ROWS:,}")
    new = rows[["date", "symbol"]].assign(score_new=model.predict(rows[columns]))
    hw4_scores = pd.read_parquet(scores_path, filters=[("fit_date", "==", FIT_DATE)])
    if len(hw4_scores) != EXPECTED_SCORED_ROWS:
        stop(f"HW4 scores for {FIT_DATE.date()}: {len(hw4_scores):,} rows")
    both = new.merge(hw4_scores[["date", "symbol", "score"]], on=["date", "symbol"],
                     how="outer", indicator=True, validate="one_to_one")
    unmatched = both._merge.ne("both").sum()
    if unmatched:
        stop(f"{unmatched:,} rows not in both sets: {both._merge.value_counts().to_dict()}")
    abs_diff = (both.score_new - both.score).abs()
    max_diff, mean_diff = float(abs_diff.max()), float(abs_diff.mean())
    print(f"  compared {len(both):,} rows: max abs diff {max_diff:.3e}, mean {mean_diff:.3e}")
    if not max_diff < MAX_ABS_DIFF:
        stop(f"max abs diff {max_diff:.3e} is not < {MAX_ABS_DIFF:g}; not saving the model")

    # g (selection). Reference rows read DIRECTLY from HW4's scores, joined to panel features.
    ref_date = hw4_scores.date[hw4_scores.date.le(REFERENCE_DATE)].max()
    day = hw4_scores.loc[hw4_scores.date.eq(ref_date), ["date", "symbol", "score", "fit_date"]]
    day = day.merge(frame[["date", "symbol"] + columns], on=["date", "symbol"],
                    how="left", validate="one_to_one")
    qualifies = np.isfinite(day[numeric].to_numpy(dtype=float)).all(axis=1) & day.sector.notna()
    candidates = day.loc[qualifies].sort_values("symbol").reset_index(drop=True)
    if candidates.empty:
        stop(f"no qualifying rows on {ref_date.date()}")
    first = (candidates.symbol.eq(PREFERRED_SYMBOL).idxmax()
             if candidates.symbol.eq(PREFERRED_SYMBOL).any() else 0)
    order = [first] + [i for i in range(len(candidates)) if i != first][:N_EXTRA_REFERENCES]
    references = candidates.loc[order].reset_index(drop=True)

    def features(row):
        """One row's model inputs in feature order, as plain JSON types (no rounding)."""
        return {c: (str(row[c]) if c == "sector" else float(row[c])) for c in columns}

    # e. Save the model, then prove it reloads WITHOUT course_data / Lectures/.
    model_path = HERE / "model.joblib"
    joblib.dump(model, model_path)
    blob = model_path.read_bytes()
    if b"course_data" in blob or b"quantile_clipper" not in blob:
        stop("model.joblib does not reference quantile_clipper (or still references course_data)")
    reload_code = (
        "import sys, json; sys.path.insert(0, sys.argv[1]);"
        "import joblib, pandas as pd;"
        "m = joblib.load(sys.argv[2]);"
        "X = pd.DataFrame(json.loads(sys.stdin.read()));"
        "clip = m.named_steps['preprocess'].named_transformers_['numeric'].named_steps['clip'];"
        "assert type(clip).__module__ == 'quantile_clipper', type(clip).__module__;"
        "assert 'course_data' not in sys.modules;"
        "print(json.dumps([float(v) for v in m.predict(X)]))"
    )
    # The reference rows as JSON records (exact floats: json writes repr, which round-trips).
    ref_json = json.dumps([features(r) for _, r in references.iterrows()])
    # -I: isolated (no cwd / script dir / PYTHON* env on sys.path); only this folder is added.
    reloaded = subprocess.run(
        [sys.executable, "-I", "-B", "-c", reload_code, str(HERE), str(model_path)],
        input=ref_json, capture_output=True, text=True)
    if reloaded.returncode:
        stop(f"clean reload failed:\n{reloaded.stderr}")
    # Recompute in-memory on the identical JSON-parsed input, then require exact equality.
    in_memory = model.predict(pd.DataFrame(json.loads(ref_json)))
    if not np.array_equal(np.array(json.loads(reloaded.stdout)), in_memory):
        stop("reloaded model predictions differ from the in-memory model")
    print("  model.joblib reloads in a clean process using quantile_clipper only")

    # Metadata: everything needed to identify and reproduce this artifact.
    encoder = model.named_steps["preprocess"].named_transformers_["sector"]
    categories = list(encoder.categories_[0])
    nan_is_category = any(pd.isna(c) for c in categories)
    metadata = {
        "model_version": MODEL_VERSION,
        "feature_order": columns,
        "numeric_features": numeric,
        "sector_categories": [None if pd.isna(c) else str(c) for c in categories],
        "sector_nan_is_category": bool(nan_is_category),
        "alpha": ALPHA,
        "fit_date": str(FIT_DATE.date()),
        "train_start": TRAIN_START,
        "train_rows": len(train),
        "last_train_label_end": str(last_label_end.date()),
        "target_definition": TARGET_DEFINITION,
        "library_versions": {
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__,
        },
        "model_sha256": hashlib.sha256(blob).hexdigest(),
        "verification": {
            "rows_compared": len(both), "max_abs_diff": max_diff, "mean_abs_diff": mean_diff,
            "threshold": MAX_ABS_DIFF, "hw4_scores_file": rel(scores_path),
        },
        "hw4_git_commit": git(hw4, "rev-parse", "HEAD"),
        "hw4_sources": {"panel": rel(panel_dir), "measures": rel(measures_path),
                        "fit_log": rel(fit_log_path), "walkforward": rel(hw4 / "walkforward.py")},
    }
    (HERE / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # f. Training-row distributions, used to justify API input bounds.
    X = train[numeric]
    ranges = pd.DataFrame({
        "count": X.notna().sum(), "share_missing": X.isna().mean(), "min": X.min(),
        "p01": X.quantile(0.01), "p50": X.quantile(0.50), "p99": X.quantile(0.99),
        "max": X.max(),
    })
    ranges.index.name = "feature"
    ranges.to_csv(HERE / "feature_ranges.csv")
    sectors = train.sector.astype(object).fillna("<missing>").value_counts()
    sectors.rename_axis("sector").rename("count").to_csv(HERE / "sector_counts.csv")

    # g (write). Example request = first reference row, exact panel values (repr round-trips).
    example = features(references.iloc[0])
    (HERE / "example_request.json").write_text(json.dumps(example, indent=2) + "\n")
    records = [{"symbol": r.symbol, "signal_date": str(r.date.date()),
                "fit_date": str(r.fit_date.date()), "hw4_score": float(r.score),
                "source_file": rel(scores_path), "features": features(r)}
               for _, r in references.iterrows()]
    (HERE / "hw4_reference.json").write_text(json.dumps(records, indent=2) + "\n")

    # No absolute local paths may appear in any output file.
    outputs = ["model.joblib", "model_metadata.json", "feature_ranges.csv",
               "sector_counts.csv", "example_request.json", "hw4_reference.json"]
    patterns = [b"C:\\", b"C:/", b"/c/", b"Users"]
    for name in outputs:
        data = (HERE / name).read_bytes()
        found = [p.decode() for p in patterns if p in data]
        if found:
            stop(f"{name} contains local path text {found}")

    # Summary.
    print("\nSUMMARY")
    print(f"  frame rows {len(frame):,} | train rows {len(train):,} | scored rows {len(rows):,}")
    print(f"  max abs diff vs HW4 {max_diff:.3e} | mean {mean_diff:.3e}")
    print(f"  model.joblib {model_path.stat().st_size:,} bytes, sha256 {metadata['model_sha256']}")
    print(f"  libraries {metadata['library_versions']}")
    print(f"  sector categories {len(categories)} (NaN is a category: {nan_is_category})")
    print(f"  example {records[0]['symbol']} {records[0]['signal_date']} "
          f"HW4 score {records[0]['hw4_score']!r}")
    print(f"  {json.dumps(example)}")
    print(f"  references: {[r['symbol'] for r in records]}")
    print(f"  outputs: {', '.join(outputs)} (no local paths found)")


if __name__ == "__main__":
    main()
