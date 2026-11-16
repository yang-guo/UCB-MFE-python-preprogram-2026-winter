"""One-off export of the HW4 model into the artifact this service loads.

Run with the HW4 training environment (repo-root ``uv`` env, scikit-learn 1.8.0) and
the HW4 database; the service itself never runs this file or touches the database:

    ../../../.venv/bin/python export_hw4_model.py --hw4 ../../HW4/Dhruv_Chopra

It refits exactly the model HW4's submitted strategy used for its last quarter
(``model.fit("ridge", panel, fit_t)`` at the first session of 2026-Q3, the same call
``ModelSchedule`` makes), then writes:

* ``artifacts/hw4_ridge_v1.joblib``: the sklearn pipeline plus metadata;
* ``tests/hw4_reference.json``: real ranked inputs from the last HW4 signal date and
  the predictions HW4's own ``RidgeOnRanks.predict`` gave for them;
* ``example_request.json``: one of those inputs (AAPL).
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIT_QUARTER_START = "2026-07-01"
SIGNAL_DATE = "2026-09-28"  # last signal close of the HW4 final test (final_cohorts.csv)
MODEL_VERSION = "hw4-v1-ridge-a10-fit-2026-07-01"
REFERENCE_SYMBOLS = ["AAPL", "MSFT", "JPM"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hw4", type=Path, required=True, help="HW4 submission folder")
    args = parser.parse_args()
    sys.path.insert(0, str(args.hw4.resolve()))
    sys.path.insert(0, str(HERE))

    import sklearn
    from features import FEATURES as HW4_FEATURES
    from model import cross_section, fit
    from pipeline import setup

    from ranks_model import FEATURES, build_pipeline

    assert FEATURES == HW4_FEATURES, "Serving feature order must match HW4"

    data, feats, _target, _label_end, panel = setup()
    cal = data.calendar
    fit_t = int(cal.searchsorted(pd.Timestamp(FIT_QUARTER_START)))
    hw4_model = fit("ridge", panel, fit_t, FEATURES)
    pipe = build_pipeline(hw4_model.model)

    # Score the last signal date exactly as HW4's ModelSchedule.scores does.
    t = int(cal.get_indexer([pd.Timestamp(SIGNAL_DATE)], method="pad")[0])
    cols, X = cross_section(data, feats, t, features=FEATURES)
    assert len(cols), f"No eligible stocks on {cal[t].date()}"
    hw4_pred = hw4_model.predict(X)
    frame = pd.DataFrame(X.astype(float), columns=FEATURES)
    pipe_pred = pipe.predict(frame)
    np.testing.assert_allclose(pipe_pred, hw4_pred, rtol=0, atol=1e-6)

    symbols = list(data.symbols[cols])
    picks = [symbols.index(s) for s in REFERENCE_SYMBOLS if s in symbols]
    # Also keep the first stock with a missing feature, to exercise null handling.
    picks.append(int(np.flatnonzero(np.isnan(X).any(axis=1))[0]))

    def as_json_row(i):
        return {f: (None if np.isnan(v) else float(v)) for f, v in zip(FEATURES, X[i])}

    cases = [
        {"symbol": symbols[i], "features": as_json_row(i), "prediction": float(hw4_pred[i])}
        for i in picks
    ]

    (HERE / "artifacts").mkdir(exist_ok=True)
    bundle = {
        "pipeline": pipe,
        "model_version": MODEL_VERSION,
        "features": FEATURES,
        "fit_date": str(cal[fit_t].date()),
        "n_train": hw4_model.n_train_,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, HERE / "artifacts/hw4_ridge_v1.joblib")
    reference = {
        "source": "HW4 RidgeOnRanks.predict on HW4 cross_section(...) rows",
        "model_version": MODEL_VERSION,
        "fit_date": bundle["fit_date"],
        "signal_date": str(cal[t].date()),
        "n_train": hw4_model.n_train_,
        "cases": cases,
    }
    (HERE / "tests/hw4_reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    (HERE / "example_request.json").write_text(
        json.dumps(cases[0]["features"], indent=2) + "\n"
    )
    print(f"max score among all {len(cols)} eligible stocks: {hw4_pred.max():.6f}")
    print(f"fit {bundle['fit_date']} n_train={hw4_model.n_train_:,}; scored {cal[t].date()}")
    for c in cases:
        print(f"  {c['symbol']:6s} {c['prediction']:.6f}")


if __name__ == "__main__":
    main()
