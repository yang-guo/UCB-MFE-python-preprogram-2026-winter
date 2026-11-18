"""Fit the chosen HW4 model once on all labeled rows and serialize it with its input contract."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from course_data import FEATURES, NUMERIC_FEATURES
from hw4_model import OUT, fit_spec

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# Fill these in by hand from course_data.py (price_features / join_financials).
UNITS = {
    # "ret_21d": "fractional return over 21 sessions, 0.05 = 5%",
    "momentum_21":"", 
    "momentum_63":"", 
    "momentum_252_21":"", 
    "volatility_63":"", 
    "drawdown_252":"",
    "net_margin":"", 
    "operating_roa":"", 
    "debt_to_assets":"", 
    "cash_to_assets":"", 
    "fcf_to_assets":"", 
    "accruals_to_assets":"",
    "filing_age_days":"",
    "sector":""
}


def build_contract(train, model_name, model_file, digest):
    features = []
    for f in FEATURES:
        if f in NUMERIC_FEATURES:
            v = pd.to_numeric(train[f], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            lo, hi = float(v.min()), float(v.max())
            pad = 0.5 * (hi - lo) or 1.0          # generous margin: reject only absurd values
            features.append({"name": f, "kind": "numeric", "min": lo - pad, "max": hi + pad,
                             "unit": UNITS.get(f, "unit not documented")})
        else:
            cats = sorted(train[f].dropna().astype(str).unique())
            features.append({"name": f, "kind": "categorical", "categories": cats,
                             "unit": "category label"})
    version = f"{model_name}-{datetime.now(timezone.utc):%Y%m%d}-{digest[:8]}"
    return {"model_file": model_file, "model_sha256": digest, "model_version": version,
            "sklearn_version": sklearn.__version__,
            "target": "90-day forward return (same units as training `target`)",
            "features": features}   # order here == training column order

def write_reference(model, train, model_path, contract):
    """Record one real panel row and the pipeline's own prediction for it.

    Provenance: the input is an actual row from outputs/hw4/panel.pkl (the HW4 feature table),
    and the expected value is model.predict(train.loc[[i], FEATURES]) on the freshly fitted
    pipeline, exactly as HW4 calls it. No API code is involved.
    """
    complete = train[FEATURES].replace([np.inf, -np.inf], np.nan).dropna()  # API requires every feature
    idx = complete.index[len(complete) // 2]                                # deterministic choice
    frame = train.loc[[idx], FEATURES]                                      # training-style DataFrame
    expected = float(model.predict(frame)[0])

    reloaded = joblib.load(model_path)                                      # serialization round trip
    assert np.isclose(float(reloaded.predict(frame)[0]), expected, rtol=1e-9, atol=1e-12)

    row = frame.iloc[0]
    payload = {c: (float(row[c]) if c in NUMERIC_FEATURES else str(row[c])) for c in FEATURES}
    ref = {"source": "HW4 fitted pipeline .predict on a real panel row, recorded before the API",
           "panel_row": {"symbol": str(train.loc[idx, "symbol"]), "date": str(train.loc[idx, "date"])},
           "model_sha256": contract["model_sha256"], "model_version": contract["model_version"],
           "sklearn_version": sklearn.__version__,
           "input": payload, "expected_prediction": expected}
    tests = Path(__file__).resolve().parent / "tests"
    tests.mkdir(exist_ok=True)
    (tests / "reference_case.json").write_text(json.dumps(ref, indent=2))
    
def main():
    ARTIFACTS.mkdir(exist_ok=True)
    panel = pd.read_pickle(OUT / "panel.pkl")
    chosen = json.loads((OUT / "chosen_model.json").read_text())
    name = chosen.pop("name")
    train = panel[panel.target.notna()]               # only rows with a completed 90-day label
    model = fit_spec(chosen, train)                   # fitted pipeline: preprocessing + estimator

    path = ARTIFACTS / "model.joblib"
    joblib.dump(model, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    contract = build_contract(train, name, path.name, digest)
    (ARTIFACTS / "feature_contract.json").write_text(json.dumps(contract, indent=2))
    write_reference(model, train, path, contract)


if __name__ == "__main__":
    main()