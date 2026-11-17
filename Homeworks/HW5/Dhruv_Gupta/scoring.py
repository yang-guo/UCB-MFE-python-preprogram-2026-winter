"""Load the frozen HW4 pipeline and score one feature row.

Request parsing stays in ``app.py``. This module only checks the artifact and
turns an already validated feature mapping into the DataFrame the fitted
``ColumnTransformer`` expects.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "artifacts" / "model.joblib"
EXAMPLE_PATH = ROOT / "example_request.json"
MODEL_VERSION = "hw4-hgb-deep-static-2016-04-01"
TRAINING_VERSIONS = {
    "sklearn": "1.9.1",
    "numpy": "2.5.4",
    "pandas": "3.1.1",
    "scipy": "1.18.1",
    "joblib": "1.6.0",
}

# Order is the HW4 training matrix order, including the categorical column.
FEATURE_COLUMNS = [
    "momentum_5",
    "momentum_21",
    "momentum_63",
    "momentum_126",
    "momentum_252_21",
    "volatility_21",
    "volatility_63",
    "drawdown_252",
    "volume_ratio_21",
    "large_cap_score",
    "net_margin",
    "operating_roa",
    "debt_to_assets",
    "cash_to_assets",
    "fcf_to_assets",
    "accruals_to_assets",
    "filing_age_days",
    "market_momentum_63",
    "market_volatility_63",
    "sector_relative_momentum",
    "sector_profitability_rank",
    "sector",
]
SECTORS = (
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
)


def _running_versions() -> dict[str, str]:
    import numpy
    import pandas
    import scipy
    import sklearn

    return {
        "sklearn": sklearn.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }


def load_fitted_pipeline() -> dict:
    """Load the trusted joblib bundle once. Do not fit it again."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # Import after ROOT is on the path so unpickling binds this QuantileClipper.
    import course_data

    bundle = joblib.load(ARTIFACT_PATH)
    if bundle.get("model_version") != MODEL_VERSION:
        raise RuntimeError(
            f"Artifact version {bundle.get('model_version')!r} is not {MODEL_VERSION}"
        )
    if list(bundle.get("feature_columns", [])) != FEATURE_COLUMNS:
        raise RuntimeError("Artifact feature order does not match the API contract")
    versions = _running_versions()
    trained = bundle.get("library_versions", {})
    if versions != trained or versions != TRAINING_VERSIONS:
        raise RuntimeError(
            "Refusing to load the HW4 artifact with a different library version. "
            f"Running {versions}, artifact {trained}."
        )
    pipeline = bundle.get("pipeline")
    if not isinstance(pipeline, Pipeline):
        raise RuntimeError("Artifact does not contain a fitted sklearn Pipeline")
    preprocess = pipeline.named_steps["preprocess"]
    clipper = preprocess.named_transformers_["numeric"].named_steps["clip"]
    if not isinstance(clipper, course_data.QuantileClipper):
        raise RuntimeError("Artifact is missing the HW4 QuantileClipper")
    learned_sectors = tuple(preprocess.named_transformers_["sector"].categories_[0])
    if learned_sectors != SECTORS:
        raise RuntimeError(f"Learned sector categories {learned_sectors} do not match the API")
    return bundle


def feature_frame(record: Mapping[str, object]) -> pd.DataFrame:
    """One row in the HW4 column order, with the training column names."""
    row = {column: record[column] for column in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_return(pipeline: Pipeline, record: Mapping[str, object]) -> float:
    """Return the 90-day holding-return forecast. Raise if it is not finite."""
    prediction = float(pipeline.predict(feature_frame(record))[0])
    if not math.isfinite(prediction):
        raise ValueError("Model returned a non-finite prediction")
    return prediction
