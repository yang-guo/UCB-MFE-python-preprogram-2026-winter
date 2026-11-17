"""Feature contract for the HW4 ridge-rank model served by this assignment.

The names and order are ``MODEL_FEATURES`` from Homework 4, which is
``NUMERIC_FEATURES`` in ``Lectures/modeling/course_data.py``. Sector is excluded
because that field was not point-in-time. Bounds are the domain of each
definition, wide enough for the training sample; the saved ``QuantileClipper``
still winsorizes each column to the training 1st and 99th percentiles.
"""

MODEL_VERSION = "hw4-ridge-rank-dev-labels-before-2016-04-01"

FEATURE_NAMES = [
    "momentum_21",
    "momentum_63",
    "momentum_252_21",
    "volatility_63",
    "drawdown_252",
    "net_margin",
    "operating_roa",
    "debt_to_assets",
    "cash_to_assets",
    "fcf_to_assets",
    "accruals_to_assets",
    "filing_age_days",
]

# gt is exclusive. ge/le are inclusive. Units are described in app.InputData.
FEATURE_BOUNDS: dict[str, dict[str, float]] = {
    "momentum_21": {"gt": -1.0, "le": 100.0},
    "momentum_63": {"gt": -1.0, "le": 100.0},
    "momentum_252_21": {"gt": -1.0, "le": 100.0},
    "volatility_63": {"ge": 0.0, "le": 100.0},
    "drawdown_252": {"ge": -1.0, "le": 0.0},
    "net_margin": {"ge": -1000.0, "le": 1000.0},
    "operating_roa": {"ge": -100.0, "le": 100.0},
    "debt_to_assets": {"ge": -10.0, "le": 1000.0},
    "cash_to_assets": {"ge": -10.0, "le": 100.0},
    "fcf_to_assets": {"ge": -100.0, "le": 100.0},
    "accruals_to_assets": {"ge": -100.0, "le": 100.0},
    "filing_age_days": {"ge": 0.0, "le": 10000.0},
}
