"""Record one real 2026 HW4 input and direct model predictions for API tests."""

import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "Lectures/modeling/data/hw4_yisu_60_sessions.duckdb"
SCHEMA = json.loads((HERE / "feature_schema.json").read_text())
SELECTION = json.loads((HERE / "artifacts/selection.json").read_text())


def quote(name):
    return '"' + name.replace('"', '""') + '"'


columns = ", ".join(
    f"{quote(source)} AS {quote(model)}"
    for source, model in zip(SCHEMA["source_features"], SCHEMA["model_features"])
)
with duckdb.connect(str(DATA), read_only=True) as db:
    row = (
        db.execute(f"""
        SELECT PRICE_symbol, PRICE_date, {columns}
        FROM features WHERE PRICE_date >= '2026-01-01'
        ORDER BY PRICE_date, PRICE_symbol LIMIT 1
    """)
        .fetchdf()
        .iloc[0]
    )

features = SCHEMA["model_features"]
frame = pd.DataFrame([[row[name] for name in features]], columns=features, dtype="float64")
input_values = [None if pd.isna(value) else float(value) for value in frame.iloc[0]]
scores = {}
for member in SELECTION["selected_models"]:
    model = lgb.Booster(model_file=str(HERE / "artifacts" / member["file"]))
    scores[member["file"].removesuffix(".txt")] = float(
        model.predict(
            frame,
            num_iteration=member["best_round"],
            num_threads=SCHEMA["num_threads"],
            validate_features=True,
        )[0]
    )

request = {"features": input_values}
reference = {
    "source": "Direct 2026 HW4 LightGBM model predictions before API wrapping",
    "signal_date": pd.Timestamp(row["PRICE_date"]).date().isoformat(),
    "symbol": row["PRICE_symbol"],
    "model_version": SCHEMA["model_version"],
    "prediction": scores["model_01"],
    "model_scores": scores,
}
(HERE / "example_request.json").write_text(
    json.dumps(request, indent=2, allow_nan=False) + "\n"
)
(HERE / "tests/reference_prediction.json").write_text(
    json.dumps(reference, indent=2, allow_nan=False) + "\n"
)
print("Recorded", reference["symbol"], reference["signal_date"], scores)
