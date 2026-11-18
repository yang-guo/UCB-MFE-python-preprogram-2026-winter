"""Record one real HW4 input and its prediction from the saved HW4 pipeline.

Writes tests/reference_example.json, the reference used by the valid-prediction test.
Run with the course environment (it needs the full course_data module and duckdb) and
the HW4 database at Lectures/modeling/data/equities_top2000.duckdb:

    <repo>/.venv/bin/python scripts/make_reference.py <repo root>
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
REPO = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(REPO / "Lectures/modeling"))  # the full course module, not the slim copy
import course_data  # noqa: E402

SIGNAL_DATE = pd.Timestamp("2016-03-31")  # first HW4 signal date

warnings.filterwarnings("ignore")
# End after the signal date so that quarter's universe membership is complete.
panel = course_data.prepare_panel(
    database=REPO / "Lectures/modeling/data/equities_top2000.duckdb", end="2016-06-30"
)
rows = panel.loc[panel.eligible & panel.date.eq(SIGNAL_DATE)].copy()
model = joblib.load(HERE / "artifacts/model.joblib")
rows["score"] = model.predict(rows[course_data.FEATURES])

# The API requires all 13 features, so pick the top-scored row with none missing.
complete = rows.dropna(subset=course_data.FEATURES).sort_values("score", ascending=False)
pick = complete.iloc[0]
example = {f: pick[f] if f == "sector" else float(pick[f]) for f in course_data.FEATURES}

result = {
    "source": "HW4 panel (course_data.prepare_panel on equities_top2000.duckdb), first HW4 signal date",
    "symbol": pick.symbol,
    "date": str(SIGNAL_DATE.date()),
    "input": example,
    "expected_prediction": float(pick.score),
    "eligible_rows_that_date": int(len(rows)),
    "complete_rows_that_date": int(len(complete)),
}
(HERE / "tests/reference_example.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
