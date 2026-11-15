"""Save the HW4 model served here, and the example request with its HW4 prediction.

Runs the code cells of my HW4 notebook up to section 3 (data, features, model), then fits
version 2 (boosting) the way the final test did for 2026. Run from this folder:
python export_model.py [path/to/hw4.ipynb]
"""

import json
import sys

import joblib
import matplotlib

matplotlib.use("Agg")  # the notebook's plots are not needed

HW4_NOTEBOOK = "../../HW4/Paul_Zamanian/hw4.ipynb"
if len(sys.argv) > 1:
    HW4_NOTEBOOK = sys.argv[1]
DATE = "2026-09-29"  # last HW4 date with predictions (Sep 30 has no next session)

# Run HW4 sections 1 and 2. Its DB_PATH is relative and works from this folder too.
with open(HW4_NOTEBOOK) as f:
    cells = json.load(f)["cells"]
hw4 = {}
for cell in cells:
    source = "".join(cell["source"])
    if source.startswith("## 3. Backtester"):
        break
    if cell["cell_type"] == "code":
        exec(source, hw4)

# The final test refits on the first session of each year: this is the 2026 fit.
calendar = hw4["calendar"]
model = hw4["fit_model"]("boosting", calendar[calendar.year == 2026][0])
joblib.dump(model, "artifacts/hw4_v2_boosting_2026.joblib")

# Example input (AAPL) and its prediction from HW4's predict_scores, for the tests.
example = {name: float(hw4["features"][name].loc[DATE, "AAPL"]) for name in hw4["FEATURES"]}
with open("example_request.json", "w") as f:
    json.dump(example, f, indent=2)
scores = hw4["predict_scores"]("boosting", "2026-01-01", DATE)
print(f"HW4 prediction for AAPL on {DATE}: {float(scores.loc[DATE, 'AAPL'])!r}")
