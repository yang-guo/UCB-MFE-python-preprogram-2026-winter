import json
from pathlib import Path

import joblib
import pandas as pd

hw5_dir = Path(__file__).resolve().parent
features_full = ["momentum_63", "volatility_63"]


print("HW5 environment works.")

if (hw5_dir / "model.joblib").is_file() and (hw5_dir / "hw4_reference.json").is_file():
    # We load the same fitted model from HW4
    model_full = joblib.load(hw5_dir / "model.joblib")
    reference = json.loads((hw5_dir / "hw4_reference.json").read_text())

    # We keep the same features and the same order
    example = pd.DataFrame([reference["features"]])[features_full]
    prediction = model_full.predict(example)

    print(example)
    print(f"Prediction: {prediction[0]}")
    print(f"Original HW4 prediction: {reference['prediction']}")
else:
    print("First save model_full and one example from the HW4 notebook.")
