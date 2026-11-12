# Run this file from the HW4 notebook with %run -i to reuse its fitted model.
import json
from pathlib import Path

import joblib

if not all(name in globals() for name in ["model_full", "predictions_full", "features_full"]):
    raise RuntimeError(
        "Run this from the HW4 notebook after model_full and predictions_full exist. This file does not train the model."
    )

# We get the fitted objects supplied by the HW4 notebook.
model_full = globals()["model_full"]
predictions_full = globals()["predictions_full"]
features_full = globals()["features_full"]

hw5_dir = Path(__file__).resolve().parent

# We save the fitted scaler and Ridge model together
joblib.dump(model_full, hw5_dir / "model.joblib")

# We keep one input and its prediction from HW4
example = predictions_full[features_full].iloc[[0]].copy()
prediction = model_full.predict(example)

reference = {
    "features": example.iloc[0].to_dict(),
    "prediction": float(prediction[0]),
    "model_version": "hw4-model-full-2016-04-01",
    "source": "model_full and the first row of predictions_full in hw4.ipynb",
}

(hw5_dir / "hw4_reference.json").write_text(json.dumps(reference, indent=2))

print("HW4 model saved for HW5.")
print(example)
print(f"Prediction: {prediction[0]}")
