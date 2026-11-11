"""Run the selected HW4 LightGBM models without refitting them."""

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd


class HW4Models:
    def __init__(self, folder: Path):
        schema = json.loads((folder / "feature_schema.json").read_text())
        selection = json.loads((folder / "artifacts/selection.json").read_text())
        self.features = schema["model_features"]
        self.threads = schema["num_threads"]
        self.models = [
            (member, lgb.Booster(model_file=str(folder / "artifacts" / member["file"])))
            for member in selection["selected_models"]
        ]

    def predict(self, features: list[float | None]) -> dict[str, float]:
        frame = pd.DataFrame([features], columns=self.features, dtype="float64")
        return {
            member["file"].removesuffix(".txt"): float(
                model.predict(
                    frame,
                    num_iteration=member["best_round"],
                    num_threads=self.threads,
                    validate_features=True,
                )[0]
            )
            for member, model in self.models
        }
