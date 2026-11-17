"""Frozen HW4 per-stock FF5 regressions and dated universe lookup."""

import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np

FEATURE_NAMES = ("MKT", "SMB", "HML", "RMW", "CMA")


class SymbolUnavailable(Exception):
    """Expected request error, distinct from an unexpected inference failure."""

    def __init__(self, symbol, code, reason=None):
        self.symbol, self.code, self.reason = symbol, code, reason
        super().__init__(code)


class FrozenFF5:
    def __init__(self, payload, membership):
        if tuple(payload["feature_names"]) != FEATURE_NAMES:
            raise ValueError("Incorrect factor order")
        if payload["prediction_horizon_calendar_days"] != 90:
            raise ValueError("Incorrect target horizon")
        self.model_version = payload["model_version"]
        self.snapshot_date = payload["snapshot_date"]
        if (
            membership["model_version"] != self.model_version
            or membership["snapshot_date"] != self.snapshot_date
        ):
            raise ValueError("Membership snapshot mismatch")
        rows = membership["members"]
        self.universe = frozenset(r["symbol"] for r in rows)
        self.eligible = frozenset(r["symbol"] for r in rows if r["eligible"])
        if len(self.universe) != len(rows):
            raise ValueError("Duplicate universe symbols")
        models = payload["models"]
        if not models or not set(models) <= self.eligible:
            raise ValueError("Model coverage violates universe eligibility")
        for row in rows:
            available = row["symbol"] in models
            reason = (
                None
                if available
                else ("NO_VALID_REGRESSION" if row["eligible"] else "STRATEGY_INELIGIBLE")
            )
            if row["model_available"] != available or row["unavailable_reason"] != reason:
                raise ValueError("Inconsistent membership/model coverage")
        coefficients, intercepts = {}, {}
        for symbol, model in models.items():
            coef = np.asarray(model["coefficients"], dtype=float)
            intercept = float(model["intercept"])
            if coef.shape != (5,) or not np.isfinite(coef).all() or not np.isfinite(intercept):
                raise ValueError("Invalid FF5 coefficients")
            coef.setflags(write=False)
            coefficients[symbol], intercepts[symbol] = coef, intercept
        self.coefficients = MappingProxyType(coefficients)
        self.intercepts = MappingProxyType(intercepts)

    @classmethod
    def load(cls, directory: Path):
        metadata = json.loads((directory / "metadata.json").read_text())
        for filename, key in [
            ("ff5.json", "artifact_sha256"),
            ("membership.json", "membership_sha256"),
        ]:
            if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != metadata[key]:
                raise ValueError("Artifact checksum mismatch")
        for name, version in metadata["library_versions"].items():
            if importlib.metadata.version(name) != version:
                raise RuntimeError(f"Training/serving version mismatch: {name}")
        payload = json.loads((directory / "ff5.json").read_text())
        membership = json.loads((directory / "membership.json").read_text())
        if (
            payload["library_versions"] != metadata["library_versions"]
            or payload["model_version"] != metadata["model_version"]
            or payload["snapshot_date"] != metadata["signal_date"]
        ):
            raise ValueError("Artifact metadata mismatch")
        model = cls(payload, membership)
        if (
            len(model.universe) != metadata["universe_count"]
            or len(model.eligible) != metadata["eligible_count"]
            or len(model.coefficients) != metadata["model_count"]
        ):
            raise ValueError("Artifact coverage count mismatch")
        return model

    def predict(self, symbol, features):
        if symbol not in self.universe:
            raise SymbolUnavailable(symbol, "SYMBOL_OUTSIDE_UNIVERSE")
        if symbol not in self.coefficients:
            reason = "NO_VALID_REGRESSION" if symbol in self.eligible else "STRATEGY_INELIGIBLE"
            raise SymbolUnavailable(symbol, "MODEL_UNAVAILABLE", reason)
        values = np.asarray(features, dtype=float)
        if values.shape != (5,) or not np.isfinite(values).all():
            raise ValueError("Exactly five finite factor features are required")
        prediction = float(self.intercepts[symbol] + self.coefficients[symbol] @ values)
        if not np.isfinite(prediction):
            raise RuntimeError("Nonfinite prediction")
        return {
            "prediction": prediction,
            "model_version": self.model_version,
            "symbol": symbol,
            "prediction_horizon_calendar_days": 90,
        }
