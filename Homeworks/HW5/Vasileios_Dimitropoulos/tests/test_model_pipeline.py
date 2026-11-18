"""Unit tests for the reusable model logic (no HTTP)."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from model_pipeline import (
    ARTIFACT_PATH,
    INPUT_COLUMNS,
    METADATA_PATH,
    RANK_FEATURES,
    SECTORS,
    RankFeatureDesign,
    load_bundle,
    to_model_frame,
)


@pytest.fixture
def example_row():
    return json.loads((ARTIFACT_PATH.parents[1] / "example_request.json").read_text())


def test_design_matches_hw4_ridgeranker_design_by_hand(example_row):
    """Hand-check of HW4's _design: centred ranks, missing flags, sector one-hot."""
    row = dict(example_row, r_roe=None)
    design = RankFeatureDesign().fit(to_model_frame(row))
    x = design.transform(to_model_frame(row))[0]

    assert x.shape == (len(RANK_FEATURES) * 2 + len(SECTORS),) == (68,)
    i_roe = RANK_FEATURES.index("r_roe")
    assert x[i_roe] == 0.0  # missing rank -> 0.5 -> centred 0
    assert x[len(RANK_FEATURES) + i_roe] == 1.0  # its missing flag
    assert x[0] == pytest.approx(row["r_mom_5"] - 0.5)
    assert x[len(RANK_FEATURES) : 2 * len(RANK_FEATURES)].sum() == 1.0  # only r_roe missing
    sector_part = x[2 * len(RANK_FEATURES) :]
    assert sector_part.sum() == 1.0 and sector_part[SECTORS.index("Technology")] == 1.0


def test_design_rejects_reordered_columns(example_row):
    frame = to_model_frame(example_row)
    design = RankFeatureDesign().fit(frame)
    with pytest.raises(ValueError, match="Expected columns"):
        design.transform(frame[list(reversed(INPUT_COLUMNS))])


def test_design_rejects_unknown_sector(example_row):
    frame = to_model_frame(example_row)
    design = RankFeatureDesign().fit(frame)
    frame["sector"] = "Crypto"
    with pytest.raises(ValueError, match="sector"):
        design.transform(frame)


def test_prediction_does_not_depend_on_other_rows(example_row):
    """No batch statistics: one stock's score is the same alone or inside a batch."""
    pipeline = load_bundle().pipeline
    alone = pipeline.predict(to_model_frame(example_row))[0]
    other = dict(example_row, r_mom_5=0.01, sector="Energy")
    batch = pd.concat([to_model_frame(example_row), to_model_frame(other)], ignore_index=True)
    assert pipeline.predict(batch)[0] == pytest.approx(alone, abs=1e-12)


def test_load_bundle_refuses_a_modified_artifact(tmp_path):
    artifact = tmp_path / "model.joblib"
    shutil.copy(ARTIFACT_PATH, artifact)
    with artifact.open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(RuntimeError, match="hash"):
        load_bundle(artifact, METADATA_PATH)


def test_load_bundle_refuses_a_different_sklearn_version(tmp_path):
    metadata = json.loads(METADATA_PATH.read_text())
    metadata["library_versions"]["scikit-learn"] = "0.0.1"
    path = tmp_path / "model_metadata.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="scikit-learn"):
        load_bundle(ARTIFACT_PATH, path)


def test_metadata_records_exact_reproduction_of_hw4():
    checks = json.loads(METADATA_PATH.read_text())["export_checks"]
    assert checks["rows_checked"] > 20_000
    assert np.isclose(checks["max_abs_score_diff_vs_recorded_hw4_predictions"], 0, atol=1e-9)
