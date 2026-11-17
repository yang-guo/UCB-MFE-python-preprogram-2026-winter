"""Check that the tests fail when the service is wrong. Not used by the service or the tests.

Each entry below is one deliberate bug: a single text replacement in ``app.py`` or
``inference.py``. The script copies the submission to a temporary folder, applies one bug, runs
the test suite there, and expects at least one test to fail. The real files are never touched.

    python mutation_check.py          # about 20 seconds; exit code 1 if a bug goes unnoticed
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOT_COPIED = shutil.ignore_patterns(".venv", "__pycache__", ".pytest_cache", ".ruff_cache")
# Stop at the first failure (-x) and list failed tests (-rf) without tracebacks.
PYTEST = [sys.executable, "-m", "pytest", "-q", "-x", "--tb=no", "-rf"]

# (what goes wrong, file, text to find, replacement)
BUGS = [
    (
        "features sent to the model in reverse order",
        "inference.py",
        "for name in model.features]",
        "for name in reversed(model.features)]",
    ),
    (
        "missing rank scored as 0.5 instead of 0.0",
        "inference.py",
        "np.nan if rank is None else rank",
        "0.5 if rank is None else rank",
    ),
    (
        "row built in float64 instead of float32",
        "inference.py",
        "dtype=np.float32)",
        "dtype=np.float64)",
    ),
    (
        "model logic accepts a non-finite prediction",
        "inference.py",
        "if not math.isfinite(prediction):",
        "if False:",
    ),
    (
        "artifact checksum not verified",
        "inference.py",
        'if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata["artifact_sha256"]:',
        "if False:",
    ),
    (
        "scikit-learn version not verified",
        "inference.py",
        "if sklearn.__version__ != trained_with:",
        "if False:",
    ),
    (
        "artifact unpickled again on every request",
        "inference.py",
        "    prediction = float(model.pipeline.predict(row)[0])",
        "    joblib.load(MODEL_PATH)\n    prediction = float(model.pipeline.predict(row)[0])",
    ),
    (
        "a model is fitted on every request",
        "inference.py",
        "    prediction = float(model.pipeline.predict(row)[0])",
        "    type(model.pipeline[-1])().fit(np.zeros((2, 24)), np.zeros(2))\n"
        "    prediction = float(model.pipeline.predict(row)[0])",
    ),
    (
        "model loaded inside the request handler",
        "app.py",
        "    model = request.app.state.model\n    request_id",
        "    model = load_model()\n    request_id",
    ),
    (
        "strings and booleans converted to numbers",
        "app.py",
        'extra="forbid", strict=True,',
        'extra="forbid",',
    ),
    (
        "extra fields accepted",
        "app.py",
        'extra="forbid", strict=True,',
        "strict=True,",
    ),
    (
        "no bounds on the ranks",
        "app.py",
        "Field(ge=-0.5, le=0.5)]",
        "Field()]",
    ),
    (
        "null accepted for a feature HW4 always had",
        "app.py",
        "mom_21: Rank = Field",
        "mom_21: Rank | None = Field",
    ),
    (
        "default 422 handler restored (NaN in a body becomes a 500)",
        "app.py",
        "@app.exception_handler(RequestValidationError)\n",
        "",
    ),
    (
        "inference error answered with a 200 and prediction 0.0",
        "app.py",
        "        raise HTTPException(\n"
        "            status_code=500,\n"
        '            detail="Prediction unavailable",\n'
        '            headers={"X-Request-ID": request_id},\n'
        "        ) from error\n",
        "        result = OutputData(prediction=0.0, model_version=model.model_version)\n",
    ),
    (
        "exception text sent to the client",
        "app.py",
        'detail="Prediction unavailable"',
        "detail=str(error)",
    ),
    (
        "request schema not compared with the model at startup",
        "app.py",
        "if tuple(InputData.model_fields) != model.features:",
        "if False:",
    ),
    (
        "wrong model version in the response",
        "app.py",
        "model_version=model.model_version)\n    except",
        'model_version="unknown")\n    except',
    ),
]


def first_failing_test(folder: Path) -> str | None:
    """Run the suite in ``folder``; return the first failing test, or None if all passed."""
    run = subprocess.run(PYTEST, cwd=folder, capture_output=True, text=True)
    if run.returncode == 0:
        return None
    failed = [line for line in run.stdout.splitlines() if line.startswith("FAILED")]
    return failed[0].split("::")[1].split(" - ")[0] if failed else "an error before any test"


def main() -> int:
    unnoticed = 0
    for label, file, old, new in BUGS:
        with tempfile.TemporaryDirectory() as scratch:
            copy = Path(scratch) / "service"
            shutil.copytree(HERE, copy, ignore=NOT_COPIED)
            source = (copy / file).read_text()
            if source.count(old) != 1:
                raise SystemExit(f"'{label}': text to replace not found once in {file}")
            (copy / file).write_text(source.replace(old, new))
            failing_test = first_failing_test(copy)
        unnoticed += failing_test is None
        print(f"{'caught' if failing_test else 'MISSED':7s} {label}\n        by {failing_test}")
    print(f"\n{len(BUGS) - unnoticed} of {len(BUGS)} bugs caught")
    return 1 if unnoticed else 0


if __name__ == "__main__":
    sys.exit(main())
