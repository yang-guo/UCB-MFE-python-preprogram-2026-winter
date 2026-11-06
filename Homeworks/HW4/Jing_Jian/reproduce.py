"""Reproduce development, final test, and report in ignored scratch outputs.

The committed pre-test freeze and submitted results remain intact. Reproduction
uses the documented execution amendment from the outset; it is not a new search
or an opportunity to retune against known final performance.
"""

import subprocess
import sys
import json
import shutil

import numpy as np

import research


def main():
    scratch = research.HERE / "outputs/reproduction"
    research.RESULTS = scratch / "results"
    research.OUTPUTS = scratch / "full"
    if research.RESULTS.exists():
        raise SystemExit("Reproduction outputs already exist; preserve or move them before a fresh run.")
    research.RESULTS.mkdir(parents=True)
    with (research.RESULTS / "test_results.txt").open("w") as stream:
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", str(research.HERE), "-v"],
                        stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=research.ROOT)
    research.development(research.DATABASE)
    # Preserve the original experiment's correction history alongside this
    # reproducibility run. Its new freeze hashes the already-corrected engine.
    shutil.copyfile(research.HERE / "results/execution_amendment.json",
                    research.RESULTS / "execution_amendment.json")
    research.final(research.DATABASE)
    expected = json.loads((research.HERE / "results/final_summary.json").read_text())
    actual = json.loads((research.RESULTS / "final_summary.json").read_text())
    for portfolio in ["strategy", "benchmark"]:
        for key in ["final_value", "cagr", "max_drawdown", "sharpe"]:
            np.testing.assert_allclose(actual["metrics"][portfolio][key],
                                       expected["metrics"][portfolio][key], rtol=1e-8, atol=1e-10)
    assert actual["strategy"] == expected["strategy"]
    subprocess.run([
        sys.executable, str(research.HERE / "make_report.py"),
        "--results-dir", str(research.RESULTS), "--report-path", str(scratch / "REPORT.md"),
    ], check=True)
    print(f"Reproduction complete: {scratch}")


if __name__ == "__main__":
    main()
