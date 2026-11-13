"""Run the one final holdout evaluation through Papermill.

    python scripts/run_final.py [--note "..."]

Executes notebooks/final_evaluation.ipynb into results/final_evaluation.ipynb.
Requires a frozen strategy (scripts/freeze_strategy.py).
"""

import argparse
import os
import sys
from pathlib import Path

import papermill as pm

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default="final evaluation")
    ap.add_argument("--acknowledge-reselection", action="store_true")
    args = ap.parse_args()
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    pm.execute_notebook(str(ROOT / "notebooks" / "final_evaluation.ipynb"),
                        str(ROOT / "results" / "final_evaluation.ipynb"),
                        parameters={"acknowledge_reselection": args.acknowledge_reselection, "note": args.note},
                        kernel_name="python3", cwd=str(ROOT), progress_bar=False)


if __name__ == "__main__":
    main()
