"""Launch development-period experiments with Papermill.

    python scripts/run_research.py experiments/demo/plan_stage_a.json [--only name1,name2]

A plan is a JSON list of {"name", "hypothesis", "overrides"}.  Each entry is
executed as its own copy of notebooks/research_experiment.ipynb with the
config passed as a Papermill parameter; the executed notebook, config,
summary, tables and chart land in outputs/experiments/<experiment_id>/ and a
row is appended to outputs/experiments/registry.csv.  Research notebooks can
only load research-mode data, so nothing here can touch the final test.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import papermill as pm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantlab.experiment import EXPERIMENT_ROOT, REGISTRY, config_id, make_config  # noqa: E402


def use_this_interpreter_for_kernels():
    """The 'python3' kernelspec runs whatever `python` is first on PATH; make it this venv."""
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")


def run_plan(plan: list[dict], only=()) -> dict:
    """Execute each plan entry as its own notebook; returns {name: output directory}."""
    use_this_interpreter_for_kernels()
    done = {}
    for entry in plan:
        if only and entry["name"] not in only:
            continue
        cfg = make_config(name=entry["name"], hypothesis=entry.get("hypothesis", ""), **entry.get("overrides", {}))
        out = EXPERIMENT_ROOT / config_id(cfg)
        out.mkdir(parents=True, exist_ok=True)
        print(f"== {config_id(cfg)}", flush=True)
        pm.execute_notebook(str(ROOT / "notebooks" / "research_experiment.ipynb"), str(out / "notebook.ipynb"),
                            parameters={"config_json": json.dumps(cfg)}, kernel_name="python3",
                            cwd=str(ROOT), progress_bar=False)
        done[entry["name"]] = out
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    run_plan(json.loads(Path(args.plan).read_text()), set(filter(None, args.only.split(","))))
    if REGISTRY.is_file():
        reg = pd.read_csv(REGISTRY).drop_duplicates("experiment_id", keep="last")
        cols = ["experiment_id", "validation_mean_IC", "validation_years_positive_IC", "walk_forward_dev_IC",
                "dev_CAGR", "dev_benchmark_CAGR", "dev_Sharpe", "dev_max_drawdown", "audit_checks_passed"]
        print(reg[cols].to_string(index=False))


if __name__ == "__main__":
    main()
