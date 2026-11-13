"""Apply the DEMONSTRATION Stage A rule (experiments/demo/SELECTION_RULES.md) to the registry.

    python scripts/select_alpha.py [--prefix A]

Rule: highest mean validation rank IC among candidates with IC > 0 in >= 3 of 4
years; a simpler candidate within 0.005 IC of the best wins
(complexity: composite < ridge < hgb).  Uses development results only.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from quantlab.experiment import REGISTRY  # noqa: E402

COMPLEXITY = {"composite": 0, "mean": 0, "ridge": 1, "hgb": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="A")
    args = ap.parse_args()
    reg = pd.read_csv(REGISTRY).drop_duplicates("experiment_id", keep="last")
    reg = reg.loc[reg.name.str.startswith(args.prefix)].copy()
    reg["model_kind"] = reg.model.map(lambda m: json.loads(m)["kind"])
    reg["complexity"] = reg.model_kind.map(COMPLEXITY)
    reg["stable"] = reg.validation_years_positive_IC >= 3
    cols = ["experiment_id", "model_kind", "validation_mean_IC", "validation_years_positive_IC", "stable",
            "walk_forward_dev_IC", "dev_CAGR", "dev_benchmark_CAGR", "dev_Sharpe", "dev_max_drawdown"]
    print(reg.sort_values("validation_mean_IC", ascending=False)[cols].round(4).to_string(index=False))
    ok = reg.loc[reg.stable]
    if ok.empty:
        print("\nNo candidate meets the stability rule (IC > 0 in >= 3 of 4 years).")
        return
    best = ok.validation_mean_IC.max()
    near = ok.loc[ok.validation_mean_IC >= best - 0.005].sort_values(["complexity", "validation_mean_IC"],
                                                                     ascending=[True, False])
    choice = near.iloc[0]
    print(f"\nBest stable IC {best:.4f}; candidates within 0.005: {list(near.experiment_id)}")
    print(f"SELECTED (pre-registered rule): {choice.experiment_id}  -> {choice.output_dir}")


if __name__ == "__main__":
    main()
