"""Step 4: freeze every choice BEFORE any 2016+ outcome is used.

    uv run python freeze_config.py --model <name> --top-n 25 --cadence weekly --universe all

Writes outputs/config.json and a committed copy frozen_config.json. The
final test refuses to run if the frozen file, the code-level settings or the
database fingerprint do not match.
"""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn

from hw4 import config, data
from hw4 import features as F
from hw4 import model as M


def model_spec(name):
    """Turn a validation model name back into its spec."""
    if name == "composite":
        return {"kind": "composite", "target": "none", "themes": M.COMPOSITE_THEMES}
    parts = name.split("_")
    if parts[0] == "ridge":
        return {"kind": "ridge", "alpha": float(parts[1][1:]), "target": parts[2]}
    if parts[0] == "hgb":
        return {"kind": "hgb", "max_iter": int(parts[1][2:]), "max_leaf_nodes": int(parts[2][2:]),
                "min_samples_leaf": int(parts[3][2:]), "learning_rate": 0.05, "l2_regularization": 1.0,
                "target": parts[4]}
    raise ValueError(f"cannot freeze {name}; ensembles need their members listed explicitly")


def main(a):
    sel = json.loads((config.OUTPUT_DIR / "cv_selection_top2000.json").read_text())
    if a.model.startswith("ens"):
        members = a.members.split(",")
        spec = {"kind": "ensemble", "members": [dict(model_spec(m), name=m) for m in members]}
    else:
        spec = dict(model_spec(a.model), name=a.model)
    cfg = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_in_validation": sel,
        "model": spec,
        "features": {"numeric": F.MODEL_NUMERIC, "categorical": F.MODEL_CATEGORICAL,
                     "raw_before_ranking": F.RAW_FEATURES, "sector_relative": F.SECTOR_REL_FEATURES,
                     "regime": F.REGIME_FEATURES,
                     "normalisation": "same-date percentile rank among eligible stocks; NaN kept"},
        "training": {"train_start": config.TRAIN_START, "purge": "label_end < fit_date",
                     "refit_rule": a.refit, "sample_every": a.sample_every,
                     "preprocessing": "fitted inside sklearn Pipeline on training rows only"},
        "signals": {"cadence_of_features": "last market session of each week (+2016-03-31)",
                    "entry": "next market session close", "holding_days": config.HOLDING_DAYS},
        "eligibility": {"dated_universe_member": True, "min_history_sessions": config.MIN_HISTORY_SESSIONS,
                        "min_median_dollar_volume_63d": config.MIN_MEDIAN_DOLLAR_VOLUME,
                        "raw_price_screen": "none (close is split-adjusted to today's basis)"},
        "strategy": {"top_n": a.top_n, "cadence": a.cadence,
                     "cohort_fraction": 1 / 13 if a.cadence == "weekly" else 1 / 3,
                     "universe": a.universe, "liquid_top": 1000, "name_cap": 0.05, "sector_cap": 0.30,
                     "weighting": "equal within cohort; capped names pass budget to next rank",
                     "cohort_budget": "min(cash + value of lots due next session, NAV * cohort_fraction)"},
        "backtest": {"initial_capital": config.INITIAL_CAPITAL, "fee_bps_per_side": config.FEE_BPS,
                     "fee_sensitivities_bps": [0, 25], "cash_interest": 0.0,
                     "max_stale_sessions": config.MAX_STALE_SESSIONS,
                     "missing_price_policy": "carry last close <= 5 sessions (flagged); longer gap => exit at "
                                             "last observed close (base) / 0% recovery (worst case)",
                     "first_signal": config.TEST_FIRST_SIGNAL, "first_entry": config.TEST_START,
                     "end": config.TEST_END, "annual_rf_for_sharpe": config.ANNUAL_RF},
        "database": data.database_fingerprint(config.DATABASES["top2000"]),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
                     "duckdb": duckdb.__version__, "matplotlib": matplotlib.__version__},
    }
    assert cfg["database"]["matches_manifest"], "database differs from manifest.json"
    text = json.dumps(cfg, indent=2, default=str)
    cfg_hash = hashlib.sha256(text.encode()).hexdigest()
    (config.OUTPUT_DIR / "config.json").write_text(text)
    (config.HW_DIR / "frozen_config.json").write_text(text)
    print(text)
    print(f"\nfrozen_config.json sha256 = {cfg_hash}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--members", default="")
    ap.add_argument("--top-n", type=int, required=True)
    ap.add_argument("--cadence", choices=["weekly", "monthly"], required=True)
    ap.add_argument("--universe", choices=["all", "liquid"], required=True)
    ap.add_argument("--refit", default="quarterly: first signal of each quarter, expanding window from 2010")
    ap.add_argument("--sample-every", type=int, default=1)
    main(ap.parse_args())
