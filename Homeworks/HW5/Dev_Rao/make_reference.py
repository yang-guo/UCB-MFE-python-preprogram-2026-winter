"""Record reference inputs + predictions straight from my HW4 experiment.

Run from the REPO root env (it needs duckdb, which the service does not):

    uv run python Homeworks/HW5/Dev_Rao/make_reference.py

Inputs come from HW4's feature cache (Homeworks/HW4/Dev_Rao/outputs/cache_top2000.duckdb,
table `panel`) and the expected outputs come from HW4's own saved final-test scores
(outputs/final_scores.csv, written by HW4's run_final_test.py). Nothing here recomputes
the score, so the reference is independent of the HW5 serving code.

Writes reference.json (a few cases, incl. rows with missing fundamentals) and
example_request.json (the first case, as a request body).
"""

import json
import math
from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).resolve().parent
HW4 = HERE.parents[1] / "HW4" / "Dev_Rao" / "outputs"
FEATURES = [
    "r_ey", "r_bm", "r_sp", "r_fcfy", "r_gp_a", "r_roe", "r_roa", "r_accruals_a",
    "r_asset_growth", "r_share_iss", "r_mom_12_1", "r_sector_mom_126", "r_ni_chg_a",
    "r_ivol_63", "r_max_ret_21",
]  # fmt: skip

# (signal date, symbol): a complete row, rows with missing fundamentals, a late-sample row
CASES = [
    ("2016-03-31", "KLAC"),  # first test signal; also lot 3 traced by hand in HW4
]


def main():
    con = duckdb.connect(str(HW4 / "cache_top2000.duckdb"), read_only=True)
    scores = pd.read_csv(HW4 / "final_scores.csv", parse_dates=["date"])
    cols = ", ".join(FEATURES)
    # deterministic extra cases: earliest test rows with (a) roe+ey missing, (b) ivol missing,
    # and (c) the last test signal's alphabetically first symbol
    extra = con.execute(
        """
        (SELECT date, symbol FROM panel WHERE date >= '2016-03-31' AND r_roe IS NULL
            AND r_ey IS NULL ORDER BY date, symbol LIMIT 1)
        UNION ALL
        (SELECT date, symbol FROM panel WHERE date >= '2016-03-31' AND r_ivol_63 IS NULL
            ORDER BY date, symbol LIMIT 1)
        UNION ALL
        (SELECT date, symbol FROM panel WHERE date = (SELECT max(date) FROM panel
            WHERE date < '2026-09-30') ORDER BY symbol LIMIT 1)
        """
    ).fetchall()
    cases = CASES + [(str(pd.Timestamp(d).date()), s) for d, s in extra]
    out = []
    for date, symbol in cases:
        row = con.execute(
            f"SELECT {cols} FROM panel WHERE date = ? AND symbol = ?", [date, symbol]
        ).fetchdf()
        assert len(row) == 1, (date, symbol)
        hit = scores.loc[(scores.date == date) & (scores.symbol == symbol), "score"]
        assert len(hit) == 1, (date, symbol)
        feats = {
            k: (None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v))
            for k, v in row.iloc[0].to_dict().items()
        }
        out.append(
            {
                "signal_date": date,
                "symbol": symbol,
                "features": feats,
                "hw4_score": float(hit.iloc[0]),
                "n_missing": sum(v is None for v in feats.values()),
            }
        )
    doc = {
        "source": "HW4 final test (run once, 2026-11-13): features from outputs/cache_top2000.duckdb "
        "table panel; hw4_score from outputs/final_scores.csv written by run_final_test.py",
        "feature_order": FEATURES,
        "cases": out,
    }
    (HERE / "reference.json").write_text(json.dumps(doc, indent=2) + "\n")
    (HERE / "example_request.json").write_text(json.dumps(out[0]["features"], indent=2) + "\n")
    for c in out:
        print(c["signal_date"], c["symbol"], c["hw4_score"], "missing:", c["n_missing"])


if __name__ == "__main__":
    main()
