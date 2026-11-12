"""Model choice, using 2013 to 2015 only.

Prices and outcomes are cut at Dec 31 2015 before anything else happens.
Each Ridge refit (Jan 2013, Jan 2014, Jan 2015) learns from 2010-onward rows
whose 90 day outcome was already known, then scores that calendar year.

Whichever rule and basket size has the best 2013-2015 Sharpe gets saved to
results/frozen_choice.json. run_final.py only reads that file.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import data_prep  # noqa: E402
import signal_model as sm  # noqa: E402
from portfolio_sim import StaggeredBook, index_as_nav, summarize_nav  # noqa: E402

TUNE_FROM, TUNE_TO = "2013-01-01", "2015-12-31"
RIDGE_PENALTIES = (1, 10, 100, 1000)
BASKET_SIZES = (20, 30)
OUT_DIR = Path(__file__).parent.resolve() / "results"


def main():
    weekly, closes, levels = data_prep.load_or_build(data_prep.BIG_DB)
    weekly = weekly[weekly.date <= TUNE_TO]
    closes = closes[closes.index <= TUNE_TO]
    view = sm.percentile_view(weekly)
    refits = sm.january_firsts(view, (2013, 2014, 2015))

    contenders = {"momentum": sm.fixed_rule_score(view, "momentum"),
                  "blend": sm.fixed_rule_score(view, "blend")}
    weight_history = []
    for a in RIDGE_PENALTIES:
        contenders[f"ridge_a{a}"], notes = sm.rolling_ridge(view, refits, a, end=TUNE_TO)
        notes["alpha"] = a
        weight_history.append(notes)

    scoreboard = []
    for label, sc in contenders.items():
        sc = sc[sc.date.between(refits[0], pd.Timestamp(TUNE_TO))]
        ic, _ = sm.score_ic(view, sc)
        picks = sm.by_signal_date(sc)
        for size in BASKET_SIZES:
            nav = StaggeredBook(closes, TUNE_FROM, TUNE_TO, basket=size).run(picks)["daily"].nav
            m = summarize_nav(nav)
            scoreboard.append({"score": label, "names": size, "rank_ic": ic, **m.to_dict()})
            print(f"{label:<13}{size:>3}  ic={ic:+.4f}  cagr={m.CAGR:.2%}  sharpe={m.Sharpe:.2f}  "
                  f"mdd={m['Max drawdown']:.2%}", flush=True)

    first_session = closes.index[closes.index >= TUNE_FROM][0]
    spx = summarize_nav(index_as_nav(levels, first_session, TUNE_TO))
    scoreboard.append({"score": "^SP500TR", "names": None, "rank_ic": None, **spx.to_dict()})
    board = pd.DataFrame(scoreboard)

    OUT_DIR.mkdir(exist_ok=True)
    board.to_csv(OUT_DIR / "development.csv", index=False, float_format="%.6f")
    pd.concat(weight_history).to_csv(OUT_DIR / "development_ridge_fits.csv", index=False, float_format="%.6f")

    # rule fixed in advance: best Sharpe wins, CAGR only breaks a tie
    ours = board[board.score != "^SP500TR"]
    winner = ours.sort_values(["Sharpe", "CAGR"], ascending=False).iloc[0]
    frozen = {"score": winner.score, "names": int(winner.names), "chosen_by": "highest 2013-2015 Sharpe",
              "dev_sharpe": float(winner.Sharpe), "dev_cagr": float(winner.CAGR),
              "dev_rank_ic": float(winner.rank_ic)}
    (OUT_DIR / "frozen_choice.json").write_text(json.dumps(frozen, indent=2))
    print(f"\n^SP500TR over 2013-2015: cagr={spx.CAGR:.2%} sharpe={spx.Sharpe:.2f}")
    print("frozen:", frozen)


if __name__ == "__main__":
    main()
