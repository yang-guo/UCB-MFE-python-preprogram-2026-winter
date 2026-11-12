"""The one out-of-sample run: Apr 1 2016 through Sep 30 2026.

Whatever results/frozen_choice.json says is what gets traded, no edits.
Saves the metric table, chart, daily NAV, every fill, every lot and the lots
still open at the end into results/.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.resolve()))
import data_prep  # noqa: E402
import signal_model as sm  # noqa: E402
from portfolio_sim import StaggeredBook, index_as_nav, summarize_nav  # noqa: E402

OOS_FROM, OOS_TO = "2016-04-01", "2026-09-30"
OPENING_SIGNAL = "2016-03-31"   # Thursday close before the first test day
OUT_DIR = Path(__file__).parent.resolve() / "results"


def draw(strategy_nav, spx_nav, label, where):
    fig, (growth_ax, dd_ax) = plt.subplots(2, 1, sharex=True, figsize=(10, 7), height_ratios=[3, 1])
    for path, name in ((strategy_nav, label), (spx_nav, "^SP500TR")):
        growth_ax.plot(path.index, path / path.iloc[0] - 1, label=name)
        dd_ax.plot(path.index, path / path.cummax() - 1, label=name)
    growth_ax.set_title("Apr 1, 2016 to Sep 30, 2026, $1,000,000 start")
    growth_ax.set_ylabel("Cumulative return")
    growth_ax.legend(loc="upper left")
    dd_ax.set_ylabel("Drawdown")
    for ax in (growth_ax, dd_ax):
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(where, dpi=110)


def main():
    frozen = json.loads((OUT_DIR / "frozen_choice.json").read_text())
    weekly, closes, levels = data_prep.load_or_build(data_prep.BIG_DB)
    closes = closes[closes.index <= OOS_TO]

    in_test = weekly[weekly.date.between(pd.Timestamp(OPENING_SIGNAL), pd.Timestamp(OOS_TO))]
    if frozen["score"] in ("momentum", "blend"):
        scores = sm.fixed_rule_score(sm.percentile_view(in_test), frozen["score"])
    else:
        # only reached if development had picked a Ridge model
        alpha = float(frozen["score"].removeprefix("ridge_a"))
        view = sm.percentile_view(weekly[weekly.date <= OOS_TO])
        refits = [pd.Timestamp(OPENING_SIGNAL)] + sm.january_firsts(view, range(2017, 2027))
        scores, notes = sm.rolling_ridge(view, refits, alpha, end=OOS_TO)
        notes.to_csv(OUT_DIR / "final_ridge_fits.csv", index=False)

    run = StaggeredBook(closes, OOS_FROM, OOS_TO, basket=frozen["names"]).run(sm.by_signal_date(scores))
    daily, fills, lots, still_held = run["daily"], run["fills"], run["lots"], run["open_lots"]
    spx_nav = index_as_nav(levels, daily.index[0], OOS_TO)

    # stress case: anything with no print in the last week before the end is worth nothing
    ghost_value = float(still_held.loc[still_held.stale, "value"].sum()) if len(still_held) else 0.0
    worst_case = daily.nav.copy()
    worst_case.iloc[-1] -= ghost_value
    table = pd.concat({"Strategy": summarize_nav(daily.nav),
                       "^SP500TR": summarize_nav(spx_nav),
                       "Strategy, stale lots at $0": summarize_nav(worst_case)}, axis=1)
    table.to_csv(OUT_DIR / "metrics.csv", float_format="%.6f")
    daily.to_csv(OUT_DIR / "daily_nav.csv", float_format="%.4f")
    fills.to_csv(OUT_DIR / "fills.csv", index=False, float_format="%.6f")
    lots.drop(columns="outlay").to_csv(OUT_DIR / "lots.csv", index=False, float_format="%.6f")
    if len(still_held):
        still_held.to_csv(OUT_DIR / "open_positions.csv", index=False, float_format="%.6f")
    draw(daily.nav, spx_nav, f"Strategy ({frozen['score']}, {frozen['names']} names)",
         OUT_DIR / "cumulative_returns.png")

    print(table.to_string(float_format=lambda v: f"{v:,.4f}"))
    n_stale = int(still_held.stale.sum()) if len(still_held) else 0
    print(f"\n{len(still_held)} lots open at the end, {n_stale} of them stale, stale value ${ghost_value:,.0f}")
    print(f"{len(lots)} lots bought, {int((fills.side == 'no_price').sum())} picks skipped for no price")


if __name__ == "__main__":
    main()
