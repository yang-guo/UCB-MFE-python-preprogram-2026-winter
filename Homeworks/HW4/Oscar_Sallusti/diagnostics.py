"""Post-test diagnostics (run AFTER final_test.py; nothing here changes the strategy).

Separates the effect of equal-weighting the 500 largest stocks from the effect
of the model's stock selection, using the same backtester and rules.

    .venv/bin/python Homeworks/HW4/Oscar_Sallusti/diagnostics.py
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
from backtester import Backtester, performance_metrics  # noqa: E402
from features import build_panel, load_fundamental_events, load_market  # noqa: E402

DB = HERE.parents[2] / "Lectures/modeling/data/equities_top2000.duckdb"

m = load_market(DB, end=C.TEST_END)
panel = build_panel(m, load_fundamental_events(DB, end=C.TEST_END), C.FIRST_SIGNAL, C.TEST_END)
prices = m["A"].loc[:C.TEST_END]
from strategy import TopScoreStrategy  # noqa: E402

# Same tranche rules; every eligible top-500 stock gets an equal budget (no model).
ew = panel[panel.universe_rank <= C.MAX_UNIVERSE_RANK].assign(score=0.0)
strat = TopScoreStrategy(ew, top_n=C.MAX_UNIVERSE_RANK, tranches=C.TRANCHES, max_weight=C.MAX_WEIGHT,
                         max_sector_share=1.0, max_universe_rank=C.MAX_UNIVERSE_RANK)
res = Backtester(prices, initial_cash=C.INITIAL_CASH, fee_bps=C.FEE_BPS).run(strat, C.FIRST_SIGNAL, C.TEST_END)
met = performance_metrics(res.history.nav)
met["final_value_if_stale_positions_worth_zero"] = res.history.nav_ex_stale.iloc[-1]
met.rename("Equal-weight top 500 (no model)").to_csv(HERE / "results/diagnostic_equal_weight.csv")
print(met.to_string())
