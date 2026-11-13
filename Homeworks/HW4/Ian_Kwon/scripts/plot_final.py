"""Supplementary final-test chart from saved outputs (no data access, no new evaluation).

    python scripts/plot_final.py   (after scripts/run_final.py)

results/cumulative_return_with_accounting_sensitivity.png: primary strategy
(the config's delisting policy), the other delisting policy as a sensitivity,
the benchmark, and the share of NAV sitting in frozen lots.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import json  # noqa: E402

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
h = pd.read_csv(ROOT / "outputs/final/daily_nav.csv", parse_dates=["date"], index_col="date")
s = pd.read_csv(ROOT / "outputs/final_alt_delisting_nav.csv", parse_dates=["date"], index_col="date")
policy = json.loads((ROOT / "results/final_summary.json").read_text())["delisting_policy"]
other = "freeze" if policy == "settle_at_maturity" else "settle_at_maturity"
fig, (ax, bx) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
for nav, label, kw in [(h.nav, f"strategy (primary; delisting policy: {policy})", {"lw": 1.6}),
                       (s.nav, f"strategy (sensitivity; delisting policy: {other})", {"lw": 1.2, "ls": ":"}),
                       (h.benchmark_nav, "S&P 500 total return ($1M at first test close)", {"lw": 1.2, "ls": "--",
                                                                                             "color": "black"})]:
    (nav / nav.iloc[0] - 1).plot(ax=ax, label=label, **kw)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set(title="Final test, Apr 2016 - Sep 2026", ylabel="Cumulative return")
ax.legend(fontsize=8)
(h.frozen_value / h.nav).plot(ax=bx, color="tab:red", lw=1)
bx.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
bx.set(ylabel="NAV in frozen lots\n(primary)", xlabel="")
fig.tight_layout()
fig.savefig(ROOT / "results/cumulative_return_with_accounting_sensitivity.png", dpi=130)
print("saved")
