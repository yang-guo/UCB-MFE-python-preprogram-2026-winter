"""Write the research-period data-quality evidence used in the README.

    python scripts/data_quality_report.py

Research mode only (2007-01 .. 2016-03).  Outputs (small, committed):
results/data_quality_research.md, results/integrity_events_research.csv,
results/feature_registry.csv.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantlab import features as F  # noqa: E402
from quantlab.experiment import prepare  # noqa: E402
from quantlab.protocol import Mode  # noqa: E402
from quantlab.report import md_table  # noqa: E402


def main():
    market, ref, audit, panel = prepare(Mode.RESEARCH)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    ev = audit.events.copy()
    ev["quarter"] = ev.date.dt.to_period("Q").dt.start_time
    mem = ref.membership.rename(columns={"quarter_start": "quarter"})[["symbol", "quarter", "universe_rank"]]
    ev = ev.merge(mem, on=["symbol", "quarter"], how="left")
    ev["universe_member_that_quarter"] = ev.universe_rank.notna()
    ev.drop(columns="quarter").to_csv(out / "integrity_events_research.csv", index=False)
    F.feature_table().to_csv(out / "feature_registry.csv", index=False)

    kinds = ev.groupby("kind").agg(events=("symbol", "size"), symbols=("symbol", "nunique"),
                                   in_universe=("universe_member_that_quarter", "sum"))
    members = panel
    screen = pd.Series({
        "member rows (signal date x member)": len(members),
        "no quote at signal close": int((~members.has_quote).sum()),
        "fewer than 57 of 63 daily returns": int((~members.history_ok).sum()),
        "median dollar volume < $1M": int((~members.liquid).sum()),
        "price break in last 6 sessions (cool-off)": int(members.integrity_cooloff.sum()),
        "artifact/unreliable history in last 252 sessions": int(members.integrity_unreliable.sum()),
        "eligible": int(members.eligible.sum()),
    })
    f = ref.fundamentals
    fin = pd.Series({
        "quarterly statements loaded": len(f),
        "filing_date <= fiscal_date (placeholder; imputed +120d)": int((f.filing_date <= f.fiscal_date).sum()),
        "eligible rows using an imputed-availability statement":
            int(members.loc[members.eligible, "availability_imputed"].sum()),
        "eligible rows with no usable statement": int(members.loc[members.eligible, "fiscal_date"].isna().sum()),
    })
    examples = ev.loc[ev.universe_member_that_quarter].sort_values("date").groupby("kind").head(6)[
        ["kind", "symbol", "date", "gross", "ratio", "dv_day_ratio", "universe_rank"]]
    lines = [
        "# Data quality evidence (research period, generated)", "",
        f"Database: {market.provenance}", "",
        "## One-day price breaks (|move| > x1.8) by classification", "", md_table(kinds), "",
        "Examples among universe members:", "", md_table(examples.round(4), index=False), "",
        "## Load-time quarantine (non-positive adj_close, off-calendar rows)", "",
        f"{len(market.load_quarantine)} rows in the research window "
        "(BHRB zero prices are in 2000-2001, before the loaded window).", "",
        "## Eligibility screen (each row can fail several screens)", "", md_table(screen.to_frame("rows")), "",
        "## Financial statements", "", md_table(fin.to_frame("rows")), "",
        "## Extreme values kept (not winsorized): largest eligible 63-day returns", "",
        md_table(members.loc[members.eligible].nlargest(5, "ret_63")[["date", "symbol", "ret_63", "vol_63", "target"]]
                 .round(3), index=False), "",
    ]
    (out / "data_quality_research.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
