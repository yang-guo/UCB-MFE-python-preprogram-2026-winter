"""Data audit the data guide asks for: implausible universe-ranking inputs,
quarantined rows, and every price repair made by research.clean_prices.
Read-only; writes small tables to results/.

Run:  python audit_data.py
"""

import duckdb
import pandas as pd

import research
from pipeline import RESULTS

# Thresholds for "probably wrong": no real US stock in this period had a
# $100,000+ share price (BRK-A peaked near $800k, so it is checked by name),
# more than 30 billion shares, or a market cap above $6 trillion.
SQL = """
SELECT quarter_start, symbol, universe_rank, selection_close, selection_shares,
       market_cap_proxy / 1e9 AS cap_proxy_billions,
       CASE WHEN selection_close > 100000 AND symbol NOT LIKE 'BRK%' THEN 'price > $100k'
            WHEN selection_shares > 30e9 THEN 'shares > 30B'
            WHEN market_cap_proxy > 6e12 THEN 'cap > $6T' END AS flag
FROM universe_membership
WHERE (selection_close > 100000 AND symbol NOT LIKE 'BRK%') OR selection_shares > 30e9 OR market_cap_proxy > 6e12
ORDER BY quarter_start, universe_rank
"""


def main():
    with duckdb.connect(str(research.FULL_DB), read_only=True) as con:
        flagged = con.execute(SQL).fetchdf()
        zero_adj = con.execute("""SELECT symbol, count(*) n, min(date) AS first_date, max(date) AS last_date FROM prices
                                  WHERE adj_close IS NULL OR adj_close <= 0 GROUP BY 1""").fetchdf()
        off_cal = con.execute("""SELECT symbol, count(*) n FROM prices
                                 WHERE date NOT IN (SELECT date FROM calendar) GROUP BY 1 ORDER BY 2 DESC""").fetchdf()
        top10 = con.execute("""SELECT universe_rank, symbol, selection_close, selection_shares,
                                      market_cap_proxy / 1e9 AS cap_proxy_billions
                               FROM universe_membership WHERE quarter_start = DATE '2016-04-01'
                               ORDER BY universe_rank LIMIT 10""").fetchdf()
    test_flags = flagged[flagged.quarter_start >= "2016-04-01"]

    # Price repairs made by research.clean_prices (every one is listed).
    repairs = research.MarketData().repairs
    repairs.to_csv(RESULTS / "audit_price_repairs.csv", index=False)
    by_kind = repairs.groupby("repair").agg(symbols=("symbol", "nunique"), rows=("rows", "sum"))
    print("\nPrice repairs:\n", by_kind.to_string())
    print("\nUnadjusted splits since 2016:\n",
          repairs[(repairs.repair == "unadjusted_split") & (repairs.date >= "2016-04-01")].to_string(index=False))

    summary = pd.Series({
        "membership rows flagged (all quarters)": len(flagged),
        "distinct flagged symbols": flagged.symbol.nunique(),
        "flagged rows in a top-10 rank": int((flagged.universe_rank <= 10).sum()),
        "flagged rows in test-period quarters": len(test_flags),
        "zero/missing adjusted-price rows quarantined": int(zero_adj.n.sum()),
        "off-calendar price rows quarantined": int(off_cal.n.sum()),
        **{f"price repair: {k} (rows)": int(v) for k, v in by_kind.rows.items()},
    })
    flagged.to_csv(RESULTS / "audit_flagged_ranking_inputs.csv", index=False)
    pd.concat({"zero_adj_close": zero_adj.set_index("symbol").n, "off_calendar": off_cal.set_index("symbol").n},
              axis=1).to_csv(RESULTS / "audit_quarantined_prices.csv")
    summary.to_csv(RESULTS / "audit_summary.csv", header=["count"])
    print(summary.to_string())
    print("\nFlagged symbols (rows):\n", flagged.groupby(["symbol", "flag"]).size().sort_values(ascending=False).head(20).to_string())
    print("\n2016-Q2 top 10 by cap proxy:\n", top10.to_string())


if __name__ == "__main__":
    main()
