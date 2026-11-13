"""HW4 signal study: does a feature (or model score) rank 90-day returns? Sizing-free.

For each monthly signal date (first session of each month, matching the strategy)
inside development, using eligible names with completed labels only:
    IC_t     = Spearman rank correlation(signal, 90-day target)
    spread_t = mean target of the top 10% by signal - mean target of all names,
               with targets winsorized per date at 1%/99% (bad-print labels)
Summary per signal: mean IC, hit rate (share of dates with IC > 0), mean spread,
t = mean / sd x sqrt(n), and t_adj = t / sqrt(3).

Why t_adj: 90-day labels on monthly dates overlap ~3 months, so neighbouring
ICs are correlated and the effective sample is ~n/3. Treat t_adj >= 2 as
"some evidence", not proof. Every signal tried is logged (multiple testing).
"""

import numpy as np
import pandas as pd

import hw4_model as hm

TOP_FRACTION = 0.10
OVERLAP = 3  # 90-day labels / ~30-day spacing


def monthly_signal_rows(panel, start=hm.DEV_START, end=hm.DEV_END):
    """Eligible rows with completed labels on the first session of each month."""
    p = panel.loc[panel.eligible & panel.target.notna() & panel.date.between(start, end)]
    sessions = pd.Series(np.sort(p.date.unique()))
    firsts = sessions.groupby(sessions.dt.to_period("M")).min()
    return p.loc[p.date.isin(firsts)]


def winsorize_by_date(rows, col="target", lo=0.01, hi=0.99):
    """Clip each date's values to that date's 1%/99% quantiles (evaluation only).

    Needed because the source has absurd labels (e.g. AAM +14,270,000% in 90 days
    from bad prints) that dominate any mean. Uses only same-date values.
    """
    g = rows.groupby("date")[col]
    return rows[col].clip(g.transform("quantile", lo), g.transform("quantile", hi))


def per_date_stats(rows, signal):
    """IC (raw target; ranks are outlier-proof) and top-decile spread (winsorized target)."""
    def one(g):
        g = g[[signal, "target", "target_w"]].dropna()
        if len(g) < 20 or g[signal].nunique() < 2:
            return pd.Series({"ic": np.nan, "spread": np.nan, "n": len(g)})
        k = max(1, int(round(TOP_FRACTION * len(g))))
        return pd.Series({"ic": g[signal].corr(g.target, method="spearman"),
                          "spread": g.nlargest(k, signal).target_w.mean() - g.target_w.mean(),
                          "n": len(g)})
    return rows.groupby("date").apply(one, include_groups=False)


def summarize_signal(stats):
    ic = stats.ic.dropna()
    t = ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic)) if len(ic) > 2 else np.nan
    return {"mean_IC": ic.mean(), "hit_rate": (ic > 0).mean(), "mean_top10_spread": stats.spread.mean(),
            "t": t, "t_adj": t / np.sqrt(OVERLAP), "dates": len(ic), "avg_names": stats.n.mean()}


def signal_table(panel, signals):
    rows = monthly_signal_rows(panel).copy()
    rows["target_w"] = winsorize_by_date(rows)
    out = {s: summarize_signal(per_date_stats(rows, s)) for s in signals}
    return pd.DataFrame(out).T.sort_values("t_adj", ascending=False)


if __name__ == "__main__":
    import sys
    import hw4_build_cache as hc
    import hw4_features as hf
    name = sys.argv[1] if len(sys.argv) > 1 else "dev"
    signals = hf.RAW_NUMERIC + hf.MODEL_FINANCIAL_4A + hf.PRICE_4B + hf.VALUE_METRICS
    panel = pd.read_parquet(hc.cache_path(name),
                            columns=["date", "symbol", "eligible", "target"] + signals)
    table = signal_table(panel, signals)
    out = hc.OUT / f"signal_table_{name}.csv"
    table.to_csv(out)
    pd.set_option("display.width", 160)
    print(table.round(4).to_string())
    print("saved", out)
