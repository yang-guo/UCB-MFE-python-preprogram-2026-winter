"""Dependency-free markdown tables (pandas.to_markdown needs `tabulate`, not in the course lockfile)."""

import pandas as pd


def md_table(frame, index=True, floatfmt="{:,.4g}") -> str:
    df = frame.to_frame() if isinstance(frame, pd.Series) else frame
    df = df.reset_index() if index else df

    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else floatfmt.format(v)
        if isinstance(v, pd.Timestamp):
            return str(v.date())
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])
