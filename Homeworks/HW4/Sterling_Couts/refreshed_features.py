"""Same rank-model inputs at arbitrary decision closes, with no future labels."""

import numpy as np
import pandas as pd

from quality_robustness import replace_financials

PRICE_FEATURES = [
    "momentum_21",
    "momentum_63",
    "momentum_252_21",
    "volatility_63",
    "drawdown_252",
    "momentum_126_21",
]


def daily_rank_features(quotes, entry_eligibility, financials, sectors, start, end):
    """Use current-close information for a NEXT-session 90-day forecast.

    Eligibility is already shifted to the entry session and includes the dated
    membership/risk screen. Cross-sectional ranks use that same available pool.
    No realized holding return, future disappearance or future quote enters.
    """
    assert quotes.index.equals(entry_eligibility.index)
    assert quotes.columns.equals(entry_eligibility.columns)
    calendar = quotes.index
    positions = np.flatnonzero(
        (calendar >= pd.Timestamp(start)) & (calendar <= pd.Timestamp(end))
    )
    positions = positions[positions + 1 < len(calendar)]
    entry_rows, symbol_rows = np.where(entry_eligibility.iloc[positions + 1].to_numpy())
    signal_rows = positions[entry_rows]
    p = pd.DataFrame(
        dict(
            date=calendar[signal_rows],
            entry_date=calendar[signal_rows + 1],
            symbol=quotes.columns[symbol_rows],
            eligible=True,
            risk_screen_pass=True,
        )
    )

    def assign(name, matrix):
        p[name] = matrix.to_numpy()[signal_rows, symbol_rows]

    assign("momentum_21", quotes / quotes.shift(21) - 1)
    assign("momentum_63", quotes / quotes.shift(63) - 1)
    assign("momentum_252_21", quotes.shift(21) / quotes.shift(252) - 1)
    assign("momentum_126_21", quotes.shift(21) / quotes.shift(126) - 1)
    assign(
        "volatility_63",
        quotes.pct_change(fill_method=None).rolling(63, min_periods=63).std(ddof=1)
        * np.sqrt(252),
    )
    assign("drawdown_252", quotes / quotes.rolling(252, min_periods=252).max() - 1)
    p = replace_financials(p, financials, sectors, 0)
    for feature in PRICE_FEATURES:
        p["rank_" + feature] = p.groupby("date")[feature].rank(pct=True)
    return p
