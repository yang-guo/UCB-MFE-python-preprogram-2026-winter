"""Explain saved 2013-2015 downside without fitting or reading later prices.

Stock PnL is cash received minus cash spent plus terminal open-lot value.
This independently reconciles the stored ledger; a loss is not a data-error
classification. The maximum permitted price date is explicit and enforced.
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from hw4.settings import DATABASE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results/development_restart"
CUTOFF = pd.Timestamp("2015-12-31")


def assert_development_dates(dates):
    values = pd.to_datetime(dates)
    if values.isna().any() or (values > CUTOFF).any():
        raise ValueError(
            "This diagnostic accepts development observations through 2015 only."
        )


def stock_contributions(fills, lots, terminal_prices):
    """Reconcile all entry/exit fees and include open positions without selling."""
    cash_flows = fills.groupby("symbol").cash_change.sum()
    marks = lots.symbol.map(terminal_prices)
    if not np.isfinite(marks).all() or (marks <= 0).any():
        raise ValueError("A terminal lot lacks a positive observed mark; investigate.")
    terminal_values = (lots.units * marks).groupby(lots.symbol).sum()
    return cash_flows.add(terminal_values, fill_value=0).rename("net_pnl").sort_values()


def run():
    protocol = json.loads((OUT / "protocol.json").read_text())
    assert protocol["data_cutoff"] == str(CUTOFF.date())
    rows = []
    for year in protocol["validation_years"]:
        folder = ROOT / f"outputs/development_momentum/long_momentum/{year}"
        fills = pd.read_csv(folder / "fills.csv")
        lots = pd.read_csv(folder / "open_lots.csv")
        nav = pd.read_csv(folder / "daily_nav.csv", index_col="date", parse_dates=True)
        assert_development_dates(nav.index)
        assert_development_dates(fills.date)
        # Mandatory maturity sales have no new prediction signal date.
        assert_development_dates(fills.loc[fills.side.eq("BUY"), "signal_date"])
        # Maturity may fall beyond year-end; that does not authorize later prices.
        assert_development_dates(lots.entry_date)
        end = nav.index[-1]
        with duckdb.connect(str(DATABASE), read_only=True) as con:
            quotes = (
                con.execute(
                    "SELECT symbol,adj_close FROM prices WHERE date=? AND date<=?",
                    [end.date(), CUTOFF.date()],
                )
                .df()
                .set_index("symbol")
                .adj_close
            )
        contribution = stock_contributions(fills, lots, quotes)
        capital = nav.nav.iloc[0]
        pnl = nav.nav.iloc[-1] - capital
        np.testing.assert_allclose(contribution.sum(), pnl, rtol=0, atol=1e-6)
        contribution.to_csv(OUT / f"{year}_stock_contributions.csv")
        monthly = (
            nav[["nav", "benchmark_nav"]]
            .resample("ME")
            .last()
            .pct_change(fill_method=None)
        )
        monthly = monthly.loc[monthly.index.year == year]
        monthly.to_csv(OUT / f"{year}_monthly_returns.csv")
        closed = fills.loc[fills.side.eq("SELL")]
        gross_loss = -contribution.loc[contribution.lt(0)].sum()
        worst_five_loss = -contribution.loc[contribution.lt(0)].head(5).sum()
        rows.append(
            {
                "year": year,
                "total_return": pnl / capital,
                "benchmark_return": nav.benchmark_nav.iloc[-1]
                / nav.benchmark_nav.iloc[0]
                - 1,
                "max_drawdown": (nav.nav / nav.nav.cummax() - 1).min(),
                "worst_month": str(monthly.nav.idxmin().date()),
                "worst_month_return": monthly.nav.min(),
                "net_pnl": pnl,
                "fees": fills.fee.sum(),
                "closed_lot_win_fraction": closed.realized_pnl.gt(0).mean(),
                "held_symbols": len(contribution),
                "net_losing_symbols": int(contribution.lt(0).sum()),
                "five_worst_share_of_gross_losses": worst_five_loss / gross_loss
                if gross_loss
                else 0,
                "pnl_reconciliation_error": contribution.sum() - pnl,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "baseline_downside.csv", index=False)
    evidence = {
        "price_cutoff": str(CUTOFF.date()),
        "accounts_reconciled": len(rows),
        "new_model_fits": 0,
        "new_backtests": 0,
        "largest_pnl_reconciliation_error": float(
            table.pnl_reconciliation_error.abs().max()
        ),
        "interpretation": "Descriptive development loss attribution; neither causal explanation nor validation of every price.",
    }
    (OUT / "verification.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    run()
