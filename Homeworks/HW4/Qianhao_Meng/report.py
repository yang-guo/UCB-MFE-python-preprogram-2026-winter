"""Charts and Markdown summaries for the HW4 runs."""

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

from backtest import HOLDING_DAYS, INITIAL_CAPITAL, performance


def plot_cumulative(navs, path, title):
    """Draw every portfolio's cumulative return since the first close on one chart."""
    ax = (navs / INITIAL_CAPITAL - 1).plot(figsize=(9, 4.5), lw=1.4)
    ax.set(title=title, xlabel="Date", ylabel="Cumulative return")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", alpha=0.25)
    ax.figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(ax.figure)


def markdown_table(frame, index_label):
    """Render a small table of already formatted values."""
    header = f"| {index_label} | " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| --- | " + " | ".join("---:" for _ in frame.columns) + " |"
    rows = [f"| {label} | " + " | ".join(map(str, row)) + " |" for label, row in zip(frame.index, frame.to_numpy())]
    return [header, rule, *rows, ""]


def metrics_table(navs):
    """Return the four required metrics and a formatted table with one row per portfolio."""
    metrics = navs.apply(performance)
    table = pd.DataFrame({"Final value": metrics.loc["final_value"].map("${:,.0f}".format),
                          "CAGR": metrics.loc["cagr"].map("{:.2%}".format),
                          "Maximum drawdown": metrics.loc["max_drawdown"].map("{:.2%}".format),
                          "Sharpe, 2% risk-free": metrics.loc["sharpe"].map("{:.3f}".format)})
    return metrics, table


def run_diagnostics(result):
    """Summarize cash use, costs, and the missing-quote incidents of one run."""
    daily, events = result.daily, result.events.event.value_counts()
    return pd.Series({
        "Average cash share": f"{(daily.cash / daily.nav).mean():.1%}",
        "Fills": f"{len(result.fills):,}",
        "Fees paid": f"${result.fills.fee.sum():,.0f}",
        "Canceled buys, no entry quote": f"{events.get('canceled_buy', 0):,}",
        "Sales delayed by a missing quote": f"{events.get('delayed_sale', 0):,}",
        "Lots with a quote gap over five sessions": f"{events.get('long_quote_gap', 0):,}",
        "Largest value at marks over five sessions old": f"${daily.long_stale_value.max():,.0f}",
        "Largest share of NAV at such marks": f"{(daily.long_stale_value / daily.nav).max():.2%}",
        "Open lots at the end": f"{len(result.open_lots):,}",
    })


def _frozen_rows(result, sessions, breaks):
    vendor = breaks.set_index("symbol")
    rows = []
    for lot in result.open_lots:
        if lot.last_session == len(sessions) - 1:
            continue
        value = lot.units * lot.last_price
        broken = lot.symbol in vendor.index
        rows.append({"symbol": lot.symbol, "entry": lot.entry_date, "last_quote": sessions[lot.last_session],
                     "reason": f"unresolved price break on {vendor.break_date[lot.symbol]:%Y-%m-%d}" if broken
                     else "no quote",
                     "value": value,
                     "vendor_value": lot.units * vendor.vendor_last_close[lot.symbol] if broken else value})
    return pd.DataFrame(rows, columns=["symbol", "entry", "last_quote", "reason", "value", "vendor_value"])


def frozen_holdings(result, sessions, breaks):
    """Describe final-day holdings that are still valued at an older price.

    A lot on an unresolved price break is also valued at the vendor's last
    quote, so the reader sees final NAV under both readings of that move.
    """
    final_nav = result.daily.nav.iloc[-1]
    rows = _frozen_rows(result, sessions, breaks)
    frozen, alternative = rows.value.sum(), rows.vendor_value.sum()
    lines = [f"Holdings carried at an older price on the final day: ${frozen:,.0f} in {len(rows)} lots, "
             f"{frozen / final_nav:.2%} of NAV.", ""]
    if rows.empty:
        return lines
    lines.extend([f"Final NAV is ${final_nav:,.0f} with those lots at their last valid quote, so the four metrics "
                  "depend on that valuation. As a diagnostic scenario only, final NAV would read "
                  f"${final_nav - frozen + alternative:,.0f} with the lots on an unresolved price break valued at the "
                  "vendor's last quote on or before the final day. That quote can be on a different price basis from "
                  "the holding, so it is neither an estimate nor a bound, and it ignores how an earlier sale would "
                  "have changed later trades.", ""])
    table = pd.DataFrame({"Entry": rows.entry.dt.strftime("%Y-%m-%d"),
                          "Last valid quote": rows.last_quote.dt.strftime("%Y-%m-%d"),
                          "Days at that price": (sessions[-1] - rows.last_quote).dt.days, "Reason": rows.reason,
                          "Value": rows.value.map("${:,.0f}".format),
                          "Share of NAV": (rows.value / final_nav).map("{:.2%}".format),
                          "Diagnostic value at vendor's last quote": rows.vendor_value.map("${:,.0f}".format)})
    return lines + markdown_table(table.set_axis(rows.symbol), "Symbol")


def label_coverage(panel, end):
    """Say how many signals with a 90-day outcome due by `end` actually have one."""
    due = panel.loc[(panel.entry_date + pd.Timedelta(days=HOLDING_DAYS)).le(pd.Timestamp(end))]
    return (f"Of {len(due):,} candidate signals whose 90-day outcome was due by {pd.Timestamp(end):%Y-%m-%d}, "
            f"{due.target.notna().mean():.2%} have an observed outcome. The others lost their entry or exit quote, "
            "so they are absent from rank IC and from any fit.")


def reconciliation(result, benchmark):
    """Re-derive the run's accounting and timing identities from its own ledger."""
    daily, fills = result.daily, result.fills
    ledger = INITIAL_CAPITAL + fills.groupby("date").cash_change.sum().reindex(daily.index, fill_value=0).cumsum()
    buys, sells = fills.loc[fills.side.eq("buy")], fills.loc[fills.side.eq("sell")]
    unrealized = sum(lot.units * lot.last_price - lot.cost for lot in result.open_lots)
    pnl_gap = daily.nav.iloc[-1] - INITIAL_CAPITAL - fills.realized_pnl.sum() - unrealized
    on_schedule = (buys.due - buys.entry_date).dt.days.eq(HOLDING_DAYS)
    nav_gap = (daily.nav - daily.cash - daily.holdings).abs().max()
    return pd.Series({
        "First and last session": f"{daily.index[0]:%Y-%m-%d} to {daily.index[-1]:%Y-%m-%d}",
        "Benchmark value at the first close": f"${benchmark.iloc[0]:,.2f}",
        "Lowest cash balance": f"${daily.cash.min():,.2f}",
        "Largest gap between NAV and cash plus holdings": f"${nav_gap:,.6f}",
        "Largest gap between cash and the fill ledger": f"${(daily.cash - ledger).abs().max():,.6f}",
        "Realized plus unrealized PnL minus the change in NAV": f"${abs(pnl_gap):,.6f}",
        "Buys filled after their signal date": f"{int(buys.signal_date.lt(buys.date).sum()):,} of {len(buys):,}",
        f"Buys given a {HOLDING_DAYS}-calendar-day maturity": f"{int(on_schedule.sum()):,} of {len(buys):,}",
        "Sales on or after maturity": f"{int(sells.date.ge(sells.due).sum()):,} of {len(sells):,}",
    }).to_frame("Result")


def write_report(path, title, sections):
    """Write a title followed by (heading, lines) sections."""
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", *body] if heading else body)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
