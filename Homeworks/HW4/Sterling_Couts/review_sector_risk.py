"""Describe development sector losses, exposure and pre-entry stock volatility.

Sector labels are the vendor's static snapshot, not certified historical labels.
The volatility statistic is a median across stock/signal observations, not the
volatility of a sector index. No future outcome selects observations; no new
strategy is fit or backtested here. Original data and account results are read-only.
"""

import asyncio
import json
import os
from pathlib import Path
import sys
import textwrap

import duckdb
import nbformat as nbf
from nbclient import NotebookClient
import numpy as np
import pandas as pd

from development_revision import corrected_data, remove_fluidigm_alias
from hw4.data import build_data, checksum
from hw4.settings import DATABASE, PROJECT

OUT = PROJECT / "results/sector_review"
YEARS = [2013, 2014, 2015]


def volatility_summary(panel):
    """Summarize already-causal monthly features by subsequent entry year.

    Use eligibility known at the signal, not target availability or later
    survival. Show the original risk screen separately so it cannot silently
    truncate the distribution used to describe the broader candidate pool.
    """
    sample = panel.loc[
        panel.eligible & panel.entry_date.between("2013-01-01", "2015-12-31")
    ].copy()
    assert sample.date.lt(sample.entry_date).all()
    assert sample.date.max() < pd.Timestamp("2016-01-01")
    sample["year"] = sample.entry_date.dt.year
    sample["quarter"] = sample.entry_date.dt.to_period("Q").astype(str)
    parts = []
    for name, frame in [
        ("before_risk_screen", sample),
        ("after_risk_screen", sample.loc[sample.risk_screen_pass]),
    ]:
        for period in ["year", "quarter"]:
            g = frame.groupby(["sector", period])
            s = (
                g.agg(
                    median_volatility=("volatility_63", "median"),
                    q25_volatility=("volatility_63", lambda x: x.quantile(0.25)),
                    q75_volatility=("volatility_63", lambda x: x.quantile(0.75)),
                    stock_signal_observations=("volatility_63", "count"),
                    distinct_symbols=("symbol", "nunique"),
                )
                .reset_index()
                .rename(columns={period: "period"})
            )
            s["scope"], s["frequency"] = name, period
            s["period"] = s.period.astype(str)
            parts.append(s)
    return pd.concat(parts, ignore_index=True)


def holdings_exposure(fills, history, quotes, sector_map):
    """Reconstruct sector market values, respecting all outstanding lots.

    An entry cap concerns aggregate exposure, not each individual purchase.
    Actual weights may drift above an entry cap; this diagnostic forces no sales.
    """
    f = fills.copy()
    f["signed_units"] = np.where(f.side.eq("BUY"), f.units, -f.units)
    units = f.groupby(["date", "symbol"]).signed_units.sum().unstack(fill_value=0)
    units = units.reindex(history.index, fill_value=0).cumsum()
    marks = quotes[units.columns].ffill(limit=5).reindex(history.index)
    assert not ((units.abs() > 1e-7) & marks.isna()).any().any()
    values = (units * marks).fillna(0)
    assert sector_map.reindex(values.columns).notna().all()
    sector_values = values.T.groupby(sector_map).sum().T
    np.testing.assert_allclose(values.sum(axis=1), history.holdings, rtol=0, atol=1e-6)
    np.testing.assert_allclose(
        sector_values.sum(axis=1) + history.cash, history.nav, rtol=0, atol=1e-6
    )
    return sector_values.div(history.nav, axis=0)


def run():
    source_hash = checksum(DATABASE)
    OUT.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(DATABASE), read_only=True) as con:
        master = con.execute(
            "SELECT symbol,name,sector,industry FROM universe ORDER BY symbol"
        ).df()
    assert not master.symbol.duplicated().any()
    sector_map = master.set_index("symbol").sector
    data = build_data(DATABASE, end="2015-12-31")
    data, _, _ = corrected_data(data)
    data, _ = remove_fluidigm_alias(data)
    vol = volatility_summary(data.panel)
    vol.to_csv(OUT / "candidate_volatility.csv", index=False)
    stocks, sectors, exposure_rows = [], [], []
    for year in YEARS:
        # These saved baseline contributions include fees and terminal open lots.
        p = pd.read_csv(
            PROJECT / f"results/development_restart/{year}_stock_contributions.csv"
        )
        p = p.merge(master, on="symbol", validate="one_to_one")
        assert p.sector.notna().all()
        p["gross_loss"] = (-p.net_pnl).clip(lower=0)
        p["gross_gain"] = p.net_pnl.clip(lower=0)
        p["year"] = year
        stocks.append(p)
        g = p.groupby("sector").agg(
            net_pnl=("net_pnl", "sum"),
            gross_loss=("gross_loss", "sum"),
            gross_gain=("gross_gain", "sum"),
            held_symbols=("symbol", "size"),
        )
        g["year"] = year
        g["return_contribution_pp"] = g.net_pnl / 1_000_000 * 100
        g["share_of_gross_losses"] = g.gross_loss / p.gross_loss.sum()
        sectors.append(g.reset_index())
        folder = PROJECT / f"outputs/development_revision/{year}/D2_alias_removed"
        fills = pd.read_csv(folder / "fills.csv", parse_dates=["date"])
        h = pd.read_csv(folder / "daily_nav.csv", parse_dates=["date"]).set_index(
            "date"
        )
        np.testing.assert_allclose(
            p.net_pnl.sum(), h.nav.iloc[-1] - h.nav.iloc[0], rtol=0, atol=1e-6
        )
        weights = holdings_exposure(fills, h, data.quotes, sector_map)
        weights.to_csv(OUT / f"{year}_daily_sector_weights.csv")
        e = pd.DataFrame(
            {
                "average_weight": weights.iloc[1:].mean(),
                "peak_weight": weights.iloc[1:].max(),
                "peak_date": weights.iloc[1:].idxmax(),
            }
        )
        e["year"] = year
        exposure_rows.append(e.reset_index())
    pd.concat(stocks, ignore_index=True).to_csv(
        OUT / "stock_contributions.csv", index=False
    )
    sector = pd.concat(sectors, ignore_index=True)
    sector.to_csv(OUT / "sector_contributions.csv", index=False)
    exposure = pd.concat(exposure_rows, ignore_index=True)
    exposure.to_csv(OUT / "sector_exposure.csv", index=False)
    p = sector.loc[sector.year.eq(2015)]
    related = p.loc[p.sector.isin(["Energy", "Basic Materials"])]
    verification = {
        "source_sha256": source_hash,
        "source_unchanged": checksum(DATABASE) == source_hash,
        "price_cutoff": "2015-12-31",
        "sector_labels": "Static vendor snapshot; descriptive proxy, not dated historical classification.",
        "volatility_definition": "Median stock-level trailing 63-return sample standard deviation times sqrt(252), across historically eligible monthly signals grouped by next entry year; no future completeness requirement.",
        "accounts_reconciled": 3,
        "new_model_fits": 0,
        "new_backtests": 0,
        "energy_materials_2015_net_pnl": related.net_pnl.sum(),
        "energy_materials_share_of_2015_net_loss": related.net_pnl.sum()
        / p.net_pnl.sum(),
        "energy_materials_share_of_2015_gross_losses": related.gross_loss.sum()
        / p.gross_loss.sum(),
        "all_2015_net_pnl": p.net_pnl.sum(),
    }
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))
    build_notebook()


def build_notebook():
    cells = []

    def md(s):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(s).strip()))

    def code(s):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(s).strip()))

    md("""
    # Sector risk and the separate distribution issue
    **Sterling Couts — development diagnostic, November 14, 2026**

    Question: did commodity-related businesses dominate the 2015 loss, and could
    sector-aware position limits address concentration more precisely than a
    smaller cap on every stock? This is a descriptive review, not a new strategy
    result. It uses only prices through 2015 and preserves existing accounts.

    The [course data guide](../../../Lectures/modeling/README.md) explicitly says
    sector/industry labels are current snapshots without dated histories. They
    can support a clearly qualified exploratory comparison, but must not be
    presented as verified point-in-time labels. Energy and Basic Materials are
    broad commodity-related equity groups, not commodity futures or pure commodity
    exposures; some Industrials also depend on commodity activity.
    """)
    code("""
    import json
    from pathlib import Path
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from IPython.display import display
    OUT = Path.cwd() / 'results/sector_review'
    vol = pd.read_csv(OUT / 'candidate_volatility.csv', dtype={'period': str})
    pnl = pd.read_csv(OUT / 'sector_contributions.csv')
    exposure = pd.read_csv(OUT / 'sector_exposure.csv')
    proof = json.loads((OUT / 'verification.json').read_text())
    assert proof['source_unchanged'] and proof['accounts_reconciled'] == 3
    assert proof['new_model_fits'] == proof['new_backtests'] == 0
    """)
    md("""
    ## 1. Distribution: a price fall need not equal a shareholder loss

    The historical Targacept position is recorded under GYRE in the database.
    [Nasdaq's event notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=ETA2015-133)
    specifies **$0.5692 cash and $1.08 note principal per old share**, with
    ex-distribution date August 20, 2015. The following session implements a
    separate 1-for-7 reverse split and the Catalyst/CBIO name change.

    The economic holding consists of the remaining stock **plus** cash and
    distributed notes. A price-only calculation can omit value moved outside
    the stock. For 100 old shares, the stated entitlement is approximately
    $56.92 cash and $108 note principal, apart from the continuing stock. The
    split changes share count and units; it does not by itself destroy value.

    Our stored adjusted-price move is about -55.8%, but that number does not
    establish the complete shareholder return. First reconcile the historical
    share basis and what the adjusted series already includes. Then use either
    a consistent total-return series or explicit entitlements, avoiding double
    counting. Note principal is not cash available for same-day new purchases.
    The [indenture](https://www.sec.gov/Archives/edgar/data/1124105/000119312515313060/d69291dex41.htm)
    defines the redemption date as five business days after notice and payment
    within five business days after that date; a simulated redemption/settlement
    policy needs to be stated and tested.

    Four trend/volatility-filter accounts held this event; their results remain
    provisional. The working baseline and smaller-cap accounts did not hold it
    then, so it does **not** explain the baseline's -25.89% 2015 return. No receipt
    or sale has been added in this diagnostic.
    """)
    md("""
    ## 2. What actually contributed to the baseline loss?

    Sector PnL adds each symbol's sales proceeds minus purchase costs/fees plus
    terminal unsold value. The sum reconciles to account profit/loss. This is
    accounting attribution, not proof that commodity prices caused each loss.
    Current company names may not be their historical names.

    Energy contributes about -$139,678 and Basic Materials -$30,378 in 2015:
    **65.7% of the $258,915 net account loss**. Together they represent **50.0%
    of gross losses across losing stocks**; gains elsewhere offset part of those
    losses. Industrials contributes another -$81,021. DDD, the largest individual
    loss, is classified as Technology. Thus commodity exposure matters, but the
    loss is not exclusively a commodity story.
    """)
    code("""
    display(pnl.pivot(index='sector', columns='year', values='net_pnl').fillna(0).round(2))
    stocks = pd.read_csv(OUT / 'stock_contributions.csv')
    display(stocks.loc[stocks.year.eq(2015)].nsmallest(12, 'net_pnl')[
        ['symbol','name','sector','industry','net_pnl']])
    """)
    md("""
    ## 3. Volatility: define the statistic before comparing it

    At each monthly signal, calculate each eligible stock's trailing 63-return
    sample standard deviation, annualized by sqrt(252). The table reports the
    median of those stock/signal observations for the subsequent entry year.
    No missing price is filled to calculate those stock returns. A December 2014
    signal for a January 2015 entry belongs to the 2015 decision sample.

    **These are typical individual-stock volatilities, not sector-index
    volatilities.** A diversified sector portfolio depends on weights and
    cross-stock correlations as well. Annual medians also hide within-year
    changes, so quarterly values are included. The main table is before the
    strategy's history/80% volatility screen; its screened counterpart is saved
    separately to make the conditioning explicit. The median limits sensitivity
    to a few huge source errors but does not certify the source.
    """)
    code("""
    annual = vol.loc[vol.frequency.eq('year') & vol.scope.eq('before_risk_screen')]
    table = annual.pivot(index='sector', columns='period', values='median_volatility')
    display(table.map('{:.1%}'.format))
    quarterly = vol.loc[vol.frequency.eq('quarter') & vol.scope.eq('before_risk_screen')
                        & vol.sector.isin(['Energy','Basic Materials'])]
    display(quarterly.pivot(index='period', columns='sector', values='median_volatility').map('{:.1%}'.format))
    display(annual.loc[annual.sector.isin(['Energy','Basic Materials']),
                       ['sector','period','stock_signal_observations','distinct_symbols']])
    """)
    md("""
    Energy median volatility is 29.2%, 27.9%, and 41.3% in 2013–2015; Basic
    Materials is 28.3%, 26.1%, and 31.4%. Energy rises to 32.8% in the fourth-quarter
    2014 entry sample and 46.9% in the first-quarter 2015 sample. Those trailing
    risk measures were available at each corresponding decision; the complete
    2015 annual median was not. Healthcare provides a useful control: its median
    2015 volatility is similarly high, so volatility alone does not identify
    the sector that will lose money.
    """)
    md("""
    ## 4. Concentration is different from individual-stock volatility

    Reconstruct all open lots each day, mark them consistently with the account,
    and aggregate by sector. These weights plus the cash fraction sum to 100%.
    Mean weights exclude the initial all-cash bookkeeping row. Peaks are actual
    observed weights, not necessarily weights at the instant of an entry.

    Energy's average portfolio weight rises from **2.9% in 2013** and **4.0% in
    2014** to **20.6% in 2015**; its 2015 peak reaches **42.7%**. A 5% symbol cap
    does not prevent many correlated positions accumulating in one sector. The
    learned contrarian ranking plausibly contributed to buying declining energy
    names, but this attribution alone does not prove that mechanism caused losses.
    """)
    code("""
    display(exposure.loc[exposure.sector.isin(['Energy','Basic Materials','Industrials'])].round(4))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
    for sector in ['Energy','Basic Materials','Industrials','Healthcare']:
        axes[0].plot([2013,2014,2015], table.loc[sector, ['2013','2014','2015']], marker='o', label=sector)
    axes[0].set(title='Median eligible-stock trailing volatility', xlabel='Entry year', ylabel='Annualized volatility')
    axes[0].set_xticks([2013,2014,2015])
    e = exposure.loc[exposure.sector.eq('Energy')].set_index('year')
    e[['average_weight','peak_weight']].rename(columns={'average_weight':'Average weight','peak_weight':'Peak weight'}).plot.bar(ax=axes[1], rot=0)
    axes[1].set(title='Baseline exposure to Energy', xlabel='Account year', ylabel='Fraction of total NAV')
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('Descriptive review using static vendor sector labels')
    fig.savefig(OUT / 'sector_risk.png', dpi=150)
    plt.show()
    """)
    md("""
    ## 5. Controlled experiments to consider next

    **Allocation first, features second.** A smaller per-stock cap only within
    Energy/Basic Materials isolates the user's targeted-sizing hypothesis, but
    many small positions can still accumulate. A separate aggregate sector cap
    directly controls that common concentration. For example, a predeclared 20%
    limit on *new-entry* aggregate exposure to any sector keeps the 5% stock cap
    and limits similar concentration in Healthcare or Technology too. The 20%
    value is a simple proposed design, not an optimized or proven threshold.
    Existing 90-day lots would not be sold early; market drift can exceed the
    entry limit. Leave blocked budget in cash in the first experiment rather
    than simultaneously adding a replacement-stock selection rule.

    Hold predictions, costs, calendar, maturity and other limits fixed. Compare
    with the baseline and universal 2.5% cap, across all three annual accounts and
    the separate continuous account. State the static-sector approximation and
    review material historical classifications before calling it point-in-time.
    No sector allocation variant has been run in this diagnostic.

    Then test one feature at a time: stock momentum minus the same-date sector
    median; sector median recent momentum; or trailing sector risk. Each peer
    set must use membership/eligibility known at the signal. A sector interaction
    lets the model learn whether a signal means something different across
    sectors; independently fitting eleven models may be unstable in this short
    development sample. These are hypotheses, not reasons to encode '2015 Energy
    loses' as a feature. Full-year future volatility must never drive an entry.

    The decision requiring investor judgment is whether the objective favors
    sector-specific limits or a uniform sector concentration budget. Neither
    allocation control nor a new feature guarantees beating the benchmark.
    """)
    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "name": "python3",
                "display_name": "HW4 Python 3.12",
                "language": "python",
            }
        },
    )
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    NotebookClient(
        nb,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT)}},
    ).execute()
    nbf.validate(nb)
    nbf.write(nb, PROJECT / "04_sector_risk_review.ipynb")
    print(f"Sector notebook executed: {len(cells)} cells.")


if __name__ == "__main__":
    run()
