"""Build executed robustness and decision reports from the saved research evidence."""

import asyncio
import json
import os
import sys
import textwrap

import nbformat as nbf
from nbclient import NotebookClient

from hw4.data import checksum
from hw4.settings import PROJECT
from research_session import save_json

FAMILIES = [
    "predictor_foundations",
    "signal_research",
    "quality_robustness",
    "rank_controls",
    "rank_model_stability",
    "allocation_stress",
    "reinvestment_stress",
    "refreshed_signals",
]
OUT = PROJECT / "results/research_summary"


def execute(name, cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = dict(
        display_name="HW4 research (.venv)", language="python", name="python3"
    )
    nb.metadata.language_info = dict(name="python", version=sys.version.split()[0])
    os.environ["PYTHONPATH"] = str(PROJECT)
    os.environ["PATH"] = (
        str(PROJECT / ".venv/Scripts") + os.pathsep + os.environ["PATH"]
    )
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Prefer the project interpreter and assert its actual path inside each
    # executed notebook; a silently selected global kernel is not accepted.
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT)}},
    )
    client.execute()
    nbf.validate(nb)
    nbf.write(nb, PROJECT / name)
    errors = [
        o
        for c in nb.cells
        if c.cell_type == "code"
        for o in c.get("outputs", [])
        if o.output_type == "error"
    ]
    assert not errors
    return dict(
        notebook=name,
        cells=len(nb.cells),
        executed_code_cells=sum(
            c.cell_type == "code" and c.execution_count is not None for c in nb.cells
        ),
        cell_errors=len(errors),
    )


def run():
    for family in FAMILIES:
        protocol = json.loads(
            (PROJECT / "results" / family / "protocol.json").read_text()
        )
        proof = json.loads(
            (PROJECT / "results" / family / "verification.json").read_text()
        )
        assert (
            proof["post2015_runs"] == 0
            and proof["source_and_original_artifact_unchanged"]
        )
        for name, value in protocol["code_sha256"].items():
            assert checksum(PROJECT / name) == value, name
        for name, value in protocol["source_sha256"].items():
            assert checksum(name) == value, name
        for name, value in protocol.get("forecast_sha256", {}).items():
            assert checksum(name) == value, name
    common = """
    from pathlib import Path
    import sys
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from IPython.display import display, Markdown
    ROOT=Path.cwd()
    assert Path(sys.executable).resolve()==(ROOT/'.venv/Scripts/python.exe').resolve()
    OUT=ROOT/'results/research_summary'
    families=['predictor_foundations','signal_research','quality_robustness','rank_controls',
              'rank_model_stability','allocation_stress','reinvestment_stress','refreshed_signals']
    rows=pd.concat([pd.read_csv(ROOT/'results'/f/'experiments.csv').assign(family=f) for f in families],ignore_index=True)
    rows['specific_open_data_issue']=rows.specific_open_data_issue.fillna('')
    rows['unflagged']=rows.status.eq('complete') & rows.specific_open_data_issue.eq('')
    def account(family,name):
        return rows.loc[rows.family.eq(family)&rows.variant.eq(name)].iloc[0]
    benchmark=account('predictor_foundations','F0_original').benchmark_return
    print(f'{len(rows)} saved accounts in this research session; replays are not independent observations.')
    """
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))

    def code(text):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))

    md("""# Rank-model robustness and unsuccessful alternatives
    **Sterling Couts — November 15, 2026**

    This notebook preserves the follow-up tests after the rank-based Ridge
    model became promising. Each stage was declared before its own results,
    but later stages were motivated by earlier development outcomes. None is
    an independent final-period test. The original code, model and database
    remain unchanged; all new accounts end December 31, 2015.

    Read `13_predictor_foundations.ipynb` and `14_signal_research.ipynb` first
    for the penalty/update checks and initial lecture-first signal comparison.
    The concise decision report is `16_research_session_summary.ipynb`.
    """)
    code(common)
    md("""## Separate the questions

    U accounts change the S13 predictor while retaining monthly top-20 entries,
    5% stock caps and the original one-third-NAV cash budget. A accounts reuse
    its exact forecasts and change allocation/costs. V accounts investigate
    the fixed-fit model; W accounts reuse its forecasts. T and J are subsequent
    portfolio stress checks. Each label's precise parent and change appear below.

    Alpha remains 1000 for timing, schedule and allocation comparisons. Changed
    feature/target sets repeat the same pre-2013 grid. Explicit alpha100 and
    alpha10000 runs are sensitivity checks, not a replacement penalty selected
    from later profits. The 30-day delay follows the Lecture 7 exercise; it
    supplements the ordinary filing-plus-one-day gate and does not repair
    restated financial-statement vintages.
    """)
    code("""
    for family in ['quality_robustness','rank_model_stability','allocation_stress','reinvestment_stress','refreshed_signals']:
        display(Markdown('### '+family.replace('_',' ').title()))
        protocol=json.loads((ROOT/'results'/family/'protocol.json').read_text())
        display(pd.DataFrame(protocol['variants'])[['name','parent','change']])
        data=rows.loc[rows.family.eq(family)]
        display(data[['variant','total_return','return_2013','return_2014','return_2015',
                      'max_drawdown','Sharpe','average_cash_fraction','incident_rows']].round(5))
    """)
    md("""## Fixed versus updated relationships

    Refreshing a forecast is different from refitting its coefficients. The
    fixed model still consumes each month's new price history and newly
    available financials. Quarterly or annual refitting also changes the
    coefficients, imputation and scaling after more targets mature. More
    refitting is not automatically better when relationships are noisy.

    A rank model is linear in **percentiles**, not in raw price ratios. The
    transformation limits extreme magnitudes without making large moves equal
    at a fixed winsorization threshold. Adding profitability and cash-flow
    ranks improves the initial price-rank account, but some financial variants
    fail to beat the index. Conditional coefficient signs are not causal effects.
    """)
    code("""
    wanted=['U00_replay','U01_fixed','U02_annual','U03_alpha100','U04_alpha10000',
            'U05_roa_only','U07_quality_only','U08_without_six_month','U09_without_volatility',
            'U11_filing_delay30','V01_delay30','V02_alpha100','V03_alpha10000','V04_without_six_month']
    shown=rows.loc[rows.variant.isin(wanted)].sort_values('total_return')
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    ax.barh(shown.variant,shown.total_return,color='steelblue')
    ax.axvline(benchmark,color='black',ls='--',label='S&P 500')
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='x',alpha=.2);ax.legend()
    ax.set(title='Model robustness: original allocation held fixed',xlabel='2013–2015 total return')
    fig.savefig(OUT/'model_robustness.png',dpi=150);plt.show()
    display(json.loads((ROOT/'results/quality_robustness/financial_checks.json').read_text()))
    """)
    md("""## Costs, deployment, sector limits and reinvestment

    Full available-cash budgeting removes the extra one-third-NAV spending
    limit. The budget is still bounded by yesterday's cash, name/sector room,
    available valid quotes and the top-ranked list. It does not guarantee full
    investment or allow borrowing. It changes cohort timing as well as cash
    exposure. Entry caps do not force immature sales when weights drift.

    The original next-session reinvestment comparison uses actual net proceeds
    from yesterday's mature sales, latest monthly rankings and fresh eligibility.
    An overlapping monthly order takes precedence. Every new purchase is a
    separate 90-day lot; repurchases incur costs and do not reset old lots.

    Zero-cost outcomes are diagnostics only. The 25-bps-per-side scenarios are
    stress tests, not an estimate of actual spreads/market impact or capacity.

    The final Q comparison refreshes all eight inputs/ranks at the preceding
    close for each entry, keeping the same coefficients and purchase triggers.
    This aligns the forecast's 90-day horizon with each new lot more strictly
    than carrying a weeks-old monthly forecast. It reproduces every original
    monthly candidate and forecast, plus the entire carried-ranking account,
    before changing the refresh rule. Refreshed forecasts alone cannot trigger
    extra daily buying. Four fresh-signal tests cover this separation, default
    engine parity, supported date collections and future-price independence.
    """)
    code("""
    show=rows.loc[rows.family.isin(['allocation_stress','reinvestment_stress','refreshed_signals'])]
    fig,ax=plt.subplots(figsize=(11,7),layout='constrained')
    label=show.family.str.replace('_stress','')+'/'+show.variant
    ax.barh(label,show.total_return,color=np.where(show.unflagged,'steelblue','darkorange'))
    ax.axvline(benchmark,color='black',ls='--',label='S&P 500')
    ax.xaxis.set_major_formatter(PercentFormatter(1));ax.grid(axis='x',alpha=.2);ax.legend()
    ax.set(title='Same fixed model: portfolio stress comparisons',xlabel='2013–2015 total return')
    fig.savefig(OUT/'portfolio_stress.png',dpi=150);plt.show()
    """)
    md("""## Random-ranking controls and failures

    Ten seeds were fixed before results. Each uses the identical screened
    opportunity set and original allocator, with random ordering instead of
    return predictions. They are diagnostics, never candidate strategies.
    Shared market history, small sample size and the preceding model search
    rule out interpreting this exercise as a formal significance test.

    Incident-marked gains remain provisional, and the blocked low-volatility
    account has no full-period return. A passing ledger check establishes
    accounting consistency, not the economic truth of every vendor quote.
    """)
    code("""
    random=rows.loc[rows.family.eq('rank_controls')]
    display(random[['variant','status','total_return','max_drawdown','incident_rows','specific_open_data_issue']])
    display(rows.loc[~rows.unflagged,['family','variant','status','total_return','specific_open_data_issue']])
    complete=rows.loc[rows.status.eq('complete')]
    np.testing.assert_allclose((1+complete[['return_2013','return_2014','return_2015']]).prod(axis=1),1+complete.total_return,atol=1e-10)
    assert rows.loc[rows.status.ne('complete'),'total_return'].isna().all()
    assert complete.max_nav_error.max()<1e-6
    print('Every completed account compounds and reconciles; incomplete accounts retain missing headline metrics.')
    """)
    checks = [execute("15_rank_model_robustness.ipynb", cells)]
    cells = []
    md("""# Research findings and the next model decision
    **Sterling Couts — November 15, 2026**

    We now have return-prediction candidates that beat the supplied S&P 500
    benchmark over the **2013–2015 development account**. The main improvements
    are percentile-based price/quality inputs and a more deliberate cash
    policy, rather than greater tree complexity or a highly tuned sector grid.

    This does **not** establish HW4's full-period bonus outcome. A revised model
    has not been evaluated over April 2016–September 2026 in this research pass.
    Earlier baseline evaluation/history remains documented. Further selection
    must not be described as a previously untouched test.
    """)
    code(common)
    md("""## Matched development comparisons

    All rows below start with $1m and the same benchmark close, carry holdings
    and cash across years, charge actual modeled trade costs and retain all
    maturity/eligibility constraints. Differences are stated in the labels.
    Lower drawdown is preferable; Sharpe uses the assignment's 2% annual RF.
    """)
    code("""
    chosen=[('predictor_foundations','F0_original','Original one-feature model'),
            ('quality_robustness','U00_replay','Rank + quality, quarterly fit'),
            ('quality_robustness','U01_fixed','Rank + quality, fixed fit'),
            ('rank_model_stability','W04_budget_full','Fixed + full cash budget'),
            ('rank_model_stability','W10_full_sector20','Fixed + full cash + 20% sector'),
            ('allocation_stress','T05_reinvestment','Carried-monthly-rank reinvestment (comparison)'),
            ('refreshed_signals','Q01_fresh','Same portfolio + fresh-forecast reinvestment')]
    selected=pd.DataFrame([account(f,n).to_dict()|{'label':label} for f,n,label in chosen])
    assert selected.unflagged.all()
    benchmark_metrics=pd.read_csv(ROOT/'results/predictor_foundations/F0_original_performance.csv',index_col=0).loc['benchmark']
    annual_benchmark=pd.read_csv(ROOT/'results/predictor_foundations/F0_original_calendar_years.csv')
    table=selected.set_index('label')[['final_value','total_return','return_2013','return_2014','return_2015','max_drawdown','Sharpe','average_cash_fraction']].copy()
    bench={k:benchmark_metrics.get(k,np.nan) for k in table.columns}
    for r in annual_benchmark.itertuples(): bench[f'return_{r.year}']=r.benchmark_return
    bench['average_cash_fraction']=0.
    table.loc['S&P 500']=bench
    table.to_csv(OUT/'decision_table.csv')
    shown=table.copy()
    shown['final_value']=shown.final_value.map('${:,.0f}'.format)
    for col in ['total_return','return_2013','return_2014','return_2015','max_drawdown','average_cash_fraction']:
        shown[col]=shown[col].map('{:.2%}'.format)
    shown['Sharpe']=shown.Sharpe.map('{:.3f}'.format)
    display(shown)
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for family,name,label in [chosen[i] for i in [0,2,4,6]]:
        h=pd.read_csv(ROOT/'results'/family/f'{name}_nav.csv',parse_dates=['date']).set_index('date')
        ax.plot(h.index,h.nav/1e6,label=label)
    ax.plot(h.index,h.benchmark_nav/1e6,color='black',ls='--',label='S&P 500')
    ax.set(title='Continuous development accounts',ylabel='Account value ($ millions)');ax.grid(alpha=.2)
    ax.legend(fontsize=8,loc='upper left');fig.savefig(OUT/'decision_accounts.png',dpi=150);plt.show()
    """)
    md("""## What the model knows and learns

    Eight inputs are same-date eligible-stock percentiles: one-month momentum,
    three-month momentum, 12-to-1-month momentum, 63-return volatility,
    distance below the 252-session high, 6-to-1-month momentum, operating
    income/assets and free cash flow/assets. A 90th percentile means high
    relative to current eligible peers; whether that is desirable is learned.

    Ridge alpha=1000 was selected using chronological 2011/2012 folds with
    fully completed outcomes before the initial end-2012 fit. Every imputation,
    scaling and training-target clipping step is fitted within the training
    data. No price-feature clipping is needed for the percentile inputs.
    Returns actually earned by the account are never clipped.

    For the fixed policy, model coefficients stay fixed during evaluation;
    monthly inputs and predictions change as new information arrives. The
    quarterly and annual alternatives were tested explicitly. Financials obey
    filing + one calendar day, exclude nonpositive filing lags, and retain the
    course stale-statement convention. An extra 30-day delay is a sensitivity,
    not a newly discovered publication date or a return-selected correction.
    """)
    md("""## Why 2015 improved

    The original negative one-feature slope systematically favored prior
    losers. The new model learns a combination of horizon-dependent price and
    quality information. The attribution below explains realized outcomes;
    it does not establish a causal economic model or justify excluding losers
    after seeing their returns. Dollar sector contributions include actual fees.
    Static supplied sectors are not certified historical classifications.
    """)
    code("""
    attribution=pd.read_csv(OUT/'account_diagnostics.csv')
    display(attribution[['variant','distinct_symbols','largest_contributor','largest_share_of_net_gain',
                         'mean_cash','max_sector_weight','best_held_daily_move','worst_held_daily_move']].round(5))
    yearly=pd.read_csv(OUT/'sector_year_contributions.csv')
    shown=yearly.loc[yearly.year.eq(2015)&yearly.variant.isin(['F0_original','W10_full_sector20','Q01_fresh'])]
    display(shown.pivot(index='sector',columns='variant',values='net_pnl').round(0))
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    shown.pivot(index='sector',columns='variant',values='net_pnl').fillna(0).div(1000).plot.barh(ax=ax)
    ax.axvline(0,color='black',lw=.8);ax.grid(axis='x',alpha=.2)
    ax.set(title='2015 dollar PnL by supplied sector',xlabel='Contribution ($ thousands)')
    fig.savefig(OUT/'sector_2015_contributions.png',dpi=150);plt.show()
    """)
    md("""## Strengths and remaining uncertainty

    The fixed model survives penalty sensitivity, delayed filings, higher
    costs and removal of cash-flow input. Sector limits reduce concentrated
    exposure. Reinvesting known maturity proceeds increases capital deployment
    and return, with a larger drawdown than the monthly sector-limited version.
    Profit is spread across many positions rather than one extreme winner.

    Broad rank correlation remains weak. Top-20 outcomes are more encouraging,
    but they have overlapping 90-day horizons, unavailable future labels and
    substantial selection history. The strategy differs from the S&P in size,
    sector exposures and cash; index outperformance is not factor-adjusted
    statistical alpha. Higher-cost scenarios are not a capacity study.

    No held-price incident was triggered in the primary candidates; this does
    not certify all source records. Revised statement vintages, issuer aliases,
    static sectors and future missing-price events remain limitations. A new
    final-period portfolio can encounter events the earlier portfolio did not.
    """)
    code("""
    display(rows.loc[rows.family.isin(['allocation_stress','reinvestment_stress','refreshed_signals']),
        ['family','variant','total_return','return_2015','max_drawdown','Sharpe','incident_rows']].round(5))
    top=pd.read_csv(OUT/'top20_outcomes.csv')
    # Keep the coverage visible: this summary excludes months without all 20
    # completed labels; those months are still present in the saved diagnostic.
    complete_top=top.loc[top.top20_known.eq(20)]
    display(complete_top.groupby('variant').agg(months=('date','size'),
        top20_return=('top20_mean_return','mean'),universe_return=('universe_mean_return','mean')))
    """)
    md("""## Recommendation and the decision still needed

    Stop expanding this development search. The two defensible candidates are
    the fixed rank/quality model with 5% stock and 20% sector entry limits,
    either with monthly purchases alone or with next-session maturity-cash
    reinvestment using freshly dated predictions. Both use the full available-
    cash budget, alpha1000 and the original filing convention. Refreshed
    reinvestment returns 87.96%, with 17.61% drawdown and 1.21 Sharpe. Monthly
    purchases return 83.68%, with 14.98% drawdown and 1.28 Sharpe, and beat the
    benchmark in each of the three years. Monthly is the recommended balanced
    choice; fresh reinvestment provides higher total return with more risk and
    about three times as many purchases. The 103.61% carried-ranking result
    remains a comparison, rather than replacing the stricter forecast timing.

    Do not select the extra filing delay or alternative penalty solely because
    its sensitivity result happens to be higher. Before a revised full-period
    comparison, Sterling should choose the trading schedule and accept its
    risk/complexity trade-off. Freeze all features, alpha, cash/sector rules,
    missing-price/event treatment and reporting criteria together.

    Then refit on all eligible 2010–2015 training signals whose 90-day labels
    are complete at the pre-April-2016 fit date. Late unfinished labels must
    stay excluded. Fixed evaluation coefficients do not mean discarding the
    2013–2015 training history in that final refit. Apply the same frozen policy
    through the required September-2026 cutoff and reconcile every new event;
    do not assume the existing supplements cover future holdings.
    """)
    md("""## Interview practice and provenance

    Explain: (1) why a rank transform changes a linear model; (2) why changing
    alpha alone cannot reorder a complete one-feature regression; (3) why
    new monthly predictions are possible with fixed coefficients; (4) why an
    overlapping 90-day target requires completed-label date splits; and
    (5) why cash deployment can improve a backtest without improving forecasts.

    Course sources: Lectures 7/8 for features, chronological validation,
    availability tests and fitting schedules; Lectures 2/4/5/9 for data,
    independent tests and reproducibility. Additional hypotheses and failures
    are identified in reports 14/15. Codex implemented and tested this work at
    Sterling's request; Sterling should review the economic choices, source
    limitations and explanation before submission. No commit or submission is
    performed by this research session.
    """)
    code("""
    verification=json.loads((OUT/'verification.json').read_text())
    assert verification['daily_pnl_equals_nav_change']
    assert verification['closed_plus_open_pnl_equals_nav_gain']
    assert verification['sectors_plus_cash_equal_nav']
    assert verification['source_and_original_artifact_unchanged']
    assert verification['post2015_runs']==0
    print('Independent profit attribution, sector valuation and preserved-source checks passed.')
    """)
    checks.append(execute("16_research_session_summary.ipynb", cells))
    save_json(
        OUT / "report_verification.json",
        dict(
            notebooks=checks, source_hashes_match=True, new_strategy_runs_from_report=0
        ),
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    run()
