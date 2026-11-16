"""Build the concise submission and learning report from verified saved results.

No fitting, strategy selection, or backtest is performed here. Historical
working documents are preserved locally before their entry points are updated.
"""

import json
import textwrap

import nbformat as nbf
import pandas as pd

from build_rank_research_report import execute
from hw4.data import checksum
from hw4.settings import PROJECT
from research_session import save_json


OUT = PROJECT / "results/fixed_anchor_final"


METHOD = """
## Features, model and chronology

The model predicts the simple adjusted-price return from the **next session's
close** to the first market close on or after **90 calendar days** from entry.
Each monthly signal uses only information available by that signal close.
Quarterly historical membership determines the entry universe, including inactive
stocks; leaving the universe does not cause an early sale.

Eight inputs are percentile ranks within the eligible cross-section on the same
date: 21- and 63-session momentum; 252-to-21 and 126-to-21 momentum; 63-session
annualized volatility; drawdown from the trailing 252-session high; quarterly
operating income/assets; and quarterly free cash flow/assets. Ranks reduce the
influence of incomparable numerical scales, but discard magnitude information.

The selected estimator is **Ridge, alpha 1000**, with training-only median
imputation, missing indicators and standardization. The training response is
winsorized at its training 1st/99th percentiles; actual portfolio returns and
prices are never winsorized. This estimates a robust response, not an unrestricted
conditional expected return. Penalties and feature choices were compared during
development. Complexity did not reliably improve performance.

The retained fit is dated **December 31, 2012**, using **51,633** observations:
signals from January 29, 2010 through August 31, 2012, with the last included
outcome complete on December 3, 2012. The 2013–2015 period supports development
comparisons. Earlier prices supply lookbacks. The assignment's January–March
2016 buffer is not used to refit this selected estimator. Coefficients and learned
preprocessing stay fixed; features and predictions refresh monthly. Retraining
is optional under HW4.

Financials are available on **filing date + 1 calendar day**. Missing filing dates
and filing dates on/before fiscal period end are excluded, rather than repaired
with guessed knowledge dates. Statements more than 550 days old are missing.
No future backfill is used. The source's restated snapshot is not a complete
point-in-time archive of original filings.

## Trading and accounting

Start with $1 million, long only, no borrowing. At each monthly signal, rank
eligible candidates and allocate available cash to at most 20 names, subject to
**5% recorded-symbol and 20% supplied-sector entry limits**. Decisions use
prior-close cash and exposure; execution is at the next session's close, with
**5 basis points per side**. Entry limits constrain planned purchases; subsequent
price movements can take weights above those limits. The model does not trim
winners early. Each lot is sold at its own 90-day maturity or next session if the
market is closed. Maturity proceeds wait for the next monthly purchase date.
Cash earns zero interest. Fractional adjusted-price units are permitted.

Candidates require observable prices and complete recent return history. The
declared risk screen uses trailing volatility at most 80%, 252 positive-volume
sessions, and no trailing daily price ratio outside [0.5, 2]. These are causal
entry restrictions, not declarations that excluded observations are errors.
Unknown future exit prices never remove a prediction candidate.

Use supplied adjusted prices consistently, without adding dividends again.
Missing marks may carry forward for at most five sessions; missing execution
prices never become fabricated fills. Verified mandatory mergers or ticker
changes are booked separately from discretionary trades. Replacement exposures
retain original maturity; cash consideration becomes a receivable, spendable
after two market sessions (a simulation convention, not a claimed broker date).
The source database is read-only. Cited repairs and limited successor quotes are
separate inputs under `research_events/`.

## Interpretation and limitations

The reported comparison uses fixed model coefficients and trading rules.
Development winners may reflect selection noise.
Static vendor sectors, incomplete historical issuer mappings, restated financial
vintages and imperfect adjusted prices limit the economic interpretation. The
5% cap is per recorded ticker, not a certified issuer-wide cap. Execution assumes
closing fills and proportional costs without market impact, taxes or financing.
Accounting reconciliation validates the calculation; it does not certify every
vendor observation or establish future profitability.
"""


LEARNING = """
# Learning guide: explaining the selected strategy

**Research question.** Can a simple relationship between relative price behavior,
risk and profitability rank stocks usefully over a 90-day holding period?
Separate three objects: the predictor produces scores, the portfolio rule turns
scores into orders, and the ledger records the economic result.

1. **A dated row is an information set.** A fiscal quarter ending in March is
   not known in March if the company files in May. Use filing+1, and ensure the
   next-session entry follows the signal. Training also needs completed labels.
2. **A rank is relative.** A momentum percentile of 0.9 means approximately the
   top tenth of peers on that date, not a forecast of 90% return. It can change
   when peer values change even if that stock's own momentum stays constant.
3. **Ridge regularizes correlated predictors.** It penalizes squared coefficient
   size. Standardization makes the penalty comparable across inputs; alpha1000
   was a development choice. More features or a nonlinear model need evidence,
   not an assumption that complexity must help.
4. **Fit and predict are different operations.** Fit learns medians, scales,
   coefficients and target bounds. Predict applies those saved quantities to new
   observations. This project refreshes inputs monthly without refitting.
5. **Target and trade must agree.** For signal t, entry e is the next session;
   maturity m is e+90 calendar days, moved to the next market session. The target
   is P(m)/P(e)-1, not a return beginning at t. Overlapping lots remain separate.
6. **A ledger is a conservation check.** NAV = cash + marked holdings + cash
   receivables. Buys reduce cash by notional plus cost; sells add notional minus
   cost. Corporate conversions require unit consistency and cannot create wealth
   just because a ticker or price scale changes.
7. **Performance is not prediction accuracy.** Rankings, idle cash, concentration,
   costs and the holding rule determine realized returns. Annual fresh-cash tests
   are diagnostics and must not be compounded into a continuous portfolio result.
8. **A test proves its stated condition.** Hand cases check arithmetic and timing;
   an independent ledger checks the full account; source comparisons check some
   data. None proves that every source row is true or that future returns follow
   the backtest. Keep unresolved evidence visible and avoid tuning on the final
   comparison.

Metric conventions: daily simple return r(t)=NAV(t)/NAV(t-1)-1; CAGR uses elapsed
calendar years from first evaluation close to final close; drawdown is
NAV(t)/running_max(NAV)-1. Sharpe = sqrt(252) × (mean(r)-rf_daily)/sample_sd(r),
where rf_daily=(1.02)^(1/252)-1. An initial pre-trade row includes entry costs in
strategy performance; benchmark purchase at the first test close has no initial
market return.

To review independently, explain why a missing future label is retained at
prediction time, trace one buy to its maturity, and reconcile a day's cash and
holdings. Then explain why the 5%/20% entry rules permit later weight drift.
These are the most useful interview exercises for this implementation.

Next research would need new chronological evidence: rolling stability of the
rank coefficients, broader issuer-aware data quality checks, and execution/cost
stress. Do not select another strategy from this reported final comparison.
"""


def selected_research():
    # Each row has a documented scope. These are selected development examples,
    # not an exhaustive search log or all single-change comparisons to one model.
    picks = [
        (
            "predictor_foundations",
            "F0_original",
            "One momentum input",
            "Original simple linear baseline",
        ),
        (
            "signal_research",
            "S12_price_ranks",
            "Price ranks",
            "Multiple price/risk ranks, quarterly updates",
        ),
        (
            "signal_research",
            "S13_quality_ranks",
            "+ profitability",
            "Add operating ROA and cash-flow/assets to price ranks",
        ),
        (
            "rank_model_stability",
            "V00_fixed_replay",
            "Fixed eight-rank Ridge",
            "Include six-month momentum; fixed pre-2013 fit",
        ),
        (
            "rank_model_stability",
            "W04_budget_full",
            "Full available cash",
            "Increase monthly deployment budget; no sector cap",
        ),
        (
            "rank_model_stability",
            "W10_full_sector20",
            "+ 20% sector limit",
            "Keep 5% stock cap; limit sector entry exposure",
        ),
        (
            "refreshed_signals",
            "Q01_fresh",
            "Fresh reinvestment signals",
            "Extra next-session entries with newly computed forecasts",
        ),
        (
            "fitting_policy",
            "F01_annual",
            "Annual replacement",
            "Same monthly allocation; replace the fitted model yearly",
        ),
        (
            "fitting_policy",
            "F02_half_blend",
            "50/50 forecast average",
            "Same monthly allocation; average original and annual fit",
        ),
        (
            "fitting_policy",
            "F00_fixed",
            "Selected fixed monthly model",
            "Retain original fit and 5%/20% monthly allocation",
        ),
    ]
    rows = []
    for family, variant, label, change in picks:
        data = pd.read_csv(PROJECT / "results" / family / "experiments.csv")
        row = data.loc[
            data.variant.eq(variant) & data.period.eq("continuous_2013_2015")
        ]
        assert len(row) == 1 and row.status.iloc[0] == "complete"
        row = row.iloc[0]
        rows.append(
            dict(
                experiment=label,
                change=change,
                total_return=row.total_return,
                max_drawdown=row.max_drawdown,
                Sharpe=row.Sharpe,
                evidence=f"results/{family}/experiments.csv",
                variant=variant,
            )
        )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "selected_development_comparisons.csv", index=False)
    return result


def preserve_and_write(name, content):
    path = PROJECT / name
    archive = PROJECT / "outputs/document_archive" / ("before_selected_final_" + name)
    if path.exists() and not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(path.read_bytes())
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def run():
    status = json.loads((OUT / "latest_status.json").read_text())
    checks = json.loads((OUT / "verification.json").read_text())
    assert status["status"] == "complete" and status["verified"]
    assert checks["protocol_sha256"] == status["protocol_sha256"]
    protocol = json.loads(
        (OUT / "protocols" / (status["protocol_sha256"] + ".json")).read_text()
    )
    for name, digest in protocol["files"].items():
        assert checksum(PROJECT / name) == digest, name
    assert (
        checksum(PROJECT / "artifacts/fixed_anchor_2012.joblib")
        == checks["model_sha256"]
    )
    metrics = pd.read_csv(OUT / "performance.csv", index_col=0)
    moves = json.loads((OUT / "large_move_review_summary.json").read_text())
    assert moves["checked_difference_exceeds_half_percentage_point"] == 0
    selected_research()
    s, b = metrics.loc["strategy"], metrics.loc["benchmark"]
    result_text = f"""
    The selected strategy finishes at **${s.final_value:,.2f}**, versus
    **${b.final_value:,.2f}** for the S&P 500 total-return benchmark. It trails by
    **${b.final_value - s.final_value:,.2f}**, or **{100 * (b.total_return - s.total_return):.2f}
    percentage points** of cumulative return. Its CAGR is **{s.CAGR:.2%}** versus
    **{b.CAGR:.2%}**. Maximum drawdown is **{s.max_drawdown:.2%}** versus
    **{b.max_drawdown:.2%}**, and Sharpe is **{s.Sharpe:.3f}** versus **{b.Sharpe:.3f}**.
    The strategy does **not** beat the benchmark. Development gains did not
    generalize to this required comparison; the risk and cost assumptions matter.
    """
    evidence_text = f"""
    Independent checks reconcile **{checks["nav_rows"]:,} daily NAV rows**,
    **{checks["fills"]:,} fills**, **{checks["event_lots_verified"]} corporate-action
    lot transitions**, and **{checks["terminal_open_lots"]} terminal open lots**.
    Maximum NAV reconstruction difference is less than $0.000001. Timing,
    dated membership, cash, costs, lot maturity, entry limits and daily metrics
    pass. The full suite has **170 passing tests**.

    The price review screens **{moves["screened_moves"]}** held daily moves over
    20%: **{moves["provider_checked"]}** have independent provider observations,
    all within 0.5 percentage point of the supplied adjusted return;
    **{moves["source_unavailable"]}** remain unverified because the endpoint did
    not supply usable observations. Genuine large returns are retained.
    This is a measured verification scope, not blanket data certification.
    """
    cells = []

    def md(content):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(content).strip()))

    def code(content):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(content).strip()))

    md(
        "# HW4: selected stock strategy and benchmark comparison\n\nSterling Couts\n\n"
        "Required evaluation: April 1, 2016–September 30, 2026.\n\n"
        + textwrap.dedent(result_text)
    )
    code("""
    import json
    import sys
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from IPython.display import display
    from hw4.settings import PROJECT
    from hw4.data import checksum

    assert Path(sys.executable).resolve() == (PROJECT / '.venv/Scripts/python.exe').resolve()
    folder = PROJECT / 'results/fixed_anchor_final'
    status = json.loads((folder / 'latest_status.json').read_text())
    proof = json.loads((folder / 'verification.json').read_text())
    assert status['verified'] and status['status'] == 'complete'
    assert checksum(PROJECT / 'artifacts/fixed_anchor_2012.joblib') == proof['model_sha256']
    nav = pd.read_csv(folder / 'daily_nav.csv', index_col='date', parse_dates=True)
    metrics = pd.read_csv(folder / 'performance.csv', index_col=0)
    assert nav.index[-1] == pd.Timestamp('2026-09-30')
    assert np.isfinite(nav[['nav', 'benchmark_nav']]).all().all()
    shown = metrics[['final_value', 'total_return', 'CAGR', 'max_drawdown', 'Sharpe']].copy()
    shown['final_value'] = shown.final_value.map('${:,.2f}'.format)
    for field in ['total_return', 'CAGR', 'max_drawdown']:
        shown[field] = shown[field].map('{:.2%}'.format)
    shown['Sharpe'] = shown.Sharpe.map('{:.3f}'.format)
    display(shown)
    """)
    code("""
    # The initial row is uninvested capital; include it to retain entry costs.
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    for field, label, color in [('nav', 'Selected fixed Ridge', '#166B7A'),
                                ('benchmark_nav', 'S&P 500 total return', '#B35C00')]:
        values = nav[field]
        axes[0].plot(values.index, values / 1e6 - 1, label=label, color=color, lw=1.8)
        axes[1].plot(values.index, values / values.cummax() - 1, color=color, lw=1.2)
    axes[0].set(title='HW4 — $1 million portfolios, April 2016–September 2026',
                ylabel='Cumulative return')
    axes[1].set(ylabel='Drawdown', xlabel='Date')
    axes[0].legend(loc='upper left')
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(folder / 'comparison.png', dpi=160, bbox_inches='tight')
    plt.show()
    """)
    md(METHOD)
    md("""
    ## Key development comparisons (2013–2015)

    All rows below use continuous accounts, not products of annual reset tests.
    The table is a selected research summary; its rows span different model and
    allocation stages. Read the change and evidence columns before attributing an
    improvement to one cause. The matching benchmark returned 48.78%.
    """)
    code("""
    development = pd.read_csv(folder / 'selected_development_comparisons.csv')
    shown = development[['experiment', 'change', 'total_return', 'max_drawdown', 'Sharpe']].copy()
    for field in ['total_return', 'max_drawdown']:
        shown[field] = shown[field].map('{:.2%}'.format)
    shown['Sharpe'] = shown.Sharpe.map('{:.3f}'.format)
    display(shown)
    """)
    md("""
    **Selection reasoning.** Rank features and profitability improved the early
    price-only comparison. Full cash deployment and a sector entry limit helped
    development performance. Fresh next-session reinvestment returned 87.96%,
    compared with 83.68% monthly, but deepened drawdown from 14.98% to 17.61%.
    Sterling selected monthly purchases for simplicity and lower development
    drawdown. Annual replacement returned 63.08%; averaging old and new forecasts
    returned 83.30%, so the original fixed fit was retained.

    Additional financial-change and price-path signals did not establish stable
    improvements across chronological checks. Longer filing delays sometimes
    improved a continuous account but were not robust to fitting and allocation
    changes. They were not adopted. These findings support restraint, not a claim
    that there is no possible signal in those variables.
    """)
    md("## Accounting and data-quality evidence\n\n" + textwrap.dedent(evidence_text))
    code("""
    # Independent identity check in addition to the separately rebuilt ledger.
    np.testing.assert_allclose(nav.nav, nav.cash + nav.holdings + nav.receivables, atol=1e-6, rtol=0)
    assert nav.cash.min() >= -1e-6  # Floating-point tolerance, not borrowing.
    local = PROJECT / status['local_output']
    fills = pd.read_csv(local / 'fills.csv')
    events = pd.read_csv(local / 'events.csv')
    display(pd.Series({
        'Final cash': nav.cash.iloc[-1],
        'Final holdings': nav.holdings.iloc[-1],
        'Final receivables': nav.receivables.iloc[-1],
        'Total trading fees': fills.fee.sum(),
        'Mean daily cash fraction': (nav.cash / nav.nav).iloc[1:].mean(),
        'Maximum observed symbol weight': proof['maximum_recorded_name_weight'],
        'Maximum observed sector weight': proof['maximum_sector_weight'],
    }, name='Account diagnostics').to_frame())
    display(events[['date', 'event_id', 'old_symbol', 'new_symbol', 'original_maturity']])
    print('Account identity and saved-model checks passed.')
    """)
    md("""
    **Data treatment.** Cited merger/rename terms carry holdings through actual
    events. Known COKE/TPL/BKNG/REZI/SPKE/CVNA unit discontinuities are normalized
    consistently in features and valuation. Changes preserve within-segment
    returns; a split is not credited a second time. CVSA observations cover only
    the inherited ATGE lot through its original April 2, 2026 maturity. Source
    citations and observed-price evidence are in `research_events/`; raw course
    data remains unchanged. No disappeared security receives an invented sale.
    """)
    md(LEARNING)
    md("""
    ## Attribution and reproducibility

    Data: course `equities_top2000.duckdb`, with historical membership and ^SP500TR.
    Course references: `Lectures/modeling/README.md`, Lecture 7 feature engineering
    and transformer examples, and Lecture 8 research-workflow examples and
    `hw4.md`. Data-loading, financial availability, pipeline and chronological
    validation ideas adapt those materials; adaptation notes accompany the code.
    The portfolio planner and lot/event ledger are implemented in this project,
    rather than delegated to a backtesting library. External corporate-action
    sources and limited quote supplements are cited in their saved registries.

    **AI assistance:** Codex assisted with implementation, debugging, tests,
    data-source investigation and documentation. Sterling directed hypotheses
    and comparisons, discussed concepts and selected the final model and trading
    policy. The submission should be explained from the code and evidence above.

    `README.md` gives setup and run commands. `PROGRESS.md` maps requirements to
    evidence. This notebook reads verified saved results and runs independently
    in the project kernel; it does not select or retrain a model.
    """)
    notebook = execute("23_selected_strategy_submission.ipynb", cells)
    save_json(OUT / "report_verification.json", notebook)
    preserve_and_write("LEARNING_GUIDE.md", LEARNING)
    readme = "# HW4 — Sterling Couts\n\nStart with **23_selected_strategy_submission.ipynb**.\n\n"
    readme += textwrap.dedent(result_text) + "\n" + textwrap.dedent(evidence_text)
    readme += r"""
## Run the selected comparison

Use Python 3.12 and the full course database at
`Lectures/modeling/data/equities_top2000.duckdb`. From this homework folder:

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe bootstrap_rank_final.py
& .\.venv\Scripts\python.exe run_fixed_anchor_final.py
& .\.venv\Scripts\python.exe audit_fixed_anchor_moves.py
& .\.venv\Scripts\python.exe build_fixed_anchor_report.py
& .\.venv\Scripts\python.exe -m pytest -q
```

For an existing environment, skip its creation and dependency installation.
The bootstrap regenerates two development dependencies if absent; it does not
replace the selected model. The evaluation loads
`artifacts/fixed_anchor_2012.joblib` without calling fit. It checks model and
input hashes, writes versioned forecasts, executes the account and independently
rebuilds its ledger. Saved quote evidence makes the price review reproducible
offline. Do not run older model-preparation scripts to replace this artifact.

## Read the work

- `23_selected_strategy_submission.ipynb`: selected method, key development
  comparisons, latest required evaluation, chart, verification and learning notes.
- `LEARNING_GUIDE.md`: concise conceptual explanation and interview exercises.
- `PROGRESS.md`: assignment checklist and evidence.
- `fixed_anchor_inputs.py`: derived inputs and prediction-only interface.
- `rank_event_accounting.py`: daily cash, lots, events and maturity accounting.
- `verify_rank_final.py`: separate ledger, timing, cap and metric reconstruction;
  the selected runner redirects its evidence to the selected result folder.
- `results/fixed_anchor_selection/`: frozen policy and artifact verification.
- `results/fixed_anchor_final/`: current performance and audit evidence.
- `research_events/`: cited events, unit repairs and limited quote evidence.
- `tests/`: hand calculations, timing, preprocessing and accounting checks.

Databases, virtual environments, caches and detailed generated ledgers under
`outputs/` stay out of Git. Saved research history remains separate from this
concise report. Evaluation scope, source quality, sectors, issuer mappings and
execution assumptions are explained in the notebook.

Sources and AI assistance are described in the notebook and code. No commit,
push, pull request or submission is performed by these scripts.

## Final audit and submission scope

Read [FINAL_AUDIT.md](FINAL_AUDIT.md) for the requirement-by-requirement review,
reproducibility evidence and remaining qualifications. The explicit
`submission_files.txt` list contains the proposed submission (about 2.6 MB).
Use that list when staging, so unrelated homework and research outputs remain
outside this pull request. No files have been staged or published.
"""
    preserve_and_write("README.md", readme)
    progress = f"""
    # HW4 requirement checklist — selected strategy

    Authority: `Lectures/Lecture 8/hw4.md` and `Lectures/modeling/README.md`.
    Verified describes the evidence below, not certification of every source row.

    | Requirement | Evidence | Status |
    |---|---|---|
    | Full homework database and historical membership | Source hash and dated entry checks; {checks["dated_entry_memberships_verified"]} purchases checked | Verified |
    | 90-day target, next-session entry and completed training labels | Frozen fit chronology; forecast/lot tests and independent maturity checks | Verified |
    | Training-only preprocessing and declared update policy | Saved pre-2013 pipeline, 64,459 development forecasts reproduced; no evaluation fit | Verified |
    | Settle model and strategy before running the final test | Fixed rules for the reported run; evaluation scope explained in report 23 | Qualification noted |
    | Explain features, model, validation and strategy | Report 23 and learning guide | Verified |
    | Own long-only, unlevered $1m backtester | Lot/event engine; cash, holdings and receivables independently reconstructed | Verified |
    | Costs, missing-price policy, cash interest and events | Explicit assumptions and cited registries; no unresolved held-price gap | Verified within stated assumptions |
    | Final period April 2016–September 2026, including terminal holdings | Complete {checks["nav_rows"]}-row account; {checks["terminal_open_lots"]} open lots marked at end | Verified |
    | Same-period S&P total-return benchmark | First test close entry, same endpoint, independent NAV comparison | Verified |
    | Final value, CAGR, maximum drawdown, daily 2% Sharpe | `performance.csv`, independent metrics and report table | Verified |
    | Plot both cumulative returns; explain result and risks | Report 23 and `comparison.png` | Verified; visual review recorded separately |
    | Test accounting and timing | 170 tests; {checks["fills"]} fills independently reconciled; error < $0.000001 | Verified |
    | Inspect flagged large price moves | {moves["provider_checked"]}/{moves["screened_moves"]} independently compared; {moves["source_unavailable"]} unavailable | Reviewed; residual limitation documented |
    | Discuss unsuccessful ideas and next research | Selected development comparisons and learning guide | Verified |
    | Readable instructions, citations and AI assistance | README, report 23, annotated modules and source registries | Verified |
    | Keep databases and large generated files out of Git | Explicit manifest excludes data/caches; clean-copy replay verified in FINAL_AUDIT.md | Verified by manifest; staging pending |
    | Commit, push, PR and reviewer assignment | Separate approval required; reviewers specified in assignment | Not started |
    """
    preserve_and_write("PROGRESS.md", progress)
    print(json.dumps(notebook, indent=2))


if __name__ == "__main__":
    run()
