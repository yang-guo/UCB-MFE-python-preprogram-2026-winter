# HW4 — Sterling Couts

Start with **23_selected_strategy_submission.ipynb**.


The selected strategy finishes at **$2,376,734.37**, versus
**$4,402,746.78** for the S&P 500 total-return benchmark. It trails by
**$2,026,012.40**, or **202.60
percentage points** of cumulative return. Its CAGR is **8.60%** versus
**15.17%**. Maximum drawdown is **-38.50%** versus
**-33.79%**, and Sharpe is **0.425** versus **0.773**.
The strategy does **not** beat the benchmark. Development gains did not
generalize to this required comparison; the risk and cost assumptions matter.


Independent checks reconcile **2,641 daily NAV rows**,
**4,617 fills**, **8 corporate-action
lot transitions**, and **32 terminal open lots**.
Maximum NAV reconstruction difference is less than $0.000001. Timing,
dated membership, cash, costs, lot maturity, entry limits and daily metrics
pass. The full suite has **170 passing tests**.

The price review screens **117** held daily moves over
20%: **92** have independent provider observations,
all within 0.5 percentage point of the supplied adjusted return;
**25** remain unverified because the endpoint did
not supply usable observations. Genuine large returns are retained.
This is a measured verification scope, not blanket data certification.

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
