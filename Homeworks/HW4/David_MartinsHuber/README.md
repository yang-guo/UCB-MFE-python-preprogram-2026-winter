# HW4 — David Martins Huber

The completed final-period backtest is in [Backtesting.ipynb](Backtesting.ipynb). Its performance summary and plots are saved. It reports extensive stale valuations, so treat this run as exploratory rather than reliable final evidence. Review [Submission_audit.md](docs/Submission_audit.md) for results and remaining limitations.

Research and modeling happen in `Research.ipynb`. `Backtesting.ipynb` runs strategies through our own simulation engine. Reusable research and strategy classes should live in Python modules so both notebooks can import them.

```text
David_MartinsHuber/
├── README.md                  # Setup and navigation
├── AGENTS.md                  # Implementation and attribution rules
├── requirements.txt           # Environment dependencies
├── Research.ipynb             # Strategy 2 HMM research and diagnostics
├── Backtesting.ipynb          # Completed 2016–2026 backtest and analysis
├── data_loading.py            # Shared, attributed course data loader
├── research/
│   └── pairs.py               # Volume-clock clusters, ADF, Kalman pair signals
├── research/
│   ├── market.py              # Benchmark/volume and cap-weighted fallback
│   ├── bars.py                # 90-session median volume bars
│   ├── fracdiff.py            # AFML fixed-width fractional differencing
│   ├── hmm.py                 # Walk-forward Gaussian HMM
│   └── pipeline.py            # End-to-end research run
├── research_outputs/          # Saved bars, states, summaries, and config
├── backtesting/
│   ├── contracts.py           # Strategy, context, and fitting interfaces
│   ├── oms.py                 # Orders, reservations, lots, and accounting
│   ├── engine.py              # Market clock, execution, refits, and results
│   ├── analytics.py           # Required metrics, alpha/beta, historical VaR
│   ├── portfolio.py           # Shadow accounts and portfolio allocation
│   ├── risk.py                # Signal-close purchase caps and entry gates
│   ├── market_data.py         # Point-in-time prices, volume, and financials
│   ├── famafrench.py          # FF5 factor construction and characteristic sorts
│   └── indexed_sales.py       # Indexed pending-sale lookup used by the run
├── strategies/
│   ├── factor.py              # FF5 regression tilt and cap-weighted baseline
│   ├── hmm_volume.py          # Volume-bar HMM market-timing strategy
│   ├── pairs.py               # Pairs_Tilt long-only relative-value sleeves
│   └── demo.py                # Example strategy; not the final model
├── tests/
│   ├── test_backtesting.py    # Accounting and timing checks
│   ├── test_portfolio.py      # Shadow coordination and allocation checks
│   ├── test_risk.py           # Risk policies and adversarial boundary checks
│   ├── test_analytics.py      # Metrics, alpha/beta, and VaR checks
│   ├── test_data_loading.py   # Loader parity and quarantine checks
│   ├── test_market_data.py    # Forward-fill, as-of timing, volume, and schema
│   ├── test_famafrench.py     # FF5 sorts, timing, exclusions, and look-ahead
│   ├── test_factor_strategy.py # Regression, quintiles, ramp, self-fit, delisting
│   ├── test_pairs.py          # Volume clock, clustering, pairs, sleeve execution
│   ├── test_erc.py            # Equal-risk-contribution allocation checks
│   ├── test_research.py       # HMM research causality and bar checks
│   ├── test_hmm_volume.py     # HMM strategy + adversarial rounds 1-3
│   └── test_audit_regressions.py
├── audits/                    # Independent reviewer scripts and evidence
│   ├── README.md
│   ├── *_checks.py
│   └── round1/                # Historical first-round evidence
└── docs/
    ├── Documentation.md       # Implementation details and research interface
    ├── Backtester_spec.md     # Design reference and agreed behavior
    ├── Portfolio.md           # Shadow accounts and allocation
    ├── Analysis.md            # Metrics and conventions
    ├── Risk.md                # Purchase controls and limits
    ├── Data.md                # Point-in-time prices, volume, and financials
    ├── FamaFrench.md          # FF5 approximations, sorts, and usage
    ├── FactorStrategy.md      # Regression signal, sleeves, and trading logic
    └── PairsTilt.md           # Volume clock, pairs, sleeves, and limits
```

## Environment

The existing local environment is `Homeworks/HW4/.venv`. To recreate it, run these commands from **Homeworks/HW4**:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r David_MartinsHuber/requirements.txt
```

Select that environment as the notebook kernel (registered locally as **Python (HW4)**). The repository already ignores virtual environments, Python caches, notebook checkpoints, and DuckDB databases.

The HW4 [ignore rules](../.gitignore) also exclude raw data directories, binary
data/model caches, bulk research bar/state exports, local secrets, and temporary
files. The database must be downloaded from bCourses; it is not submitted.
`Research.ipynb` regenerates the ignored research exports. Small research summary
tables, `research_outputs/config.json`, provenance, tests, and audit evidence stay
eligible for Git, along with the source and notebooks. Keep the required final
comparison and cumulative-return chart in the submitted notebook or a `reports/`
directory, which is not ignored. Do not place required deliverables in `data/`,
cache directories, or `outputs/` (the course repository ignores `outputs/`).

## Market data

`backtesting/market_data.py` provides one `MarketData` object for research and backtesting:

```python
from backtesting import MarketData

data = MarketData.load(database="data/equities_top2000.duckdb", end="2026-09-30")
panel = data.training_panel(feature_columns=("momentum_63", "net_margin"))
```

For research/development, load through `2015-12-31` only. The `2026-09-30`
example is the separate frozen final-test load. The 2016-01-01–2016-03-31
buffer is an embargo period and is excluded from research and fitting.

Financials become visible at `filing_date + 1 day` and are forward-filled until the next filing; price rows are never dropped. Volume is exposed as a wide panel, and strategies declare their financial columns with `required_financials`. Known non-USD reporting currencies restrict new entries from their filing availability date; earlier history and existing holdings are preserved. Unknown currency is not proof of USD. See [Data.md](docs/Data.md) for the timing rules, currency policy, fields, and limitations.

## Factor model

`backtesting/famafrench.py` builds monthly FF5 factor returns and the point-in-time eligible cross-section from `MarketData`. `strategies/factor.py` contains the rolling FF5 regression tilt and the cap-weighted market baseline. The tilt excludes the bottom quintile, targets the top quintile, and uses the Q1 2016 buffer to build three staggered 90-day sleeves. See [FamaFrench.md](docs/FamaFrench.md) and [FactorStrategy.md](docs/FactorStrategy.md).

## HMM research (strategy 2)

`Research.ipynb` runs two **independent samples**:

- **volume bars** — 90-session median volume, `5 x median` bucket, bar log returns, then the HMM;
- **daily FFD** — daily log price, fractional differentiation, then the HMM on daily observations.

The FFD study is not aggregated into the volume bars, and the volume bars are
not filtered by the FFD warm-up. Every fit is walk-forward: the volume
pipeline uses 250 bars minimum, and the daily FFD pipeline uses 504 daily
observations. The notebook writes separate bars, state series, summaries, and
forward 90-day diagnostics to `research_outputs/`.

**Data window:** research and fitting use only the HW4 development window,
`2010-01-01`–`2015-12-31`. The `2016-01-01`–`2016-03-31` buffer/embargo is not
loaded and is not used for fitting or research. The `2016-04-01`–`2026-09-30`
final test is reserved for a separate frozen-model evaluation. Data before
2010 is used only for lookbacks and HMM warm-up. Saved research states are
development-only.

The benchmark `^SP500TR` has no volume, so volume is the sum of the effective
point-in-time universe members. A winsorized top-2000 cap-weighted index and
cap-weighted volume are included as the documented fallback. Reusable code
lives in `research/`; `tests/test_research.py` covers FFD causality, volume
bar construction, and walk-forward no-lookahead checks.

## HMM volume strategy (strategy 2)

`strategies/hmm_volume.py` implements the volume-only HMM strategy as a
self-fitting `Strategy`:

- builds 90-session median volume bars with a `5 x median` bucket;
- fits a 3-state Gaussian HMM on bar log returns, refitting every 13 bars and
  scoring each new bar with the last fitted model;
- buys a market-cap-weighted basket of the top `top_n` current members by
  point-in-time market cap when the state is risk-on (`risk_on_states`, default
  all states except the lowest-mean state); volume is used only for the HMM bar
  aggregation, not for stock ranking;
- never sells early: the engine's OMS holds every lot for 90 calendar days.

`Backtesting.ipynb` contains the completed April 1, 2016–September 30, 2026
full-universe run. It combines the FF5 factor tilt and HMM volume strategy
through ERC. The result underperforms the benchmark and records extensive stale
held-price marks; the notebook flags those valuations as a material limitation,
so do not interpret the saved metrics as reliable final performance. `Pairs_Tilt`
is researched separately in `Research.ipynb` section 10.

Each fit uses only observations before the bar being scored. The strategy
rejects signals after its supplied benchmark/volume history, backward
signals, `NaT`/timezone-aware dates, and context panels with future rows.

```python
from copy import deepcopy
from backtesting import Backtester, MarketData
from research.market import build_market_series, load_market_cap_frame
from strategies.hmm_volume import HMMVolumeStrategy

# Development/research data only: no 2016+ rows.
markets = build_market_series("data/equities_top2000.duckdb", end="2015-12-31")
price, volume = markets["benchmark"].aligned()
market_caps = load_market_cap_frame(
    "data/equities_top2000.duckdb", end="2015-12-31"
)
data = MarketData.load(database="data/equities_top2000.duckdb", end="2015-12-31")
strategy = HMMVolumeStrategy(
    price, volume,
    market_cap=market_caps,
    top_n=50,
)
result = Backtester(market_data=data, strategy=deepcopy(strategy)).run(
    "2013-01-02", "2015-12-31"
)
```

For the separate final test, keep the strategy frozen at its development
configuration and load data through `2026-09-30` only for that evaluation. Do
not use the 2016 Q1 buffer or the final-test period for fitting, validation,
or research.

Pass a fresh strategy per run; the engine consumes the supplied object and
advances its state between refits. The HMM strategy tests cover lookahead,
cash/OMS limits, refit lifecycle, numerical edges, real-data integration,
mutation strength, and determinism.

## Run checks

From **Homeworks/HW4/David_MartinsHuber**:

```bash
../.venv/bin/python -m unittest discover -s tests -v
```

Automated accounting, timing, and portfolio checks are under `tests/`. The former interactive synthetic checks are retained at [audits/Testing.ipynb](audits/Testing.ipynb).

## Run the submitted backtest

[Backtesting.ipynb](Backtesting.ipynb) explicitly loads `data/equities_top2000.duckdb` from this submission and records the April 2016–September 2026 test. Download that database from bCourses to the specified path; the ignore rules keep it out of Git. The executed outputs are saved in the notebook.

The loader defaults are copied from the lecture. The notebook loads data through `end="2026-09-30"` and requests the assigned test interval. The engine rejects ranges beyond available benchmark coverage by default. Inspect `result.coverage` for requested and effective dates. Calendar endpoints between supplied sessions (such as weekends) are supported; endpoints beyond the available calendar require more data or an explicit partial-run opt-in.

The completed run compares the portfolio with $1,000,000 invested in `^SP500TR` at the first test close. Its stale-valuation limitation must be resolved before claiming a reliable evaluation. See [implementation documentation](docs/Documentation.md) and the [backtester specification](docs/Backtester_spec.md) for timing, accounting, refitting, and data limitations. The loader reuses course code with attribution; the engine is independently implemented with Codex assistance.

Portfolio metrics and conventions are documented in [Analysis.md](docs/Analysis.md). `BacktestAnalytics(result)` provides a summary for both portfolios, benchmark outperformance, and the required cumulative-return plot.

[Portfolio.md](docs/Portfolio.md) describes synchronized shadow portfolios, causal model refits, component capital allocation, and the equal-weight, minimum-variance, maximum-Sharpe, maximum-diversification, and equal-risk-contribution optimizers. `audits/Testing.ipynb` retains a synthetic interactive example. All shadow and main accounts decide before any account consumes the next execution close.

The [independent audit report](audits/README.md) records findings, corrections, reproducible reviewer scripts, and the limits of the validation. Historical first-round evidence is retained in `audits/round1/`.

[Risk.md](docs/Risk.md) describes the optional `RiskManager`, single-name caps, drawdown restrictions, and realized-volatility scaling. Pass `risk_manager=...` to the standalone engine or shadow coordinator. Risk review uses an engine-owned signal-close snapshot, not the strategy's planning copy. These controls reduce new purchases; they do not liquidate 90-day lots early or guarantee weights after price changes.
