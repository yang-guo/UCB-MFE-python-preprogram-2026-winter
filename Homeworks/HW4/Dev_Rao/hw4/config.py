"""Central settings for HW4. Everything the final test depends on lives here.

Paths are resolved relative to this file so the scripts work from any cwd.
"""

from pathlib import Path

HW_DIR = Path(__file__).resolve().parents[1]           # Homeworks/HW4/dev_rao
REPO_ROOT = HW_DIR.parents[2]
DATA_DIR = REPO_ROOT / "Lectures" / "modeling" / "data"
OUTPUT_DIR = HW_DIR / "outputs"                         # gitignored
RESULTS_DIR = HW_DIR / "results"                        # small committed files

DATABASES = {
    "class": DATA_DIR / "equities_class.duckdb",
    "top2000": DATA_DIR / "equities_top2000.duckdb",
}

# ---- study periods (from hw4.md) -------------------------------------------------
LOOKBACK_START = "2007-01-01"   # price history loaded for warm-up only
TRAIN_START = "2010-01-01"      # first signal date usable as a training row
DEV_END = "2015-12-31"
VALIDATION_YEARS = (2013, 2014, 2015)
TEST_FIRST_SIGNAL = "2016-03-31"  # first entry at the 2016-04-01 close
TEST_START = "2016-04-01"
TEST_END = "2026-09-30"

# ---- timing rules (from the data guide / course_data.make_labels) ----------------
HOLDING_DAYS = 90
MAX_EXIT_DELAY = 5              # sessions an exit may slip before the label is missing
FILING_LAG_DAYS = 1             # financials usable from filing_date + 1 calendar day
MAX_FISCAL_AGE_DAYS = 550       # blank financial features older than this

# ---- eligibility screens (past data only) ----------------------------------------
MIN_HISTORY_SESSIONS = 252
MIN_MEDIAN_DOLLAR_VOLUME = 5e6  # 63d median of close*volume (split-invariant)
# NOTE: no raw-price screen. `prices.close` is split-adjusted to today's basis
# (NVDA shows $0.46 in 2010), so a "$5 close" filter would depend on FUTURE splits.

# ---- backtest rules (from hw4.md) ------------------------------------------------
INITIAL_CAPITAL = 1_000_000.0
FEE_BPS = 10.0
MAX_STALE_SESSIONS = 5
ANNUAL_RF = 0.02
