"""Strategy settings LOCKED on 2026-11-11, after development and before the final test.

Chosen from develop.py results (2012-2015 walk-forward backtests, see README):
rule = highest development Sharpe among settings whose calendar-year return
never trailed the S&P 500 TR by more than 1 percentage point. The rule was
written after looking at the development grid, but uses no test-period data.
Do not edit these values after running final_test.py.
"""

MODEL = "ridge"            # Ridge regression on same-date feature ranks + sector
TOP_N = 50                 # names bought per purchase date
MAX_UNIVERSE_RANK = 500    # only the quarter's 500 largest (capitalisation proxy) are candidates
MIN_DOLLAR_VOL = 0.0       # no extra liquidity screen beyond the $1M eligibility floor
MAX_WEIGHT = 0.05          # no stock above 5% of NAV across its open lots
MAX_SECTOR_SHARE = 0.30    # at most 30% of a purchase date's names from one sector
TRANCHES = 3               # ramp in over three monthly tranches
FEE_BPS = 10               # commission + spread per side, in basis points
INITIAL_CASH = 1_000_000
CASH_RATE = 0.0            # idle cash earns nothing (conservative)
REFIT = "annual"           # refit on the first signal date on/after each March 31,
                           # using only labels whose exit close is before the fit date

TEST_START = "2016-04-01"  # first fill (= first test close)
TEST_END = "2026-09-30"
FIRST_SIGNAL = "2016-03-31"
