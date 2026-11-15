"""
Part 4: My backtester for a long only and 90-day holding strategy.

What it tracks: every purchase and sale, cash, open positions, 
and the portfolio value at every market close.

Trading rules:
- $1,000,000; long only; no borrowing or leverage
- Any trading day, at the close; decisions must be made before that fill price is known
- Hold each purchase for 90 calendar days; sell at the close on maturity, or the next market session if closed
- At the end date, open lots are valued at their last close on or before it.

My assumptions (stated as the assignment asks):
- Transaction cost: 0.10% of the traded amount on every buy and sell.
- Cash earns no interest.
- Prices are adjusted closes, so dividends are already included.
- No quote on the entry day: skip that purchase
- No quote while holding: the lot keeps its last observed price.
- No quote within 5 sessions after maturity: the stock has stopped trading 
during our holding period. "last_price" sells it at its last observed price.

Sources: the structure of strategy -> orders -> fills -> daily history
follows the ideas in Lectures/modeling/backtest.py and Lecture 8. No code was reused.

Brief summary of each function:
    plan_picks(scores)          -> which 30 stocks to buy after each signal date
    load_price_matrix(symbols)  -> their daily prices and the benchmark
    run_backtest(prices, picks) -> the daily simulation
    check_accounting(result)    -> audit of the finished run
    performance(values)         -> final value, CAGR, drawdown, Sharpe
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# Settings
# transaction fee: $1.00 of fees per $1,000 traded
FEE_RATE = 0.001
# extra sessions to wait for a quote after maturity before exiting
MAX_EXIT_DELAY = 5
# calendar days
HOLDING_DAYS = 90


@dataclass
class Lot:
    """One purchase of one stock. Possible of having the same stock but different lots"""
    # running number of 0 for the first purchase, 1 for the next, etc.
    lot_id: int
    symbol: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    shares: float
    entry_price: float
    cost: float
    due_date: pd.Timestamp
    # most recent close
    last_price: float = field(init=False)
    last_quote_date: pd.Timestamp = field(init=False)

    def __post_init__(self):
        # Runs after __init__. On day one the last price is the entry price
        self.last_price = self.entry_price
        self.last_quote_date = self.entry_date


@dataclass
class BacktestResult:
    # one row per session: cash, positions_value, value, open_lots, stale_lots
    history: pd.DataFrame
    # one row per buy/ sell
    ledger: pd.DataFrame
    # one row per lot: entry, exit, return
    lots: pd.DataFrame
    # one row per planned monthly batch
    orders: pd.DataFrame


def plan_picks(scores, n):
    """Topn tickers by score on each signal date"""
    ranked = scores.sort_values(["date", "score", "symbol"], ascending=[True, False, True])
    # Get top n rows of each date, then collect their symbols into one list per date.
    return ranked.groupby("date").head(n).groupby("date").symbol.apply(list)


def run_backtest(prices, picks, start, end, initial_cash=1000000, batch_fraction=1/3,
                 fee_rate=FEE_RATE, missing_exit="zero"):
    """
    Simulate the strategy day by day.

    prices : DataFrame, index = market sessions, columns = symbols, adjusted close
    picks  : Series, index = signal date, value = list of symbols to buy at the next session
    start  : first session that may hold positions
    end    : last session
    batch_fraction : each monthly batch spends min(cash, batch_fraction x portfolio value)
    missing_exit   : "zero" (write off) or "last_price" (sensitivity) for stocks that stop trading

    Each session:
        1. Sell lots that are due
        2. Buy yesterday's orders
        3. Mark open lots at the close
        4. Plan tomorrow's orders
    """
    assert missing_exit in ("zero", "last_price")
    sessions = prices.index
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    # Begin one session before `start`. That's the signal date whose orders fill on `start`.
    # position of `start` in the calendar (-1 if missing)
    first = sessions.get_indexer([start])[0]
    assert first >= 1 and sessions[first] == start, "start must be a market session with one before it"
    run_days = sessions[first - 1:sessions.searchsorted(end, side="right")]
    assert run_days[-1] <= end, "never use prices after the end date"

    # account status
    cash = float(initial_cash)
    open_lots: list[Lot] = []
    # rows for the result tables
    ledger, closed, history, orders = [], [], [], []
    pending = None
    next_id = 0

    def book(day, kind, lot, shares, price, gross, fee):
        """Write one ledger row."""
        ledger.append({"date": day, "type": kind, "lot_id": lot.lot_id, "symbol": lot.symbol,
                       "shares": shares, "price": price, "gross": gross, "fee": fee, "cash_after": cash})

    def close_lot(day, lot, price, kind):
        """Sell or write off a whole lot at `price`, add the cash, and record the result."""
        nonlocal cash
        # value of the shares at the sale price
        gross = lot.shares * price
        # 0.1% of transaction fee
        fee = gross * fee_rate
        cash += gross - fee
        book(day, kind, lot, lot.shares, price, gross, fee)
        closed.append({"lot_id": lot.lot_id, "symbol": lot.symbol, "signal_date": lot.signal_date,
                       "entry_date": lot.entry_date, "entry_price": lot.entry_price, "shares": lot.shares,
                       "cost": lot.cost, "due_date": lot.due_date, "exit_date": day, "exit_price": price,
                       "proceeds": gross - fee, "exit_type": kind,
                       # return after both fees
                       "return_net": (gross - fee) / lot.cost - 1})

    for i, day in enumerate(run_days):
        # today's closes for every symbol. Nan means no quote 
        quotes = prices.loc[day]

        # 1. Sell lots that are due today or overdue and waiting for a quote
        still_open = []
        for lot in open_lots:
            price = quotes.get(lot.symbol, np.nan)
            # Normal case: due or overdue and there is a close today, so we sell at close
            if day >= lot.due_date and np.isfinite(price):
                close_lot(day, lot, float(price), "sell")
            
            # Check if the stock has stopped trading if number of sessions on/after the due date
            # is >=5 up to the current date. 
            elif day >= lot.due_date and (run_days[:i + 1] >= lot.due_date).sum() > MAX_EXIT_DELAY:
                if missing_exit == "zero":
                    close_lot(day, lot, 0.0, "write_off")
                else:
                    close_lot(day, lot, lot.last_price, "exit_at_last_price")
            else:
                # Not due yet or due but still within the 5 session waiting time.
                still_open.append(lot)
        open_lots = still_open

        # 2. Buy the orders planned at yesterday's close at today's close.
        if pending is not None:
            signal_date, symbols, budget = pending
            # budget per stock
            per_stock = budget / len(symbols)
            for symbol in symbols:
                price = quotes.get(symbol, np.nan)
                # if no quote today we skip and keep the cash
                if not np.isfinite(price) or price <= 0:
                    continue
                # Shares we can buy
                shares = per_stock / (price * (1 + fee_rate))
                gross = shares * price
                fee = gross * fee_rate
                # The budget was capped at yesterday's cash and only sales happened since.
                # the assert checks it on every purchase.
                assert gross + fee <= cash + 1e-6, "would spend cash we don't have"
                cash -= gross + fee
                lot = Lot(next_id, symbol, signal_date, day, shares, float(price), gross + fee,
                          day + pd.Timedelta(days=HOLDING_DAYS))
                next_id += 1
                open_lots.append(lot)
                book(day, "buy", lot, shares, float(price), gross, fee)
            # orders are used once
            pending = None

        # 3. mark every open lot at today's close or its last observed close.
        positions_value, stale = 0.0, 0
        for lot in open_lots:
            price = quotes.get(lot.symbol, np.nan)
            # fresh quote
            if np.isfinite(price):
                lot.last_price, lot.last_quote_date = float(price), day
            else:
                # keep the old price and count it
                stale += 1
            positions_value += lot.shares * lot.last_price
        # total portfolio value at today's closez
        value = cash + positions_value
        assert cash >= -1e-6, "cash went negative"
        if day >= start:
            history.append({"date": day, "cash": cash, "positions_value": positions_value,
                            "value": value, "open_lots": len(open_lots), "stale_lots": stale})

        # 4. Plan tomorrow's orders if today is a signal date and uses only today's information.
        # Nothing is planned on the last session since there's no future session.
        if day in picks.index and i + 1 < len(run_days):
            # Spend 1/3 of the portfolio, but never more than the cash available.
            budget = min(cash, batch_fraction * value)
            symbols = list(picks.loc[day])
            # skip current month if no cash avialble
            if budget > 0 and symbols:
                pending = (day, symbols, budget)
                orders.append({"signal_date": day, "entry_date": run_days[i + 1], "budget": budget,
                               "cash_at_signal": cash, "value_at_signal": value, "picks": len(symbols)})

    # Open lots at the end are listed here without an exit
    for lot in open_lots:
        closed.append({"lot_id": lot.lot_id, "symbol": lot.symbol, "signal_date": lot.signal_date,
                       "entry_date": lot.entry_date, "entry_price": lot.entry_price, "shares": lot.shares,
                       "cost": lot.cost, "due_date": lot.due_date, "exit_date": pd.NaT,
                       "exit_price": lot.last_price, "proceeds": np.nan, "exit_type": "open_at_end",
                       # unrealized before exit fee
                       "return_net": lot.shares * lot.last_price / lot.cost - 1})

    # return tables
    return BacktestResult(pd.DataFrame(history).set_index("date"), pd.DataFrame(ledger),
                          pd.DataFrame(closed).sort_values("lot_id").reset_index(drop=True),
                          pd.DataFrame(orders))


def load_price_matrix(symbols, end):
    """
    Daily adjusted closes for the given symbols on the market calendar.

    Uses the course's load_data so invalid or noncalendar prices are quarantined
    the same way as in Parts 1-2. Gaps stay NaN. Also returns the
    ^SP500TR benchmark series on the same calendar.
    """
    from data import DATABASE
    from course_data import load_data

    # Load only the stocks the strategy ever picks
    prices, _, _, benchmark = load_data(DATABASE, end=end, symbols=sorted(set(symbols)))
    calendar = pd.DatetimeIndex(benchmark.date)
    # pivot table to wide table with dates as rows and symbol as columns
    wide = prices.pivot(index="date", columns="symbol", values="adj_close").reindex(calendar)
    return wide, benchmark.set_index("date").adj_close


def check_accounting(result, prices, end, initial_cash=1000000):
    """
    Audit a finished run against the raw price table. Returns a table of checks.

    Every check is recomputed independently from ``prices`` and the ledger. They must hold for EVERY row.
    """
    h, ledger, lots = result.history, result.ledger, result.lots
    sessions = prices.index
    next_session = pd.Series(sessions[1:], index=sessions[:-1])
    buys, exits = ledger[ledger.type == "buy"], ledger[ledger.type != "buy"]
    # lots that have an exit
    closed = lots[lots.exit_type != "open_at_end"]
    sold = closed[closed.exit_type == "sell"]

    def first_quote_on_or_after(row):
        """The first session on/ after the due date where this stock has a price."""
        quotes = prices[row.symbol].loc[row.due_date:].dropna()
        return quotes.index[0] if len(quotes) else pd.NaT

    checks = {
        # The daily value is cash plus the marked positions
        "value = cash + positions, every day": np.allclose(h.value, h.cash + h.positions_value),
        "cash never negative": bool((h.cash >= -1e-6).all()),
        # Rebuild the final cash from the ledger alone
        "final cash = start - buys - fees + sale proceeds": np.isclose(
            initial_cash - (buys.gross + buys.fee).sum() + (exits.gross - exits.fee).sum(), h.cash.iloc[-1]),
        # budget always smaller than or equal to actual cash
        "each batch budget <= cash at its signal": bool((result.orders.budget <= result.orders.cash_at_signal + 1e-6).all()),
        # decide at the signal close and filled at the next session's close.
        "every buy fills at the session after its signal": bool(
            lots.entry_date.eq(lots.signal_date.map(next_session)).all()),
        # use real close from the price table
        "entry price = that day's adjusted close": np.allclose(
            lots.entry_price, [prices.at[d, s] for d, s in zip(lots.entry_date, lots.symbol)]),
        "every closed lot held >= 90 calendar days": bool(((closed.exit_date - closed.entry_date).dt.days >= 90).all()),
        "sells happen at the first quote on/after maturity": bool(
            sold.exit_date.eq(sold.apply(first_quote_on_or_after, axis=1)).all()),
        "exit price = that day's adjusted close": np.allclose(
            sold.exit_price, [prices.at[d, s] for d, s in zip(sold.exit_date, sold.symbol)]),
        # No looking ahead
        "no trade or value after the end date": bool(ledger.date.max() <= pd.Timestamp(end) and h.index.max() <= pd.Timestamp(end)),
    }
    return pd.Series(checks, name="passed").to_frame()


def benchmark_values(benchmark, start, end, initial_cash=1000000):
    """
    $1M invested in the index at the first test close, held to the end.
    Since the index already includes dividends, value = $1M x level / starting level.
    """
    level = benchmark.loc[start:end]
    return initial_cash * level / level.iloc[0]


def performance(values, risk_free=0.02):
    """
    Final value, CAGR, maximum drawdown, and Sharpe ratio from daily values.

    Conventions (same as the course): CAGR uses calendar days / 365.25.
    volatility and Sharpe use 252 trading days a year.
    """
    values = values.dropna()
    # elapsed time in years
    years = (values.index[-1] - values.index[0]).days / 365.25
    # daily returns
    daily = values.pct_change().dropna()
    # 2% a year as a daily rate
    rf_daily = (1 + risk_free) ** (1 / 252) - 1
    # daily return
    excess = daily - rf_daily
    # % below the highest value so far
    drawdown = values / values.cummax() - 1
    return pd.Series({
        "final_value": values.iloc[-1],
        "total_return": values.iloc[-1] / values.iloc[0] - 1,
        # Constant yearly growth rate
        "CAGR": (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1,
         # worst fall
        "max_drawdown": drawdown.min(),
        "annual_volatility": daily.std() * np.sqrt(252),
        # Average excess return per unit of risk, scaled from daily to yearly
        "sharpe": excess.mean() / excess.std() * np.sqrt(252),
    })
