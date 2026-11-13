import os
import sys
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import duckdb
# import Path
import matplotlib.ticker as mtick
from matplotlib.backends.backend_pdf import PdfPages
from trading import Portfolio, OrderBook, Trader, Backtest
from report import (create_summary_table, create_final_value_table, plot_summary_table,
                    plot_total_return, plot_active_return, plot_worst_bad_day, FINAL_VALUE_FMT)



from sklearn.linear_model import LassoCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
## Data Loading Module
print(sys.version)


##### Please note code block sourced from Berkeley MFE Python Lecture 8 #####
def load_data(end = '2026-09-30'):
    """Load prices, fundamentals, universe, benchmark and universe candidates from equities_top2000.duckdb.

    Parameters
    ----------
    end : str
        Last date to load (inclusive). Fundamentals are also limited to filings on or before this date.

    Returns
    -------
    tuple of DataFrames
        (prices, fundamentals, universe, benchmark, universe_candidates), with date columns as datetime64.
    """
    symbols = None
    symbol_filter = " AND symbol IN (SELECT unnest(?))" if symbols is not None else ""
    symbol_params = [list(symbols)] if symbols is not None else []
    with duckdb.connect(str('equities_top2000.duckdb'), read_only=True) as con: ## From Berkeley Lecture 8 Code
        prices = con.execute(f"""SELECT symbol,date,close,adj_close,volume FROM prices
                WHERE date<=? {symbol_filter} ORDER BY symbol,date""", [end] + symbol_params).fetchdf()
        fundamentals = con.execute(f"""SELECT symbol,fiscal_date,period,filing_date,
                reported_currency,revenue,net_income,operating_income,total_assets,capex,
                enterprise_value,total_equity,total_liabilities,
                total_debt,cash_and_equivalents,free_cash_flow,operating_cash_flow
                FROM fundamentals WHERE fiscal_date<=? AND (filing_date<=? OR filing_date IS NULL)
                {symbol_filter} ORDER BY symbol,fiscal_date""", [end, end] + symbol_params).fetchdf()
        universe_candidates = con.execute("SELECT * FROM universe_candidates").fetchdf()
        universe = con.execute(f"SELECT symbol,sector FROM universe WHERE true {symbol_filter} ORDER BY symbol", symbol_params).fetchdf()
        benchmark = con.execute("SELECT date,adj_close FROM benchmark WHERE date<=? ORDER BY date", [end]).fetchdf()
        for frame, columns in [(prices, ["date"]), (fundamentals, ["fiscal_date", "filing_date"]), (benchmark, ["date"])]:
            for column in columns:
                frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    return prices, fundamentals, universe, benchmark, universe_candidates


#### Please note code is adapted from MFE Python Modelling Lecture Material ############

def load_membership(end="2026-09-30"):
    """Load quarterly universe membership from equities_top2000.duckdb.

    Quarterly membership is a dated decision input, not the static stock pool.

    Parameters
    ----------
    end : str
        Last quarter_start date to load (inclusive).

    Returns
    -------
    DataFrame
        One row per symbol per quarter: symbol, quarter_start, quarter_end, selection_date, universe_rank.
    """

    with duckdb.connect(str('equities_top2000.duckdb'), read_only=True) as con: ## From Berkeley Lecture 8 Code
        membership = con.execute("""SELECT symbol,quarter_start,quarter_end,selection_date,universe_rank
            FROM universe_membership WHERE quarter_start<=? ORDER BY quarter_start,universe_rank""", [end]).fetchdf()
    for column in ["quarter_start", "quarter_end", "selection_date"]:
        membership[column] = pd.to_datetime(membership[column]).astype("datetime64[ns]")
    return membership

# prices, fundamentals, universe, benchmark,universe_candidates = load_data()



#### Please note code is adapted from MFE Python Lecture 8 & Converted to my purpose with Claude############
def apply_membership(data, membership):
    """Keep only rows whose symbol is in the universe for the NEXT trading session's quarter.

    A signal at date t is traded at the next session, so membership is taken from the quarter of the
    next trading date. Asserts: required columns exist, no duplicate (symbol, quarter_start), no
    look-ahead (selection_date <= date), and no duplicate date-symbol pairs in the result.

    Parameters
    ----------
    data : DataFrame
        Any DataFrame with 'date' and 'symbol' columns. Its own dates are used as the trading calendar.
    membership : DataFrame
        'symbol', 'quarter_start', 'selection_date' (one row per symbol per quarter).

    Returns
    -------
    DataFrame
        Member rows only, with the same columns as `data`, sorted by date and symbol.
    """
    missing = {'date', 'symbol'} - set(data.columns)
    assert not missing, f'data is missing required columns: {missing}'
    assert not membership.duplicated(['symbol', 'quarter_start']).any(), 'Duplicate (symbol, quarter_start) in membership'

    # Trading calendar from the data itself: each date -> next session
    calendar = pd.DatetimeIndex(sorted(pd.to_datetime(data['date']).unique()))
    next_session = pd.Series(calendar[1:], index=calendar[:-1])
    membership_quarter = pd.to_datetime(data['date']).map(next_session).dt.to_period('Q').dt.start_time

    merged = data.assign(membership_quarter=membership_quarter.values).merge(
        membership[['symbol', 'quarter_start', 'selection_date']]
            .rename(columns={'quarter_start': 'membership_quarter'}),
        on=['symbol', 'membership_quarter'], how='left', validate='many_to_one', indicator=True)

    members = merged[merged['_merge'] == 'both']
    assert (members['selection_date'] <= members['date']).all(), 'Look-ahead: universe selected after signal date'
    assert not members.duplicated(['date', 'symbol']).any(), 'Duplicate date-symbol pairs after membership filter'

    print(f'Membership: kept {len(members):,} of {len(data):,} rows '
          f'({len(members) / len(data):.1%}), {members["symbol"].nunique():,} symbols')
    return members[data.columns].sort_values(['date', 'symbol']).reset_index(drop=True)


def wrapApplyMembership(prices):
    """Load the quarterly membership table and filter `prices` to universe members.

    Parameters
    ----------
    prices : DataFrame
        Daily price panel with 'date' and 'symbol' columns.

    Returns
    -------
    DataFrame
        `prices` restricted to rows where the symbol is a universe member (see apply_membership).
    """
    membership = load_membership()
    prices_Memb = apply_membership(prices, membership)
    return prices_Memb













## CLAUDE CODE
def clean_market_cap(data, col='market_cap', jump=50, cap_max=1e13):
    """Remove bad market cap prints and forward-fill them within each symbol.

    A value is treated as bad if it is more than `jump` times the symbol's expanding mean of prior
    values, or above `cap_max` (10 trillion). Bad values are set to NA, then forward-filled per symbol.

    Parameters
    ----------
    data : DataFrame
        Must have 'symbol', 'date' and `col`.
    col : str
        Market cap column to clean.
    jump : float
        Multiple of the expanding mean above which a value is flagged.
    cap_max : float
        Absolute ceiling for a valid market cap.

    Returns
    -------
    DataFrame
        Sorted by symbol and date with `col` cleaned.
    """
    data = data.sort_values(['symbol', 'date'], ignore_index=True)
    exp_mean = data.groupby('symbol')[col].transform(lambda s: s.expanding().mean().shift())
    data[col] = data[col].mask((data[col] > jump * exp_mean) | (data[col] > cap_max))  # 1) replace with NA
    data[col] = data.groupby('symbol')[col].ffill()                                     # 2) ffill
    return data

## CLAUDE CODE

def clean_prices(prices, col='adj_close', max_bad=20):
    """Clean the price panel: cut off failing tickers, fill bad prints, and drop extreme jumps.

    Steps, per symbol:
      1. Count NA or 0 prices cumulatively; keep only rows before the count reaches `max_bad`.
      2. Replace 0 with NA and forward-fill.
      3. Compute the daily return 'ret' and drop rows with a one-day return of 70% or more
         (outliers, unadjusted splits and similar).

    Parameters
    ----------
    prices : DataFrame
        Must have 'symbol', 'date' and `col`.
    col : str
        Price column to clean.
    max_bad : int
        Number of NA/0 prices after which a symbol's remaining history is dropped.

    Returns
    -------
    DataFrame
        Cleaned prices with a 'ret' column and no NA in `col`.
    """
    prices = prices.sort_values(['symbol', 'date'], ignore_index=True)
    bad = prices[col].isna() | prices[col].eq(0)
    prices['bad_count'] = bad.groupby(prices['symbol']).cumsum()
    prices = prices[prices['bad_count'] < max_bad].copy()
    prices[col] = prices[col].replace(0, np.nan)
    prices[col] = prices.groupby('symbol')[col].ffill()
    prices['ret'] = prices.sort_values(['symbol', 'date'])[['symbol', 'date', 'adj_close']].groupby('symbol')['adj_close'].pct_change()
    prices['ret90'] = prices.groupby('symbol')['adj_close'].shift(-90) / prices['adj_close'] - 1
    prices = prices[(prices['ret']) < .7] ## Get rid of outliers / stock splits and the like
    return prices.dropna(subset=[col]).drop(columns='bad_count')




def deal_with_membership():
    """Placeholder: list of tickers excluded for bad data. Not used; the same list is applied
    in process_data_and_dates."""
    badTickers = ['RCA' , 'LTCHW']
    
    
def process_data_and_dates(prices, fundamentals, universe, benchmark, universe_candidates, reporting_lag = 2):
    """Build the daily stock panel: prices joined with sector, fundamentals and market cap.

    Fundamentals become available `reporting_lag` days after their filing date, and market caps
    `reporting_lag` days after their shares-available date, to avoid look-ahead. Fundamental values
    are forward-filled for up to 252 trading days per symbol. Known bad tickers are removed, market
    cap is cleaned (clean_market_cap) and duplicate date-symbol rows are collapsed to the first.

    Parameters
    ----------
    prices : DataFrame
        Daily price panel (the main panel everything is left-joined to).
    fundamentals : DataFrame
        Fundamentals with 'filing_date'.
    universe : DataFrame
        'symbol', 'sector'.
    benchmark : DataFrame
        Not used here; kept for a consistent call signature.
    universe_candidates : DataFrame
        Market cap proxy with 'shares_available_date'.
    reporting_lag : int
        Days between filing and availability.

    Returns
    -------
    DataFrame
        One row per date and symbol with prices, sector and fundamentals.
    """
    full_dataset = prices.merge(universe, on="symbol", how="left", indicator=False)
    fundamentals['date'] = fundamentals['filing_date'] + pd.Timedelta(days = reporting_lag)
    fundamentals = fundamentals.groupby(['date', 'symbol'], as_index=False).first()
    full_dataset = full_dataset.merge(fundamentals, on=["date", "symbol"], how="left", indicator = False)
    universe_candidates = universe_candidates.rename(columns={"market_cap_proxy": "market_cap"})
    universe_candidates['date'] = universe_candidates['shares_available_date'] + pd.Timedelta(days = reporting_lag)
    full_dataset = full_dataset.merge(universe_candidates[['date', 'symbol', 'market_cap']], on=["date", "symbol"], how="left", indicator = False)

    
    full_dataset = full_dataset[['date', 'symbol', 'sector', 'adj_close', 'close', 'volume', 'period', 'reported_currency', 'revenue', 
        'net_income', 'operating_income', 'total_assets', 'total_liabilities',
       'market_cap', 'enterprise_value', 'total_equity', 'total_debt','capex',
       'cash_and_equivalents', 'free_cash_flow', 'operating_cash_flow']]
    
    
    id_cols = ['symbol']
    val_cols = ['revenue', 'net_income', 'operating_income', 'total_assets','capex',
       'market_cap', 'enterprise_value', 'total_equity', 'total_liabilities',
       'cash_and_equivalents', 'free_cash_flow', 'operating_cash_flow']
    
    wide_df = full_dataset.copy()
    wide_df[val_cols] = full_dataset.groupby(id_cols)[val_cols].ffill(limit = 252)
    badTickers = ['RCA' , 'LTCHW']
    wide_df = wide_df[wide_df['symbol'].isin(badTickers) == False]
    wide_df = clean_market_cap(wide_df)
    wide_df = wide_df.groupby(['date', 'symbol'], as_index=False).first()
    return wide_df  

    
def manufacture_stock_signals(wide_df, benchmark):
    """Compute the five raw Fama-French style signals for each stock and date.

    Signals:
      - book_value:            (total_assets - total_liabilities) / market_cap
      - small_large:           market_cap (runs long large caps)
      - profitability:         operating_income / revenue
      - investment_aggression: 252-day rolling mean capex / total_assets
      - beta:                  -1 x rolling 3-year beta to the S&P (inverted: low beta is preferred)

    Also adds daily 'stock_return' and 'sp_return' (adds 'sp_return' to `benchmark` in place), and
    drops dates with fewer than 10 symbols.

    Parameters
    ----------
    wide_df : DataFrame
        Output of process_data_and_dates.
    benchmark : DataFrame
        'date', 'adj_close' for the S&P 500.

    Returns
    -------
    DataFrame
        Stock panel with the raw signals and returns.
    """
    # Book Value Signal # Assets - Liabilities / Cap
    # Low Beta - High Beta # Covariance with market / Variance Stock
    # Small - Large # Market Cap
    # Profitability # Operating Income / Revenue 
    # Investment # Capex / Total Assets Rolling 252  


    wide_df['book_value'] = (wide_df['total_assets'] - wide_df['total_liabilities'])/wide_df['market_cap']
    wide_df['small_large'] = wide_df['market_cap'] # I am going to let it run long large caps.
    wide_df['profitability'] = wide_df['operating_income']/wide_df['revenue']
    wide_df['investment_aggression'] = (wide_df.groupby('symbol')['capex'].transform(lambda s: s.rolling(252, min_periods=1).mean()) / wide_df['total_assets']) #negative will be worse which is what we want
    
    wide_df['stock_return'] = wide_df['adj_close']/ wide_df.groupby(['symbol'])['adj_close'].shift(1) - 1
    benchmark['sp_return'] = benchmark['adj_close']/benchmark['adj_close'].shift(1) - 1
    benchmark_2 = benchmark[['date', 'sp_return']]
    wide_df = wide_df.merge(benchmark_2, on="date", how="left", indicator=False)
    
    wide_df['roll_cov'] = wide_df['stock_return'].rolling(window = 252 * 3).cov(wide_df['sp_return'])
    wide_df['sp_var'] = wide_df['sp_return'].rolling(window= 252 * 3).var()
    wide_df['beta'] = (wide_df['roll_cov'] / wide_df['sp_var']) * -1 # Invert Beta Buy Low Beta Sell High Beta
    wide_df = wide_df[['date', 'symbol', 'sector', 'adj_close', 'close', 'volume', 'reported_currency',
        'market_cap', 'book_value', 'small_large', 'profitability', 'investment_aggression', 'beta',
        'stock_return', 'sp_return']]
    wide_df = wide_df[wide_df.groupby('date')['symbol'].transform('nunique') >= 10] ## Must have 10 tickers otherwise its an odd trading day datapoint. 
    return wide_df
    


def manufacture_sector_signals(stocksignals_df):
    """Add sector-level aggregates and remove very small stocks.

    For each date and sector: median and total market cap; stocks below 10% of the sector median
    market cap are dropped. Adds market-cap-weighted sector averages of book_value,
    investment_aggression, profitability and beta, and the simple mean of small_large.

    Parameters
    ----------
    stocksignals_df : DataFrame
        Output of manufacture_stock_signals.

    Returns
    -------
    DataFrame
        Stock panel with 'total_market_cap', 'median_market_cap' and the 'wavg_*' sector columns.
    """
    #stocksignals_df[stocksignals_df.groupby('date')['symbol'].transform('nunique') >= 10] ## Must have 10 tickers otherwise its an odd trading day datapoint. 

    ## The process here is simple: filter out deminimus, wavg of all the metrics, market cap should just be sum / number I'm not going to double count cap
    stocksignals_df['median_market_cap'] = stocksignals_df.groupby(['date', 'sector'])['market_cap'].transform('median')
    stocksignals_df['total_market_cap'] = stocksignals_df.groupby(['date', 'sector'])['market_cap'].transform('sum')
    stocksignals_df['median_market_cap'] = stocksignals_df.groupby(['date', 'sector'])['median_market_cap'].ffill(limit = 252)
    stocksignals_df = stocksignals_df[stocksignals_df['market_cap'] >= stocksignals_df['median_market_cap'] * .1] ## Get rid of anything very small.  
    #wavg of all the metrics, market cap should just be sum / number I'm not going to double count cap

    w = stocksignals_df.market_cap.where(stocksignals_df.book_value.notna())
    g = stocksignals_df.assign(wv=stocksignals_df.book_value * w, w=w).groupby(["date", "sector"])
    stocksignals_df["wavg_book_value"] = g.wv.transform("sum") / g.w.transform("sum")   
   
    w = stocksignals_df.market_cap.where(stocksignals_df.small_large.notna())
    g = stocksignals_df.assign(wv=stocksignals_df.small_large, w=w).groupby(["date", "sector"])
    stocksignals_df["wavg_small_large"] = g.wv.transform("mean") # Just avg cap
    
    w = stocksignals_df.market_cap.where(stocksignals_df.investment_aggression.notna())
    g = stocksignals_df.assign(wv=stocksignals_df.investment_aggression * w, w=w).groupby(["date", "sector"])
    stocksignals_df["wavg_investment_aggression"] = g.wv.transform("sum") / g.w.transform("sum")   
    
    w = stocksignals_df.market_cap.where(stocksignals_df.profitability.notna())
    g = stocksignals_df.assign(wv=stocksignals_df.profitability * w, w=w).groupby(["date", "sector"])
    stocksignals_df["wavg_profitability"] = g.wv.transform("sum") / g.w.transform("sum")   
    
    w = stocksignals_df.market_cap.where(stocksignals_df.beta.notna())
    g = stocksignals_df.assign(wv=stocksignals_df.beta * w, w=w).groupby(["date", "sector"])
    stocksignals_df["wavg_beta"] = g.wv.transform("sum") / g.w.transform("sum")   

    return stocksignals_df


# 5 factors 3 variations
def create_model_ideal_positions(stocksignals_df2, sector_signals_df, min_increment = 10, smooth_days = 10, lag_days = 3):
    """Build sector-neutral ideal weights for the five factor portfolios and the Umbrella.

    Steps:
      1. Base weight = stock market cap / sector market cap.
      2. Tilt = signal rank within (date, sector), scaled to +/-3%, added to the base weight.
      3. Multiply by the sector's share of total market cap (sector weights match the market).
      4. Smooth over `smooth_days`, then lag `lag_days` trading days, per symbol.
      5. Drop weights below `min_increment` bps and round the rest to `min_increment` bps.
      6. Rescale so each date sums to 1; drop the first week; Umbrella = mean of the five.

    Asserts each portfolio's weights sum to 1 on every date and there are no duplicate
    date-symbol pairs.

    Parameters
    ----------
    stocksignals_df2 : DataFrame
        Stock panel with signals, 'market_cap' and 'sector'.
    sector_signals_df : DataFrame
        Panel with 'date', 'sector', 'total_market_cap'.
    min_increment : int
        Minimum position and rounding increment, in bps.
    smooth_days : int
        Rolling-mean window for weight smoothing.
    lag_days : int
        Trading days to lag weights (avoids look-ahead).

    Returns
    -------
    DataFrame
        date, sector, symbol, Umbrella, bv, sm_lg, inv_agg, prof, bta, base_weight.
    """
    stocksignals_df2["base_weight"] = stocksignals_df2.market_cap / stocksignals_df2.groupby(["date", "sector"]).market_cap.transform("sum")    
    stocksignals_df2["base_weight"] = stocksignals_df2["base_weight"].fillna(0)
    ## CLAUDE CODE ##
    sig = {"book_value": "bv", "small_large": "sm_lg", "investment_aggression": "inv_agg", "profitability": "prof", "beta": "bta"}
    g = stocksignals_df2.groupby(["date", "sector"])[list(sig)]
    stocksignals_df2[list(sig.values())] = (((g.rank() - 1) / (g.transform("count") - 1) - 0.5) * 0.06).values
    
    stock_positions_df = stocksignals_df2[['date', 'symbol', 'sector', 'adj_close', 'close', 'volume', 'base_weight', 'bv', 'sm_lg', 'inv_agg', 'prof', 'bta']]
    portfolio_names = ['bv', 'sm_lg', 'inv_agg', 'prof', 'bta']
    # Next steps. Then Add base_weights to these wiehgts, and then remove any negative values and true up to 100.
    cols2keep  = ['date','sector', 'symbol', 'bv', 'sm_lg', 'inv_agg', 'prof', 'bta', 'base_weight']
    stock_positions_df[cols2keep].drop_duplicates().groupby("date")[portfolio_names].sum()
    stock_positions_df = stock_positions_df[cols2keep]
    stock_positions_df[portfolio_names] = stock_positions_df[portfolio_names].fillna(0) # If it doesnt have weight then go market cap weight, to exclude is an active position
    stock_positions_df[portfolio_names] = stock_positions_df[portfolio_names].add(stock_positions_df.base_weight, axis = 0)
    stock_positions_df[cols2keep].drop_duplicates().groupby(["date", "sector"])[portfolio_names].transform('sum') ### Will be less than 1 because of NA Values
    
    df = sector_signals_df[['date', 'sector', 'total_market_cap']].drop_duplicates()
    df["sector_pct"] = df.total_market_cap / df.groupby("date").total_market_cap.transform("sum")
    
    # assert np.allclose(df.groupby("date").sector_pct.sum(), 100)
    
    stock_positions_df = stock_positions_df.merge(df, on = ['date', 'sector'], how ='left')
    stock_positions_df[portfolio_names] = stock_positions_df[portfolio_names].mul(stock_positions_df.sector_pct, axis = 0)
    stock_positions_df[portfolio_names] = stock_positions_df[portfolio_names].fillna(0)
    
    s = stock_positions_df.groupby("date")[portfolio_names].sum()
    s[~np.isclose(s, 1).all(axis=1)]
    
    stock_positions_df2 = stock_positions_df.copy()
    print(stock_positions_df2[portfolio_names].min())
    print(stock_positions_df2[portfolio_names].max())

    ## CLAUDE CODE ###
    stock_positions_df2 = stock_positions_df2.sort_values(['symbol', 'date'])

    # 1) Smooth over smooth_days, then 2) lag by days_lag trading days (per symbol)



    
    stock_positions_df2[portfolio_names] = stock_positions_df2.groupby('symbol')[portfolio_names].transform(
        lambda s: s.rolling(smooth_days, min_periods=1).mean().shift(lag_days))
    
    # 3) Exclude positions < 10bps, round the rest to 10bp increments
    stock_positions_df2[portfolio_names] = stock_positions_df2[portfolio_names].where(stock_positions_df2[portfolio_names] >= (min_increment/10000), 0)
    stock_positions_df2[portfolio_names] = (stock_positions_df2[portfolio_names] / (min_increment/10000)).round() * (min_increment/10000)
    print(stock_positions_df2[portfolio_names].min())
    print(stock_positions_df2[portfolio_names].max())


    s = stock_positions_df2.groupby("date")[portfolio_names].sum()

    # 4) Rescale so each date's weights sum to 1
    stock_positions_df2 = stock_positions_df2[cols2keep].drop_duplicates()
    stock_positions_df2[portfolio_names] = stock_positions_df2[portfolio_names].div(stock_positions_df2.groupby('date')[portfolio_names].transform('sum'), axis=0).fillna(0)    #####

    stock_positions_df2 = stock_positions_df2[stock_positions_df2['date'] > stock_positions_df2['date'].min() + pd.Timedelta(days = 7)]
    s = stock_positions_df2.groupby("date")[portfolio_names].sum()

    stock_positions_df2['Umbrella'] = stock_positions_df2[portfolio_names].mean(axis =1)
    print(stock_positions_df2.groupby(['date', 'symbol']).size().loc[lambda s: s >= 2].sort_values(ascending=False))
    cols2keep  = ['date','sector', 'symbol','Umbrella', 'bv', 'sm_lg', 'inv_agg', 'prof', 'bta', 'base_weight']
    # Assert each portfolio's weights sum to 1 on every date
    final_sums = stock_positions_df2.groupby('date')[['Umbrella'] + portfolio_names].sum()
    bad_sums = final_sums[~np.isclose(final_sums, 1).all(axis=1)]
    assert bad_sums.empty, f'{len(bad_sums)} date(s) where weights do not sum to 1:\n{bad_sums.head()}'
    
    # Assert no duplicate date-symbol pairs
    n_dups = stock_positions_df2.duplicated(['date', 'symbol']).sum()
    assert n_dups == 0, f'{n_dups} duplicate date-symbol pairs:\n{stock_positions_df2[stock_positions_df2.duplicated(["date", "symbol"], keep=False)].sort_values(["date", "symbol"]).head(10)}'
    
    return stock_positions_df2[cols2keep]
    



    ## Actual Return is well the actual return of the ideal weights.


def calculate_return(stock_positions_long, prices_clean, benchmark):
    """Frictionless total return of each portfolio's ideal weights, plus the S&P 500.

    Daily portfolio return = sum over stocks of weight x daily stock return (no trading constraints
    or transaction costs). Returns are compounded into a total return index.

    Parameters
    ----------
    stock_positions_long : DataFrame
        Long positions indexed by date with 'symbol', 'portfolio', 'position'.
    prices_clean : DataFrame
        Cleaned price panel with 'symbol', 'date', 'adj_close'.
    benchmark : DataFrame
        'date', 'sp_return'.

    Returns
    -------
    DataFrame
        Long format: date, portfolio, total_return_index (includes 'sp500').
    """
    returns =prices_clean.sort_values(['symbol', 'date'])[['symbol', 'date', 'adj_close', 'ret']]
    
    
    #returns['ret'] = prices_clean.sort_values(['symbol', 'date'])[['symbol', 'date', 'adj_close']].groupby('symbol')['adj_close'].pct_change()

    portfolio_returns = (stock_positions_long.reset_index()
    .merge(returns[['symbol', 'date', 'ret']], on=['date', 'symbol'], how='left')
    .set_index('date'))
    
    portfolio_returns['wret'] = portfolio_returns['position'] * portfolio_returns['ret']

 
    # 1) Strategy daily returns, one column per portfolio
    strategy_returns = (portfolio_returns.assign(wret=portfolio_returns['position'] * portfolio_returns['ret'])
                        .groupby(['date', 'portfolio'])['wret'].sum()
                        .unstack('portfolio'))
    
    # 2) Join S&P daily returns on date (left join keeps only strategy dates)
    strategy_returns = strategy_returns.join(benchmark.set_index('date')['sp_return'].rename('sp500'), how='left')
    
    # 3) Total return index for all columns
    total_return_df = (1 + strategy_returns.fillna(0)).cumprod()
    
    # 4) Long format: date, portfolio, total_return_index
    total_return_long = total_return_df.reset_index().melt(id_vars='date', var_name='portfolio', value_name='total_return_index')
    return(total_return_long)    



### CLAUDE ##
def create_time_series_and_table_assessments(total_return_long,plotTitle, rf=0.02):
    """Plot total return of every portfolio vs the S&P 500 and build the performance table.

    The period is detected from the data, and every series is rebased to 1 at the start.

    Parameters
    ----------
    total_return_long : DataFrame
        date, portfolio, total_return_index (must include 'sp500').
    plotTitle : str
        Prefix for the chart title.
    rf : float
        Annual risk-free rate for the Sharpe ratio.

    Returns
    -------
    fig : matplotlib Figure
        Total return chart with the S&P 500 highlighted in black.
    table : DataFrame
        One row per portfolio: CAGR, Vol, IR (vs S&P), Sharpe, SP CAGR, SP Sharpe.
    """
    tri =total_return_long.pivot(index='date', columns='portfolio', values='total_return_index')
    tri = tri / tri.iloc[0]                                   # rebase to 1 at start
    rets = tri.pct_change().dropna()
    start, end = tri.index.min(), tri.index.max()
    years = (end - start).days / 365.25

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for col in tri.columns.drop('sp500'):
        ax.plot(tri.index, tri[col], lw=1.6, label=col)
    ax.plot(tri.index, tri['sp500'], lw=3, color='black', label='S&P 500', zorder=5)
    ax.axhline(1, color='gray', lw=0.8, ls='--')
    ax.set_title(f'{plotTitle}: Factor Portfolios vs S&P 500 ({start:%b %Y}–{end:%b %Y})', fontsize=14, fontweight='bold', loc='left')
    ax.set_ylabel('Growth of $1')
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, ncol=len(tri.columns), loc='upper left')
    fig.tight_layout()

    # --- Table ---
    sp = rets['sp500']
    active = rets.sub(sp, axis=0)
    cagr = tri.iloc[-1] ** (1 / years) - 1
    vol = rets.std() * np.sqrt(252)
    table = pd.DataFrame({
        'CAGR': cagr,
        'Vol': vol,
        'IR': active.mean() * 252 / (active.std() * np.sqrt(252)),
        'Sharpe': (rets.mean() * 252 - rf) / vol,
        'SP CAGR': cagr['sp500'],
        'SP Sharpe': (sp.mean() * 252 - rf) / vol['sp500'],
    }).drop('sp500')

    return fig, table






## Claude
def bad_days_screener(trader, prices_wide, threshold=0.10, top_n=5):
    """Find days where the backtest NAV moved more than `threshold`, and the stocks behind them.

    Holdings are rebuilt from the trade log. Each stock's contribution is the previous day's shares
    x today's price change, as a share of the previous day's NAV.

    Parameters
    ----------
    trader : Trader
        A Trader that has completed run().
    prices_wide : DataFrame
        Prices, date x symbol.
    threshold : float
        Absolute daily NAV return that counts as a bad day.
    top_n : int
        Number of top contributing stocks to list per bad day.

    Returns
    -------
    DataFrame
        One row per bad day and stock: date, nav_ret, symbol, contribution, shares_held,
        price_prev, price, stock_ret. Empty if there are no bad days.
    """
    daily, trades = trader.orderbook.daily, trader.orderbook.trades
    nav_ret = daily['nav'].pct_change()
    bad = nav_ret[nav_ret.abs() > threshold].index
    if bad.empty:
        return pd.DataFrame()

    px = prices_wide.ffill().reindex(daily.index)
    # Shares held at each day's close, rebuilt from the trade log
    shares = (trades.pivot_table(index='date', columns='symbol', values='shares', aggfunc='sum')
              .reindex(daily.index).fillna(0).cumsum())
    # Each stock's P&L: yesterday's shares × today's price change, as % of yesterday's NAV
    contrib = (shares.shift() * px[shares.columns].diff()).div(daily['nav'].shift(), axis=0)

    rows = []
    for d in bad:
        top = contrib.loc[d].abs().nlargest(top_n).index
        prev_d = daily.index[daily.index.get_loc(d) - 1]
        rows.append(pd.DataFrame({
            'date': d, 'nav_ret': nav_ret[d], 'symbol': top,
            'contribution': contrib.loc[d, top].values,
            'shares_held': shares.loc[prev_d, top].values,
            'price_prev': px.loc[prev_d, top].values,
            'price': px.loc[d, top].values,
            'stock_ret': (px.loc[d, top] / px.loc[prev_d, top] - 1).values}))
    return pd.concat(rows, ignore_index=True)






def run_backtest(stock_positions_df, prices_clean, benchmark, portfolio = 'Umbrella',  startDate = '2016-04-01', endDate = '2026-09-30',
                 lag_days=1, cash=1_000_000, plotTitle=None):
    """Run the constrained trading backtest for one portfolio.

    Uses the Trader from trading.py: 90-day holding period, 2 bp transaction cost, trades only above
    5 bps and 1 share, cash kept between 0 and 20 bps, with daily invariant asserts.

    Parameters
    ----------
    stock_positions_df : DataFrame
        Ideal weights from create_model_ideal_positions.
    prices_clean : DataFrame
        Cleaned price panel ('date', 'symbol', 'adj_close').
    benchmark : DataFrame
        'date', 'sp_return'.
    portfolio : str
        Weight column to trade (e.g. 'Umbrella', 'bv').
    startDate, endDate : str
        Backtest window.
    lag_days : int
        Extra trading-day lag applied to the target weights.
    cash : float
        Starting cash (starting NAV).
    plotTitle : str or None
        Chart title; defaults to the portfolio name.

    Returns
    -------
    Backtest
        Holds the trader, prices_wide, total return index (portfolio and sp500) and bad days.
    """
    # 1) Prices: date x symbol
    prices_wide = (prices_clean.assign(date=pd.to_datetime(prices_clean['date']))
                   .pivot(index='date', columns='symbol', values='adj_close'))

    # 2) Target weights for this portfolio (lagged), restricted to the window
    targets = (stock_positions_df[['date', 'symbol', portfolio]]
               .drop_duplicates(subset=['date', 'symbol'])
               .pivot(index='date', columns='symbol', values=portfolio)
               .shift(lag_days).fillna(0)
               .loc[startDate:endDate])

    # 3) Run the trader
    trader = Trader(Portfolio(cash=cash), OrderBook())
    daily = trader.run(targets, prices_wide.loc[targets.index.min():targets.index.max()])
    bad_days = bad_days_screener(trader, prices_wide, threshold=0.10)
    print(bad_days.round(4))
    # 4) Total return index + S&P, joined on date, both rebased to 1
    total_return = (daily[['nav']].rename(columns={'nav': portfolio}) / daily['nav'].iloc[0]).join(
        (1 + benchmark.set_index('date')['sp_return']).cumprod().rename('sp500'), how='left')
    total_return['sp500'] = total_return['sp500'] / total_return['sp500'].iloc[0]

    # 5) Plot + assessment
    fig, assessment = create_time_series_and_table_assessments(
        total_return.reset_index(names='date').melt(id_vars='date', var_name='portfolio',
                                                    value_name='total_return_index'),
        plotTitle=plotTitle or portfolio)

    return Backtest(portfolio, trader, prices_wide, total_return, bad_days)




def create_backtest_plots_and_summary(main_backtest, report_name, save_location):
    """Build the summary tables + charts and save them as one PDF.

    Pages: performance summary table, final value table, total return chart, cumulative active
    return chart, and the worst bad day table.

    Parameters
    ----------
    main_backtest : Backtest
        Output of run_backtest.
    report_name : str
        PDF file name (without extension) and title.
    save_location : str
        Folder to save into (created if missing).

    Returns
    -------
    summary : DataFrame
        CAGR, Vol, IR, Sharpe, Max Drawdown, Annual Turnover, Annual TC.
    final_values : DataFrame
        Starting value, final value, cash, open positions, # positions, total return.
    """
    tri = main_backtest.total_return
    period = f'{tri.index[0]:%b %Y} to {tri.index[-1]:%b %Y}'
    summary = create_summary_table(main_backtest)
    final_values = create_final_value_table(main_backtest)

    figs = [plot_summary_table(summary, f'{report_name} — {period}'),
            plot_summary_table(final_values, f'Final Value — {period}', fmt=FINAL_VALUE_FMT),
            plot_total_return(main_backtest),
            plot_active_return(main_backtest),
            plot_worst_bad_day(main_backtest)]

    os.makedirs(save_location, exist_ok=True)
    pdf_path = os.path.join(save_location, f'{report_name}.pdf')
    with PdfPages(pdf_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f'Saved {pdf_path}')
    return summary, final_values


  
    
def model_assessor(stock_positions_df, sector_signals_df,  prices_clean, benchmark, startDate = '2010-01-01', endDate =' 2015-12-31', plotTitle = 'Training'):
    """Assess all portfolios' ideal weights (frictionless) over a date window.

    Parameters
    ----------
    stock_positions_df : DataFrame
        Ideal weights from create_model_ideal_positions.
    sector_signals_df : DataFrame
        Not used; kept for a consistent call signature.
    prices_clean : DataFrame
        Cleaned price panel.
    benchmark : DataFrame
        'date', 'sp_return'.
    startDate, endDate : str
        Assessment window.
    plotTitle : str
        Chart title prefix (e.g. 'Training').

    Returns
    -------
    fig : matplotlib Figure
        Total return chart.
    assessment_table : DataFrame
        CAGR, Vol, IR, Sharpe, SP CAGR, SP Sharpe per portfolio.
    """
    stock_positions_df.columns
    stock_positions_long = (stock_positions_df
    .drop(columns=['base_weight'])
    .melt(id_vars=['date', 'sector', 'symbol'], var_name='portfolio', value_name='position')
    .set_index('date'))
    
    total_return_long = calculate_return(stock_positions_long, prices_clean, benchmark)
    total_return_long = total_return_long[total_return_long['date'] <= endDate]
    total_return_long = total_return_long[total_return_long['date'] >= startDate]
    fig, assessment_table = create_time_series_and_table_assessments(total_return_long, plotTitle)
    #plt.show()
    print(assessment_table.round(4))
    return fig, assessment_table


def plot_final_assesment(stock_positions_df, sector_signals_df,  prices_clean, benchmark, startDate = '2016-04-01', endDate =' 2026-09-30'):
    """Run model_assessor over the out-of-sample test window, titled 'Final Assessment'.

    Returns
    -------
    fig : matplotlib Figure
        Total return chart for the test period.
    assessment_table : DataFrame
        CAGR, Vol, IR, Sharpe, SP CAGR, SP Sharpe per portfolio.
    """
    fig, assessment_table = model_assessor(stock_positions_df, sector_signals_df,  prices_clean, benchmark, startDate =startDate, endDate = endDate, plotTitle = 'Final Assessment')
    return fig, assessment_table
    ## I will call a final model assessment for April 2016 -> Sept 2026 once after its all built
    
def plot_scatter_with_fit(x, y, title='', xlabel='', ylabel='', max_points=50_000):
    """Fast scatter for very large data: plots a random sample, OLS line fit on ALL points.

    Parameters
    ----------
    x, y : array-like
        Data to plot. Inf and NA pairs are dropped.
    title, xlabel, ylabel : str
        Chart labels.
    max_points : int
        Maximum number of points drawn (the fit, correlation and n use every point).

    Returns
    -------
    matplotlib Figure
    """
    data = pd.DataFrame({'x': x, 'y': y}).replace([np.inf, -np.inf], np.nan).dropna()
    sample = data.sample(min(len(data), max_points), random_state=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sample['x'], sample['y'], s=2, alpha=0.15, color='#2a78d6', linewidths=0, rasterized=True)

    slope, intercept = np.polyfit(data['x'], data['y'], 1)
    xs = np.array([data['x'].min(), data['x'].max()])
    ax.plot(xs, slope * xs + intercept, color='#eb6834', lw=2,
            label=f'fit: y = {slope:.2f}x + {intercept:.3f}  (corr {data["x"].corr(data["y"]):.3f}, n={len(data):,})')

    ax.set_title(title, fontsize=12, fontweight='bold', loc='left')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='upper left', fontsize=9)
    return fig



def plot_quarterly_predicted_vs_actual(stock_positions_df, lasso_predictions, portfolio_names):
    """Quarterly predicted vs actual 90-day return for each portfolio.

    On the last trading day of each quarter: predicted = sum(weight x pred_<portfolio>),
    actual = sum(weight x ret90). One panel per portfolio, one point per quarter.

    Parameters
    ----------
    stock_positions_df : DataFrame
        Weights with 'pred_<portfolio>' prediction columns merged in.
    lasso_predictions : DataFrame
        Long predictions with 'date', 'symbol', 'ret90'.
    portfolio_names : list of str
        Portfolios to plot.

    Returns
    -------
    fig : matplotlib Figure
    quarterly : dict
        Portfolio name -> DataFrame of predicted and actual per quarter.
    """
    # Actual ret90 per date/symbol (same for every signal, so take one)
    actual = lasso_predictions.drop_duplicates(['date', 'symbol'])[['date', 'symbol', 'ret90']]
    data = stock_positions_df.merge(actual, on=['date', 'symbol'], how='left')

    # Last trading day of each quarter
    quarter_ends = data.groupby(data['date'].dt.to_period('Q'))['date'].max()
    data = data[data['date'].isin(quarter_ends)]

    fig, axes = plt.subplots(len(portfolio_names), 1, figsize=(11, 2.6 * len(portfolio_names)), sharex=True)
    quarterly = {}
    for ax, p in zip(np.atleast_1d(axes), portfolio_names):
        q = data.groupby('date').apply(lambda g: pd.Series({
            'predicted': (g[p] * g[f'pred_{p}']).sum(min_count=1),
            'actual':    (g[p] * g['ret90']).sum(min_count=1)}))
        quarterly[p] = q

        ax.scatter(q.index, q['actual'], s=40, color='#2a78d6', label='Actual', zorder=3)
        ax.scatter(q.index, q['predicted'], s=40, color='#eb6834', marker='D', label='Predicted', zorder=3)
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_title(f'{p}  (corr {q["predicted"].corr(q["actual"]):.2f})', fontsize=11, fontweight='bold', loc='left')
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    np.atleast_1d(axes)[0].legend(frameon=False, loc='upper left', ncol=2)
    fig.suptitle('Quarterly Predicted vs Actual 90-Day Portfolio Return', fontsize=14, fontweight='bold', x=0.01, ha='left')
    #fig.tight_layout()
    return fig, quarterly

def signal_assessor(stock_positions_df, stocksignals_df,  prices_clean, benchmark, startDate = '2010-01-01', endDate =' 2015-12-31', plotTitle = 'Training'):
    """Test each raw signal's power to predict the next 90-day stock return with Lasso.

    For each of the five signals, fits LassoCV (TimeSeriesSplit, 5 folds) of ret90 on the signal over
    the window, predicts every row (clipped to +/-20%), and maps predictions onto the portfolio
    weights to compare predicted vs actual quarterly portfolio returns.

    Parameters
    ----------
    stock_positions_df : DataFrame
        Ideal weights from create_model_ideal_positions.
    stocksignals_df : DataFrame
        Raw signals from manufacture_stock_signals.
    prices_clean : DataFrame
        Cleaned price panel (used for forward 90-day returns).
    benchmark : DataFrame
        Not used; kept for a consistent call signature.
    startDate, endDate : str
        Training window.
    plotTitle : str
        Not used; kept for a consistent call signature.

    Returns
    -------
    signal_scatters : dict
        Signal -> predicted vs actual scatter figure.
    figs : matplotlib Figure
        Quarterly predicted vs actual portfolio return chart.
    quarterly : dict
        Portfolio -> DataFrame of quarterly predicted and actual returns.
    """
    signal_names =['book_value', 'small_large', 'profitability', 'investment_aggression', 'beta']
    cols2keep = ['date', 'symbol', 'book_value', 'small_large', 'profitability', 'investment_aggression', 'beta']
    stocksignals_df[cols2keep]
    rets_clean = prices_clean.copy()
    rets_clean = rets_clean.sort_values(['symbol', 'date'])

    lasso_df = stocksignals_df[cols2keep].merge(rets_clean[['date', 'symbol', 'ret90']],  on = ['date', 'symbol'], how ='left')
    lasso_df = lasso_df[lasso_df['date'] >= startDate]
    lasso_df = lasso_df[lasso_df['date'] <= endDate]
    last_dates = lasso_df.groupby(lasso_df['date'].dt.year)['date'].max()
    # Training so just use all date
    lasso_long = lasso_df.melt(id_vars=['date', 'symbol', 'ret90'], var_name='signal', value_name='value').dropna()
    lasso_long = lasso_long.replace([np.inf, -np.inf], np.nan).dropna()
    
    
    # Fit on raw data (rows with a known target)
    lasso_models = {}
    for signal_name, signal_data in lasso_long.dropna(subset=['value', 'ret90']).sort_values('date').groupby('signal'):
        lasso_models[signal_name] = LassoCV(cv=TimeSeriesSplit(n_splits=5)).fit(
            signal_data[['value']], signal_data['ret90'])
    
    # Predict on ALL rows with a signal value
    lasso_predictions = pd.concat(
        [signal_data[['date', 'symbol', 'signal', 'ret90']].assign(
            prediction=lasso_models[signal_name].predict(signal_data[['value']]).clip(-0.20, 0.20))
         for signal_name, signal_data in lasso_long.dropna(subset=['value']).groupby('signal')],
        ignore_index=True)

    lasso_predictions_wide = lasso_predictions.pivot_table(index=['date', 'symbol'], columns='signal', values='prediction').reset_index()
    signal_scatters = {s: plot_scatter_with_fit(d['prediction'], d['ret90'], title=f'{s}: Predicted vs Actual ret90',
                                            xlabel='Predicted', ylabel='Actual')
                   for s, d in lasso_predictions.groupby('signal')}
    
    signal_to_portfolio = {'book_value': 'bv', 'small_large': 'sm_lg', 'investment_aggression': 'inv_agg',
                       'profitability': 'prof', 'beta': 'bta'}

    pred_wide = (lasso_predictions.assign(signal=lasso_predictions['signal'].map(signal_to_portfolio))
                 .pivot_table(index=['date', 'symbol'], columns='signal', values='prediction')
                 .add_prefix('pred_')
                 .reset_index())
    
    stock_positions_df_pred = stock_positions_df.merge(pred_wide, on=['date', 'symbol'], how='left') 
    portfolio_names = ['bv', 'sm_lg', 'inv_agg', 'prof', 'bta'] 
    figs, quarterly = plot_quarterly_predicted_vs_actual(stock_positions_df_pred, lasso_predictions, portfolio_names)
    return(signal_scatters, figs, quarterly)
    


ASSESSMENT_FMT = {'CAGR': '{:.2%}', 'Vol': '{:.2%}', 'IR': '{:.2f}', 'Sharpe': '{:.2f}',
                  'SP CAGR': '{:.2%}', 'SP Sharpe': '{:.2f}'}
def wrap_signal_assessment_report(train_fig, train_assessment_table, test_fig, test_assessment_table,
                                    signal_scatters, figs, report_name='Fama French Signal Assessment',
                                    save_location= os.getcwd()):
    """Save train/test time series + tables, quarterly time series, and signal scatters to one PDF.

    Page order: train chart, train table, test chart, test table, quarterly chart(s), signal scatters.

    Parameters
    ----------
    train_fig, test_fig : matplotlib Figure
        Total return charts from model_assessor / plot_final_assesment.
    train_assessment_table, test_assessment_table : DataFrame
        Matching performance tables.
    signal_scatters : dict, list or Figure
        Scatter figures from signal_assessor.
    figs : dict, list or Figure
        Quarterly predicted vs actual figure(s).
    report_name : str
        PDF file name (without extension).
    save_location : str
        Folder to save into (created if missing).
    """
    def table_fig(table, title):
        fmt = {c: ASSESSMENT_FMT.get(c, '{:.4f}') for c in table.columns}
        return plot_summary_table(table, title, fmt=fmt)

    as_list = lambda f: list(f.values()) if isinstance(f, dict) else (list(f) if isinstance(f, (list, tuple)) else [f])

    pages = ([train_fig, table_fig(train_assessment_table, 'Train Period: Assessment'),
              test_fig, table_fig(test_assessment_table, 'Test Period: Assessment')]
             + as_list(figs) + as_list(signal_scatters))

    os.makedirs(save_location, exist_ok=True)
    pdf_path = os.path.join(save_location, f'{report_name}.pdf')
    with PdfPages(pdf_path) as pdf:
        for fig in pages:
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f'Saved {pdf_path} ({len(pages)} pages)')     
 

def main():
    """Run the full pipeline: load -> membership -> signals -> ideal weights -> assessments -> backtest.

    Saves two PDFs to the current working directory:
      - 'Fama French Signal Assessment.pdf'
      - 'Fama-French 5 Factor Sector Neutral Backtest.pdf'

    Returns
    -------
    Backtest
        The Umbrella portfolio backtest.
    """
    # 1) Load data and restrict to the quarterly universe
    prices, fundamentals, universe, benchmark, universe_candidates = load_data()
    prices_Memb = wrapApplyMembership(prices)

    # 2) Signals and ideal weights
    wide_df = process_data_and_dates(prices_Memb, fundamentals, universe, benchmark, universe_candidates)
    stocksignals_df = manufacture_stock_signals(wide_df, benchmark)
    sector_signals_df = manufacture_sector_signals(stocksignals_df)
    stock_positions_df = create_model_ideal_positions(sector_signals_df, sector_signals_df, min_increment = 10, smooth_days = 10, lag_days = 3)
    prices_clean = clean_prices(prices_Memb)

    # 3) Model and signal assessment (train 2010-2015, test Apr 2016 - Sep 2026)
    train_fig, train_assessment_table = model_assessor(stock_positions_df, sector_signals_df,  prices_clean, benchmark, startDate = '2010-01-01', endDate =' 2015-12-31', plotTitle = 'Training')
    test_fig, test_assessment_table = plot_final_assesment(stock_positions_df, sector_signals_df,  prices_clean, benchmark, startDate = '2016-04-01', endDate =' 2026-09-30')
    signal_scatters, figs, quarterly = signal_assessor(stock_positions_df, stocksignals_df,  prices_clean, benchmark, startDate = '2010-01-01', endDate =' 2015-12-31', plotTitle = 'Training')

    wrap_signal_assessment_report(train_fig, train_assessment_table, test_fig, test_assessment_table,
                                        signal_scatters, figs, report_name='Fama French Signal Assessment',
                                        save_location= os.getcwd())

    # 4) Trading backtest of the Umbrella portfolio
    main_backtest = run_backtest(stock_positions_df, prices_clean, benchmark, portfolio = 'Umbrella',
                      startDate='2016-04-01', endDate='2026-09-30', lag_days=1)

    create_backtest_plots_and_summary(main_backtest, report_name = 'Fama-French 5 Factor Sector Neutral Backtest', save_location = os.getcwd())
    return main_backtest


if __name__ == '__main__':
    main()



# 145
## test the entire program.
## Short write up
## Push to git


