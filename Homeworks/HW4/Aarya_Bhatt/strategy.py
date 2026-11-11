import os
import glob
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def find_db_path():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../Lectures/modeling/data"))
    top2000 = os.path.join(base_dir, "equities_top2000.duckdb")
    class_db = os.path.join(base_dir, "equities_class.duckdb")
    if os.path.exists(top2000):
        return top2000
    if os.path.exists(class_db):
        return class_db
    matches = glob.glob(os.path.join(base_dir, "*.duckdb"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No duckdb database found in {base_dir}")


def pick_col(cols, candidates):
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None


def load_data(db_path):
    con = duckdb.connect(db_path, read_only=True)
    tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
    if not tables:
        raise ValueError(f"Database {db_path} contains no tables.")

    price_table = next((t for t in tables if "price" in t.lower() or "daily" in t.lower() or "ohlc" in t.lower()), tables[0])
    p_cols = [r[0] for r in con.execute(f"DESCRIBE {price_table}").fetchall()]

    t_col = pick_col(p_cols, ["ticker", "symbol", "permno", "secid"])
    d_col = pick_col(p_cols, ["date", "ts", "datadate", "trade_date", "dt"])
    px_col = pick_col(p_cols, ["adj_close", "adjclose", "close_adj", "close", "px_last", "price"])
    vol_col = pick_col(p_cols, ["volume", "vol"])

    vol_select = f", CAST({vol_col} AS DOUBLE) AS volume" if vol_col else ""
    q_prices = f"""
        SELECT {t_col} AS ticker, CAST({d_col} AS DATE) AS date, CAST({px_col} AS DOUBLE) AS close {vol_select}
        FROM {price_table}
        WHERE {px_col} IS NOT NULL AND {px_col} > 0
        ORDER BY ticker, date
    """
    prices = con.execute(q_prices).df()
    prices["date"] = pd.to_datetime(prices["date"])

    # Load benchmark (^SP500TR) from dedicated benchmark table
    bench_series = None
    if "benchmark" in tables:
        b_df = con.execute("SELECT CAST(date AS DATE) AS date, CAST(adj_close AS DOUBLE) AS close FROM benchmark ORDER BY date").df()
        b_df["date"] = pd.to_datetime(b_df["date"])
        bench_series = b_df.drop_duplicates("date").set_index("date")["close"].sort_index()

    if "universe_membership" not in tables:
        raise RuntimeError("CRITICAL: 'universe_membership' table is missing from the database.")

    u_cols = [r[0] for r in con.execute("DESCRIBE universe_membership").fetchall()]
    ut_col = pick_col(u_cols, ["ticker", "symbol"])
    ud_col = pick_col(u_cols, ["date", "quarter_start", "start_date", "effective_date", "quarter", "as_of_date", "period"])
    univ = con.execute(f"SELECT {ut_col} AS ticker, {ud_col} AS q_raw FROM universe_membership").df()
    con.close()
    return prices, univ, bench_series


def map_universe_quarters(univ_df):
    u = univ_df.copy()
    s = u["q_raw"].astype(str)
    if s.str.contains("Q", case=False).any():
        u["q_period"] = pd.PeriodIndex(s, freq="Q")
    else:
        u["q_period"] = pd.to_datetime(u["q_raw"]).dt.to_period("Q")
    return u.groupby("q_period")["ticker"].apply(set).to_dict()


def build_features_and_targets(prices, univ_map):
    df = prices.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    cal = np.sort(prices["date"].unique())
    next_date_map = pd.Series(cal[1:], index=cal[:-1]).to_dict()
    df["entry_date"] = df["date"].map(next_date_map)

    unique_entries = pd.Series(df["entry_date"].dropna().unique()).sort_values()
    maturity_target = unique_entries + pd.Timedelta(days=90)
    idx = np.searchsorted(cal, maturity_target.values, side="left")
    valid_mask = idx < len(cal)

    exit_map = {}
    for ent, ok, i in zip(unique_entries.values, valid_mask, idx):
        if ok:
            exit_map[ent] = cal[i]

    df["exit_date"] = df["entry_date"].map(exit_map)

    px_lookup = df[["ticker", "date", "close"]].rename(columns={"date": "entry_date", "close": "entry_px"})
    df = df.merge(px_lookup, on=["ticker", "entry_date"], how="left")

    px_exit_lookup = df[["ticker", "date", "close"]].rename(columns={"date": "exit_date", "close": "exit_px"})
    df = df.merge(px_exit_lookup, on=["ticker", "exit_date"], how="left")

    df["target_90d"] = df["exit_px"] / df["entry_px"] - 1.0

    g = df.groupby("ticker")["close"]
    df["ret_1d"] = g.pct_change(1)
    df["ret_5d"] = g.pct_change(5)
    df["ret_21d"] = g.pct_change(21)
    df["ret_63d"] = g.pct_change(63)
    df["ret_126d"] = g.pct_change(126)
    df["ret_252d"] = g.pct_change(252)

    df["mom_12_1"] = (1.0 + df["ret_252d"]) / (1.0 + df["ret_21d"]) - 1.0
    df["vol_21d"] = df.groupby("ticker")["ret_1d"].transform(lambda x: x.rolling(21).std() * np.sqrt(252))
    df["vol_63d"] = df.groupby("ticker")["ret_1d"].transform(lambda x: x.rolling(63).std() * np.sqrt(252))

    if "volume" in df.columns:
        df["dollar_volume"] = df["close"] * df["volume"]
        df["adv_21d"] = df.groupby("ticker")["dollar_volume"].transform(lambda x: x.rolling(21).mean())
    else:
        df["adv_21d"] = np.nan

    ma_200 = df.groupby("ticker")["close"].transform(lambda x: x.rolling(200).mean())
    high_252 = df.groupby("ticker")["close"].transform(lambda x: x.rolling(252).max())
    df["dist_ma200"] = df["close"] / ma_200 - 1.0
    df["dd_252d"] = df["close"] / high_252 - 1.0

    feature_cols = [
        "ret_5d", "ret_21d", "ret_63d", "ret_126d",
        "mom_12_1", "vol_21d", "vol_63d", "dist_ma200", "dd_252d"
    ]

    df["q_per"] = df["date"].dt.to_period("Q")

    print("Enforcing vectorized historical universe membership filter...")
    valid_rows = []
    for q_per, grp in df.groupby("q_per", sort=False):
        eligible_set = univ_map.get(q_per)
        if eligible_set is None:
            continue
        valid_mask = grp["ticker"].isin(eligible_set)
        valid_rows.append(grp[valid_mask])

    df = pd.concat(valid_rows, ignore_index=True)
    df = df.dropna(subset=feature_cols).copy()

    print("Computing cross-sectional z-scores on eligible universe...")
    for c in feature_cols:
        clipped = df.groupby("date")[c].transform(lambda x: x.clip(x.quantile(0.01), x.quantile(0.99)))
        mean = clipped.groupby(df["date"]).transform("mean")
        std = clipped.groupby(df["date"]).transform("std").replace(0, np.nan)
        df[f"{c}_z"] = ((clipped - mean) / std).fillna(0.0)

    z_cols = [f"{c}_z" for c in feature_cols]
    return df, z_cols, cal


def make_model(alpha=100.0):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])


def run_dev_validation(df, z_cols):
    train_mask = (
        (df["date"] >= "2010-01-01") &
        (df["date"] <= "2013-12-31") &
        (df["exit_date"] <= "2013-12-31") &
        df["target_90d"].notna()
    )
    val_mask = (
        (df["date"] >= "2014-01-01") &
        (df["date"] <= "2015-12-31") &
        (df["exit_date"] <= "2015-12-31") &
        df["target_90d"].notna()
    )

    train_df = df.loc[train_mask]
    val_df = df.loc[val_mask].copy()

    print("\n--- Development Sample Diagnostics ---")
    print("Training observations:", len(train_df))
    print("Validation observations:", len(val_df))
    print("Training dates:", train_df["date"].nunique())
    print("Validation dates:", val_df["date"].nunique())
    print("Training tickers:", train_df["ticker"].nunique())
    print("--------------------------------------\n")

    best_alpha, best_ic = 100.0, -np.inf
    for alpha in [10.0, 100.0, 500.0, 1000.0]:
        m = make_model(alpha)
        m.fit(train_df[z_cols], train_df["target_90d"].clip(-0.8, 1.5))
        val_df["pred"] = m.predict(val_df[z_cols])
        ic = val_df.groupby("date")[["pred", "target_90d"]].apply(
            lambda g: g["pred"].corr(g["target_90d"], method="spearman") if len(g) > 10 else np.nan
        ).mean()
        print(f"[Dev Validation] Ridge alpha={alpha:<6} | Mean Rank IC: {ic:.4f}")
        if ic > best_ic:
            best_ic = ic
            best_alpha = alpha

    return best_alpha


class Backtester:
    def __init__(self, initial_cash=1_000_000.0, cost_bps=5.0, n_tranches=3, top_k=25):
        self.initial_cash = float(initial_cash)
        self.cost_rate = cost_bps / 10_000.0
        self.n_tranches = n_tranches
        self.top_k = top_k

    def run(self, trading_dates, price_pivot, signal_schedule):
        cash = self.initial_cash
        open_positions = []
        trade_log = []
        daily_records = []

        ffill_prices = price_pivot.ffill()
        last_trading_date = trading_dates[-1]

        for dt in trading_dates:
            still_open = []
            is_final_day = (dt == last_trading_date)

            for pos in open_positions:
                if dt >= pos["exit_date"] or is_final_day:
                    px = price_pivot.at[dt, pos["ticker"]] if pos["ticker"] in price_pivot.columns else np.nan
                    if pd.isna(px):
                        px = ffill_prices.at[dt, pos["ticker"]]
                    
                    proceeds = pos["shares"] * px * (1.0 - self.cost_rate)
                    cash += proceeds
                    trade_log.append({
                        "action": "LIQUIDATE_FINAL" if (is_final_day and dt < pos["exit_date"]) else "SELL",
                        "date": dt, "ticker": pos["ticker"], "shares": pos["shares"],
                        "price": px, "cash_delta": proceeds, "cash_after": cash
                    })
                else:
                    still_open.append(pos)
            open_positions = still_open

            if not is_final_day and dt in signal_schedule and cash > 100.0:
                picks = signal_schedule[dt]
                equity_est = cash + sum(
                    p["shares"] * ffill_prices.at[dt, p["ticker"]] for p in open_positions
                )
                tranche_budget = min(cash, equity_est / self.n_tranches)
                valid_picks = []
                for tk, exit_dt in picks:
                    if tk in price_pivot.columns and pd.notna(price_pivot.at[dt, tk]) and price_pivot.at[dt, tk] > 0:
                        valid_picks.append((tk, exit_dt, price_pivot.at[dt, tk]))
                    if len(valid_picks) >= self.top_k:
                        break

                if valid_picks and tranche_budget > 100.0:
                    per_stock_budget = tranche_budget / len(valid_picks)
                    for tk, exit_dt, px in valid_picks:
                        cost_load = per_stock_budget
                        if cost_load > cash:
                            cost_load = cash
                        shares = (cost_load * (1.0 - self.cost_rate)) / px
                        if shares > 0:
                            cash -= cost_load
                            open_positions.append({
                                "ticker": tk, "shares": shares, "entry_date": dt,
                                "exit_date": exit_dt, "entry_px": px
                            })
                            trade_log.append({
                                "action": "BUY", "date": dt, "ticker": tk,
                                "shares": shares, "price": px, "cash_delta": -cost_load, "cash_after": cash
                            })

            pos_val = 0.0 if is_final_day else sum(pos["shares"] * ffill_prices.at[dt, pos["ticker"]] for pos in open_positions)

            daily_records.append({
                "date": dt, "cash": cash, "positions_value": pos_val,
                "total_value": cash + pos_val, "n_positions": len(open_positions)
            })

        return pd.DataFrame(daily_records), pd.DataFrame(trade_log)


def verify_backtester_accounting():
    dates = pd.bdate_range("2020-01-02", "2020-05-15")
    px_df = pd.DataFrame(index=dates, data={"AAA": 100.0, "BBB": 50.0})
    entry_dt = dates[1]
    target_exit = entry_dt + pd.Timedelta(days=90)
    exit_dt = dates[dates >= target_exit][0]

    px_df.loc[px_df.index >= entry_dt, "AAA"] = 100.0
    px_df.loc[px_df.index >= exit_dt, "AAA"] = 110.0

    schedule = {entry_dt: [("AAA", exit_dt)]}
    bt = Backtester(initial_cash=1_000_000.0, cost_bps=0.0, n_tranches=1, top_k=1)
    daily, trades = bt.run(dates, px_df, schedule)

    assert (daily["cash"] >= -1e-9).all(), "Cash went negative!"
    final_val = daily["total_value"].iloc[-1]
    assert np.isclose(final_val, 1_100_000.0, rtol=1e-6), f"Expected $1,100,000, got {final_val}"

    report = (
        "=== Hand-Verified Backtester Accounting Check ===\n"
        f"Initial Cash:        $1,000,000.00\n"
        f"Signal Date (t):     {dates[0].strftime('%Y-%m-%d')}\n"
        f"Entry Date (t+1):    {entry_dt.strftime('%Y-%m-%d')} | Buy 10,000 sh AAA @ $100.00 | Cash After: $0.00\n"
        f"90-Day Maturity:     {exit_dt.strftime('%Y-%m-%d')} ({(exit_dt - entry_dt).days} calendar days)\n"
        f"Exit Fill:           {exit_dt.strftime('%Y-%m-%d')} | Sell 10,000 sh AAA @ $110.00 | Cash After: $1,100,000.00\n"
        f"Min Cash Observed:   ${daily['cash'].min():,.2f} (No double-spending or leverage)\n"
        "Accounting Verification Status: PASSED\n"
    )
    print(report)
    with open("accounting_check.txt", "w") as f:
        f.write(report)


def compute_metrics(series, rf_annual=0.02):
    vals = series.dropna()
    rets = vals.pct_change().dropna()
    n_days = (vals.index[-1] - vals.index[0]).days
    years = n_days / 365.25
    cagr = (vals.iloc[-1] / vals.iloc[0]) ** (1.0 / years) - 1.0
    cum_max = vals.cummax()
    max_dd = ((vals - cum_max) / cum_max).min()
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    excess = rets - rf_daily
    sharpe = (excess.mean() / rets.std()) * np.sqrt(252.0) if rets.std() > 0 else np.nan
    return {
        "Final Value ($)": vals.iloc[-1],
        "CAGR (%)": cagr * 100.0,
        "Max Drawdown (%)": max_dd * 100.0,
        "Sharpe Ratio (Rf=2%)": sharpe
    }


def main():
    verify_backtester_accounting()

    db_path = find_db_path()
    print(f"Loading database: {db_path}")
    prices, univ_raw, bench_series = load_data(db_path)
    univ_map = map_universe_quarters(univ_raw)

    df, z_cols, cal = build_features_and_targets(prices, univ_map)
    best_alpha = run_dev_validation(df, z_cols)
    print(f"Selected Ridge alpha={best_alpha} from development validation.")

    test_start = pd.Timestamp("2016-04-01")
    test_end = pd.Timestamp("2026-09-30")
    test_dates = cal[(cal >= test_start) & (cal <= test_end)]

    rebal_signal_dates = []
    all_signal_dates = np.sort(df["date"].unique())
    prior_signals = all_signal_dates[all_signal_dates < test_dates[0]]
    cur_sig = prior_signals[-1] if len(prior_signals) > 0 else test_dates[0]

    while cur_sig <= test_end:
        rebal_signal_dates.append(cur_sig)
        target_next = cur_sig + pd.Timedelta(days=31)
        future_sigs = all_signal_dates[all_signal_dates >= target_next]
        if len(future_sigs) == 0 or future_sigs[0] > test_end:
            break
        cur_sig = future_sigs[0]

    model = None
    last_fit_year = None
    signal_schedule = {}

    for sig_dt in rebal_signal_dates:
        sig_year = pd.Timestamp(sig_dt).year
        if model is None or sig_year != last_fit_year:
            train_sub = df[
                (df["date"] >= "2010-01-01") &
                (df["exit_date"] <= sig_dt) &
                df["target_90d"].notna()
            ]
            if train_sub.empty:
                raise ValueError(f"Training subset for signal date {sig_dt} is empty.")
            model = make_model(best_alpha)
            model.fit(train_sub[z_cols], train_sub["target_90d"].clip(-0.8, 1.5))
            last_fit_year = sig_year

        day_slice = df[df["date"] == sig_dt].copy()
        if day_slice.empty:
            continue

        day_slice = day_slice[day_slice["close"] >= 5.0]
        if day_slice.empty:
            continue

        day_slice["pred"] = model.predict(day_slice[z_cols])
        day_slice["score"] = day_slice["pred"] - 0.15 * day_slice["vol_63d_z"]
        top_rows = day_slice.sort_values("score", ascending=False).head(35)

        ent_dt = top_rows["entry_date"].iloc[0]
        if pd.isna(ent_dt) or ent_dt > test_end:
            continue

        picks = []
        for _, r in top_rows.iterrows():
            ex_dt = r["exit_date"] if pd.notna(r["exit_date"]) else test_end + pd.Timedelta(days=90)
            picks.append((r["ticker"], ex_dt))
        signal_schedule[ent_dt] = picks

    test_px = prices[(prices["date"] >= test_start) & (prices["date"] <= test_end)]
    if test_px.empty:
        raise ValueError("Test price dataframe is empty.")
    price_pivot = test_px.pivot(index="date", columns="ticker", values="close").reindex(test_dates)

    bt = Backtester(initial_cash=1_000_000.0, cost_bps=5.0, n_tranches=3, top_k=25)
    daily_df, trades_df = bt.run(test_dates, price_pivot, signal_schedule)
    daily_df = daily_df.set_index("date")

    if bench_series is None or bench_series.empty:
        raise ValueError("Benchmark price series (^SP500TR) is empty.")
    sp = bench_series.reindex(test_dates).ffill()
    sp_portfolio = 1_000_000.0 * (sp / sp.iloc[0])

    strat_metrics = compute_metrics(daily_df["total_value"])
    bench_metrics = compute_metrics(sp_portfolio)

    comp_df = pd.DataFrame([strat_metrics, bench_metrics], index=["Strategy (Ridge 90d)", "S&P 500 TR (^SP500TR)"])
    print("\n=== Final Test Performance (Apr 1, 2016 - Sep 30, 2026) ===")
    print(comp_df.to_string(float_format=lambda x: f"{x:,.2f}"))
    comp_df.to_csv("performance_comparison.csv")

    plt.figure(figsize=(11, 6))
    strat_cum = daily_df["total_value"] / 1_000_000.0 - 1.0
    bench_cum = sp_portfolio / 1_000_000.0 - 1.0
    plt.plot(strat_cum.index, strat_cum * 100.0, label=f"Strategy (Final: ${strat_metrics['Final Value ($)']:,.0f})", linewidth=1.8)
    plt.plot(bench_cum.index, bench_cum * 100.0, label=f"^SP500TR (Final: ${bench_metrics['Final Value ($)']:,.0f})", linewidth=1.5, alpha=0.85)
    plt.title("Cumulative Return: 90-Day Ridge Strategy vs. S&P 500 Total Return (2016–2026)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cumulative_returns.png", dpi=150)
    plt.close()
    print("\nSaved performance_comparison.csv, accounting_check.txt, and cumulative_returns.png")


if __name__ == "__main__":
    main()