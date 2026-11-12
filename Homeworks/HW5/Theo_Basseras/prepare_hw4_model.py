# We reuse only the full-data model cells from HW4, without the backtests.
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

course_root = next(
    p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").is_file()
)
modeling_dir = course_root / "Lectures/modeling"
sys.path.insert(0, str(modeling_dir))
# The course helpers can be imported only after adding modeling_dir to sys.path.
from course_data import apply_membership, load_data, load_membership, price_features  # noqa: E402

# We keep the settings recorded in the HW4 model output.
best_alpha = 100000
half_life = 180


# We load the full homework data
database = modeling_dir / "data/equities_top2000.duckdb"

prices_full, fundamentals_full, universe_full, benchmark_full = load_data(
    database, end="2026-09-30"
)
membership_full = load_membership(database, end="2026-09-30")

calendar_full = pd.DatetimeIndex(benchmark_full.date)

print(f"Number of stocks: {prices_full.symbol.nunique()}")
print(f"First date: {prices_full.date.min()}")
print(f"Last date: {prices_full.date.max()}")


# We compute only the two features used by our model
features_full = ["momentum_63", "volatility_63"]
parts_full = []
membership_by_symbol = {symbol: group for symbol, group in membership_full.groupby("symbol")}

# The loading function already sorts prices by stock and date
symbols_full = prices_full["symbol"].to_numpy()
changes = np.flatnonzero(symbols_full[1:] != symbols_full[:-1]) + 1
starts = np.r_[0, changes]
ends = np.r_[changes, len(prices_full)]

# Membership depends on the quarter of the next trading day
next_quarters = calendar_full.to_series().shift(-1).dt.to_period("Q").dt.start_time.to_numpy()

for i in range(len(starts)):
    prices_symbol = prices_full.iloc[starts[i] : ends[i]][["symbol", "date", "adj_close"]]
    symbol = prices_symbol["symbol"].iloc[0]
    adjusted_price = prices_symbol.set_index("date")["adj_close"].reindex(calendar_full)

    # These are the same formulas as the teacher's price_features function
    daily_returns = adjusted_price.pct_change(fill_method=None)
    momentum = adjusted_price.pct_change(63, fill_method=None)
    volatility = daily_returns.rolling(63, min_periods=63).std() * np.sqrt(252)

    membership_symbol = membership_by_symbol.get(symbol, membership_full.iloc[:0])
    selection_dates = (
        membership_symbol.set_index("quarter_start")["selection_date"]
        .reindex(next_quarters)
        .to_numpy(dtype="datetime64[ns]")
    )
    eligible = selection_dates <= calendar_full.to_numpy()

    panel_symbol = pd.DataFrame(
        {
            "date": calendar_full,
            "symbol": symbol,
            "momentum_63": momentum.to_numpy(),
            "volatility_63": volatility.to_numpy(),
        }
    )
    panel_symbol = panel_symbol.loc[eligible & panel_symbol.date.ge("2010-01-01")]
    panel_symbol = panel_symbol.replace([np.inf, -np.inf], np.nan).dropna(subset=features_full)

    # We check a few stocks against the original preprocessing
    if i in [0, 500, 1500, 3000]:
        expected = apply_membership(
            price_features(prices_symbol, calendar_full), membership_symbol, calendar_full
        )
        expected = expected.loc[
            expected.eligible & expected.date.ge("2010-01-01"),
            ["date", "symbol"] + features_full,
        ]
        expected = expected.replace([np.inf, -np.inf], np.nan).dropna(subset=features_full)
        assert panel_symbol["date"].tolist() == expected["date"].tolist()
        assert np.allclose(panel_symbol[features_full], expected[features_full])

    parts_full.append(panel_symbol)

    if (i + 1) % 500 == 0:
        print(f"Stocks processed: {i + 1}", flush=True)

model_data_full = (
    pd.concat(parts_full, ignore_index=True)
    .sort_values(["date", "symbol"])
    .reset_index(drop=True)
)
del parts_full

assert not model_data_full.duplicated(["date", "symbol"]).any()
assert np.isfinite(model_data_full[features_full].to_numpy()).all()

print(f"Feature preparation complete: {len(model_data_full)} rows")
print(f"Stocks with eligible feature rows: {model_data_full.symbol.nunique()}")
display(model_data_full.head())


# We use only development signal dates
fit_date_full = pd.Timestamp("2016-04-01")

train_full = (
    model_data_full.loc[
        model_data_full.date.ge("2010-01-01") & model_data_full.date.lt("2016-01-01")
    ]
    .copy()
    .reset_index(drop=True)
)
train_full["target"] = np.nan
train_full["label_end"] = pd.NaT

# We find each stock's rows in the sorted price table
symbols_full = prices_full["symbol"].to_numpy()
changes = np.flatnonzero(symbols_full[1:] != symbols_full[:-1]) + 1
starts = np.r_[0, changes]
ends = np.r_[changes, len(prices_full)]

price_ranges = {}

for i in range(len(starts)):
    symbol = prices_full["symbol"].iloc[starts[i]]
    price_ranges[symbol] = (starts[i], ends[i])

count = 0

for symbol, index in train_full.groupby("symbol", sort=False).indices.items():
    start, end = price_ranges[symbol]
    prices_symbol = prices_full.iloc[start:end].set_index("date")["adj_close"]

    # We do not use any final-test prices to compute training outcomes
    prices_symbol = prices_symbol.loc[prices_symbol.index < fit_date_full]
    prices_symbol = prices_symbol.loc[np.isfinite(prices_symbol) & (prices_symbol > 0)]
    prices_symbol = prices_symbol.loc[prices_symbol.index.isin(calendar_full)].sort_index()

    if prices_symbol.empty:
        continue

    # We find the first available price after each signal
    entry_positions = prices_symbol.index.searchsorted(
        train_full.loc[index, "date"], side="right"
    )
    available = entry_positions < len(prices_symbol)

    index = index[available]
    entry_positions = entry_positions[available]

    entry_dates = prices_symbol.index[entry_positions]
    due_dates = entry_dates + pd.Timedelta(days=90)

    # We find the first available price on or after each due date
    exit_positions = prices_symbol.index.searchsorted(due_dates, side="left")
    available = exit_positions < len(prices_symbol)

    index = index[available]
    entry_positions = entry_positions[available]
    exit_positions = exit_positions[available]

    entry_prices = prices_symbol.iloc[entry_positions].to_numpy()
    exit_prices = prices_symbol.iloc[exit_positions].to_numpy()

    train_full.loc[index, "target"] = exit_prices / entry_prices - 1
    train_full.loc[index, "label_end"] = prices_symbol.index[exit_positions].to_numpy()

    count += 1
    if count % 500 == 0:
        print(f"Stocks processed: {count}", flush=True)

# We keep only outcomes completed before the final test
train_full = train_full.dropna(subset=features_full + ["target", "label_end"])
train_full = train_full.loc[train_full.label_end.lt(fit_date_full)].copy()

assert train_full.date.lt("2016-01-01").all()
assert train_full.label_end.lt(fit_date_full).all()

print(f"Training rows: {len(train_full)}")
print(f"Last completed outcome: {train_full.label_end.max()}")
display(train_full.head())


# We keep the settings chosen during development
alpha_full = best_alpha
half_life_full = half_life

# We give more weight to recent training examples
age_full = (fit_date_full - train_full["date"]).dt.days
weights_full = 0.5 ** (age_full / half_life_full)
weights_full = weights_full / weights_full.mean()

# We refit the same model on the full development data
model_full = make_pipeline(StandardScaler(), Ridge(alpha=alpha_full))
model_full.fit(
    train_full[features_full],
    train_full["target"],
    standardscaler__sample_weight=weights_full,
    ridge__sample_weight=weights_full,
)

print(f"Alpha: {alpha_full}")
print(f"Weight half-life: {half_life_full} calendar days")
print("Model fitted using outcomes known before April 1, 2016.")


# We take the last trading day of each month
signal_dates_full = []

for i in range(len(calendar_full) - 1):
    current_date = calendar_full[i]
    next_date = calendar_full[i + 1]

    if current_date.month != next_date.month:
        if current_date >= pd.Timestamp("2016-03-31") and next_date <= pd.Timestamp(
            "2026-09-30"
        ):
            signal_dates_full.append(current_date)

print(f"Number of signals: {len(signal_dates_full)}")
print(f"First signal: {signal_dates_full[0]}")
print(f"Last signal: {signal_dates_full[-1]}")


# We predict only on our monthly signal dates
predictions_full = model_data_full.loc[model_data_full.date.isin(signal_dates_full)].copy()

# We check that training outcomes were known before the first decision
assert train_full["label_end"].max() < predictions_full["date"].min()

predictions_full["prediction"] = model_full.predict(predictions_full[features_full])

assert np.isfinite(predictions_full["prediction"]).all()

display(predictions_full[["date", "symbol", "prediction"]].head())
print(f"Number of predictions: {len(predictions_full)}")

hw5_dir = Path(__file__).resolve().parent

# We save the fitted scaler and Ridge model together
joblib.dump(model_full, hw5_dir / "model.joblib")

# We keep one input and its prediction from HW4
example = predictions_full[features_full].iloc[[0]].copy()
prediction = model_full.predict(example)

reference = {
    "features": example.iloc[0].to_dict(),
    "prediction": float(prediction[0]),
    "model_version": "hw4-model-full-2016-04-01",
    "source": "model_full and the first row of predictions_full in hw4.ipynb",
}

(hw5_dir / "hw4_reference.json").write_text(json.dumps(reference, indent=2))

print("HW4 model saved for HW5.")
print(example)
print(f"Prediction: {prediction[0]}")
