# Lecture 7: Introduction to Machine Learning with Scikit-Learn

## Overview

This lecture introduces machine learning using scikit-learn, Python's most widely-used ML library. We'll learn how sklearn is designed, how to transform features for modeling, and how to properly evaluate models - with a special focus on techniques for time-series data that you'll encounter constantly in quantitative finance.

**Total Duration**: ~2.5 hours (with 15-minute break)

## Why This Matters for Quants

Machine learning is now a core tool in quantitative finance:
- **Alpha generation**: Predicting returns, identifying trading signals
- **Risk management**: Credit scoring, fraud detection, default prediction
- **Portfolio optimization**: Factor models, asset allocation
- **Derivatives pricing**: Calibration, hedging strategies

But ML in finance is different from ML in tech. Financial data is:
- **Time-dependent**: You can't randomly shuffle data without creating lookahead bias
- **Noisy**: Signal-to-noise ratio is extremely low
- **Non-stationary**: Relationships change over time (regime shifts)
- **Small sample**: Often limited historical data for rare events

This lecture teaches you to handle these challenges properly.

## Learning Objectives

By the end of this lecture, you will be able to:
- Explain sklearn's design philosophy and why it makes ML workflows efficient
- Build data transformation pipelines that prevent data leakage
- Choose appropriate transformers for different data types (numeric, categorical, time-series)
- Select the right model type for your problem (regression vs classification)
- Evaluate models using appropriate metrics
- Implement cross-validation correctly, especially for time-series data
- Understand why we cross-validate and what can go wrong if we don't

## Notebooks

### 1. Scikit-Learn Fundamentals: The Grammar of ML (~45 min)
`1_sklearn_fundamentals.ipynb`

**Topics:**
- The ML workflow: from raw data to predictions
- Sklearn's three interfaces: Estimator, Transformer, Predictor
- The `fit`/`transform`/`predict` pattern
- Pipelines: chaining operations together
- Avoiding data leakage with proper pipeline usage

**Key Skills:**
- `model.fit(X, y)` and `model.predict(X)`
- `transformer.fit_transform(X)` vs separate `fit()` then `transform()`
- `Pipeline([('step1', transformer), ('step2', model)])`
- Accessing pipeline components and their attributes

---

### 2. Feature Engineering with Transformers (~50 min)
`2_transformers.ipynb`

**Topics:**
- Preprocessing: StandardScaler, MinMaxScaler, RobustScaler
- Why scaling matters (and when it doesn't)
- Encoding categorical variables: OneHotEncoder, OrdinalEncoder
- Handling missing data: SimpleImputer, KNNImputer
- **Time-series features**: Lag features, rolling statistics, technical indicators
- Custom transformers: Building your own
- ColumnTransformer: Different transformations for different columns

**Key Skills:**
- Choosing the right scaler for your data
- Building lag features without lookahead bias
- Creating rolling window features (moving averages, volatility)
- Writing custom transformers with `BaseEstimator` and `TransformerMixin`
- `ColumnTransformer` for mixed-type DataFrames

---

### **[15-MINUTE BREAK]**

---

### 3. Model Selection, Evaluation & Cross-Validation (~55 min)
`3_models_and_evaluation.ipynb`

**Topics:**
- Choosing a model: regression vs classification, linear vs nonlinear
- The sklearn model selection flowchart
- Building and training models
- **Evaluation metrics**:
  - Regression: MSE, MAE, RMSE, R², MAPE
  - Classification: Accuracy, Precision, Recall, F1, AUC-ROC
- Train/test split: why and how
- **Cross-validation**:
  - Why a single train/test split isn't enough
  - K-Fold cross-validation
  - **TimeSeriesSplit**: The right way for financial data
  - Nested cross-validation for hyperparameter tuning
- GridSearchCV and RandomizedSearchCV
- Putting it all together: A complete ML pipeline

**Key Skills:**
- `train_test_split(X, y, test_size=0.2)`
- `cross_val_score(model, X, y, cv=5)`
- `TimeSeriesSplit(n_splits=5)` for temporal data
- `GridSearchCV(model, param_grid, cv=tscv)`
- Interpreting cross-validation results

## Prerequisites

- Lecture 2-4: Pandas, data cleaning, visualization
- Basic understanding of statistics (mean, std, correlation)
- Familiarity with linear regression concepts (helpful but not required)

## Data

This lecture uses **generated financial data** to ensure:
- No API keys required
- Reproducible results
- Realistic scenarios (stock returns, factor data, credit risk)

We simulate:
- Stock returns with realistic properties (fat tails, volatility clustering)
- Factor exposures for asset pricing models
- Binary classification targets (e.g., default/no-default)

## Quick Reference

### Sklearn Pattern
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Create and train
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Ridge(alpha=1.0))
])
pipe.fit(X_train, y_train)

# Predict
predictions = pipe.predict(X_test)

# Access fitted parameters
pipe.named_steps['scaler'].mean_
pipe.named_steps['model'].coef_
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score, TimeSeriesSplit

# Standard k-fold (DO NOT use for time series!)
scores = cross_val_score(model, X, y, cv=5)

# Time series split (USE THIS for financial data)
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv)

print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

### Evaluation Metrics
```python
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score

# Regression
mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# Classification
accuracy = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred_proba)
```

### Time-Series Features
```python
# Lag features
df['return_lag1'] = df['return'].shift(1)
df['return_lag5'] = df['return'].shift(5)

# Rolling features
df['volatility_20d'] = df['return'].rolling(20).std()
df['momentum_10d'] = df['close'].pct_change(10)
```

## Key Concepts to Remember

1. **Always fit on training data only** - Never let test data influence your transformations
2. **Use pipelines** - They automatically handle fit/transform correctly
3. **Time-series requires TimeSeriesSplit** - Random shuffling creates lookahead bias
4. **Cross-validation reduces variance** - A single split might be lucky or unlucky
5. **Choose metrics that match your goal** - MSE penalizes big errors more; MAE treats all errors equally

## Further Reading

- [Scikit-Learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) by Marcos López de Prado
- [Cross-Validation Pitfalls in Finance](https://arxiv.org/abs/1904.08991)
