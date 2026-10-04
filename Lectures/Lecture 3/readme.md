# Lecture 3: Data Visualization & Cleaning

## Overview

This lecture covers two foundational skills: visualizing data to understand what you're working with, and cleaning data to make it usable. These skills apply to virtually every data analysis task.

**Total Duration**: ~3 hours (with 15-minute break)

## Learning Objectives

By the end of this lecture, you will be able to:
- Create visualizations to explore datasets
- Identify data quality issues visually and programmatically
- Handle missing values, outliers, duplicates, and messy text
- Apply a systematic workflow for cleaning data

## Notebooks

### 1. Data Visualization: Seeing Your Data (~60 min)
`1_visualization_seeing_your_data.ipynb`

**Topics:**
- Why visualization matters (Anscombe's Quartet)
- Quick plots with pandas `.plot()` - line, histogram, scatter, bar, box
- Choosing the right chart for your question
- Matplotlib architecture: procedural vs object-oriented
- Combining pandas plotting with matplotlib
- Scatter matrices, lag plots, autocorrelation
- Saving figures

**Key Skills:**
- `df.plot(kind='...')` for quick exploration
- `fig, ax = plt.subplots()` for more control
- Scatter matrices for pairwise analysis
- Rolling statistics for time series

---

### 2. Data Cleaning Fundamentals (~50 min)
`2_data_cleaning_fundamentals.ipynb`

**Topics:**
- Missing values: None, NaN, and Inf
- Detection methods: `.isnull()`, `np.isfinite()`
- Handling strategies: drop, fill, interpolate
- Outlier detection: Z-score, IQR, percentile methods
- Handling outliers: remove, cap, replace
- Duplicates: detection and removal
- Text cleaning: strip, case, regex, extraction

**Key Skills:**
- `df.dropna()`, `df.fillna()`, `df.ffill()`
- `df.interpolate()` for time series
- `df.clip()` for capping outliers
- `df.drop_duplicates()`
- `.str.strip()`, `.str.replace()`, `.str.extract()`

---

### **[15-MINUTE BREAK]**

---

### 3. Case Study: Cleaning a Messy Dataset (~60 min)
`3_real_world_data_challenge.ipynb`

**Topics:**
- End-to-end walkthrough with messy financial data
- The diagnostic workflow: info, describe, visualize
- Creating and executing a cleaning plan
- Validating cleaned data
- Before/after comparison
- Documentation

**Key Skills:**
- Systematic approach to data cleaning
- Using visualization to find problems
- Making and documenting cleaning decisions
- Validating that cleaning worked

## Prerequisites

- Lecture 2: Pandas basics (Series and DataFrames)
- Basic Python proficiency

## Data

This lecture uses **generated data** to ensure:
- No API keys required
- Reproducible results
- Intentional data quality issues for learning

The generated data simulates stock OHLCV (Open, High, Low, Close, Volume) data with various issues: missing values, outliers, duplicates, and formatting errors.

## Quick Reference

### Visualization
```python
# Quick pandas plots
df['col'].plot()                     # Line chart
df['col'].plot(kind='hist', bins=50) # Histogram
df['col'].plot(kind='box')           # Box plot
df.plot(kind='scatter', x='A', y='B') # Scatter

# Matplotlib OOP style
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title('Title')
fig.savefig('chart.png', dpi=300)

# Pairwise relationships
from pandas.plotting import scatter_matrix
scatter_matrix(df, diagonal='kde')
```

### Data Cleaning
```python
# Missing values
df.info()                    # See non-null counts
df.isnull().sum()           # Count nulls per column
df.dropna()                 # Remove rows with nulls
df.fillna(value)            # Fill with specific value
df['col'].ffill()           # Forward fill
df['col'].interpolate()     # Linear interpolation

# Outliers
z_scores = (df['col'] - df['col'].mean()) / df['col'].std()
outliers = df[np.abs(z_scores) > 3]
df['col'].clip(lower, upper)  # Cap values

# Duplicates
df.duplicated().sum()       # Count duplicates
df.drop_duplicates()        # Remove duplicates

# Text cleaning
df['col'].str.strip().str.upper()
df['col'].str.replace(pattern, replacement, regex=True)
df['col'].str.extract(r'(\d+)')
```

### The Cleaning Workflow
1. **DIAGNOSE**: info(), describe(), value_counts(), visualize
2. **PLAN**: Document issues, decide strategies, determine order
3. **CLEAN**: Fix text first, then duplicates, then outliers, then missing
4. **VALIDATE**: Re-run diagnostics, compare before/after
5. **DOCUMENT**: Save clean data and cleaning log

## Further Reading

- [Pandas Visualization Documentation](https://pandas.pydata.org/docs/user_guide/visualization.html)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Pandas Working with Missing Data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
