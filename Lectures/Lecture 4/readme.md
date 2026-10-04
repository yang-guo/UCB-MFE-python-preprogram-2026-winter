# Lecture 4: Advanced Data Processing and Automation

## Goals
- Apply outlier detection strategies for financial time series data
- Ingest data from CSV, JSON, and SQL databases
- Understand SQL fundamentals and pandas-SQL integration
- Automate repetitive analysis using parameterized notebooks

## Notebooks

### 1. Advanced Outlier Detection (~50 min)
`1_advanced_outlier_detection.ipynb`

Focuses on practical outlier detection for OHLC (price) data:
- Rolling Z-score for time series
- Domain-specific heuristics (OHLC integrity checks)
- Return-based detection
- Spike-and-reversal patterns
- Strategies for handling detected outliers

### 2. Data Ingestion and SQL (~60 min)
`2_data_ingestion_and_sql.ipynb`

Covers reading data from multiple sources:
- CSV files with various options
- JSON files including nested structures
- SQLite databases with raw SQL
- SQL fundamentals: SELECT, WHERE, GROUP BY, JOIN
- Pandas-SQL integration with `pd.read_sql()` and `df.to_sql()`
- Explores the `ohlc_hw` table used in Homework 2

### 3. Stock Analysis Template (~15 min)
`3_stock_analysis_template.ipynb`

A parameterized notebook designed for reuse:
- Loads OHLC data for any ticker and date range
- Calculates Yang-Zhang volatility
- Produces summary statistics and visualizations
- Exports results via scrapbook for aggregation

### 4. Running Analysis at Scale (~35 min)
`4_running_analysis_at_scale.ipynb`

Demonstrates automation with papermill and scrapbook:
- Running the template across multiple stocks
- Collecting and aggregating results
- Building comparative reports
- Multi-period analysis (quarterly)

## Data Files

- Download `data.db` from bCourses and place it in `Lectures/Lecture 4/data/` (it is intentionally gitignored). The SQLite database contains:
  - `ohlc` - Clean OHLC price data for 30 stocks
  - `ohlc_hw` - OHLC data with intentional issues for Homework 2
  - `market_caps` - Market capitalization data

## Prerequisites

Install required packages from the project root (`sqlite3` is part of Python):
```bash
uv sync --locked
```

## Homework Preparation

After completing this lecture, students will be ready for Homework 2, which involves:
- Loading data from `ohlc_hw` table
- Identifying and handling data quality issues
- Applying cleaning strategies learned in Notebook 1

