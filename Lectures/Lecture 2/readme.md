# Lecture 2

## Goals
- Use Cursor to work with Python interactively
- Use pandas to explore, update and join data

## Setup
### Setting up the environment
- From the project root, create a virtual environment and install dependencies with uv: `uv sync`
- The `.venv` directory will be created in the project root
- Register the kernel for Jupyter notebooks (from the project root): `uv run python -m ipykernel install --user --name=mfe-preprogram-2026-winter --display-name "Python (mfe-preprogram-2026-winter)"`

### Selecting the kernel in Cursor/VSCode
When opening a notebook in Cursor or VSCode, you need to select the Python interpreter from your virtual environment:

**Option 1: Select from registered kernel**
- Click on the kernel selector in the top right of the notebook
- Look for "Python (mfe-preprogram-2026-winter)" in the list

**Option 2: Manually browse to Python interpreter**
- Click on the kernel selector → "Select Another Kernel" → "Browse"
- Navigate to the project root and select: `.venv/bin/python`

**Option 3: Enter interpreter path directly**
- Click on the kernel selector → "Select Another Kernel" → "Enter interpreter path"
- Enter: `.venv/bin/python` (relative to project root) or the full absolute path to `.venv/bin/python` in the project root

### Tutorial
Check out `1_jupyter_tutorial.ipynb` - you can open and run this notebook directly in Cursor

## Pandas
Check out:
- `2_pandas_series_basics.ipynb`
- `3_pandas_dataframes_basics.ipynb`

## Data
- The Jupyter tutorial runs without credentials using the bundled historical AAPL sample in `data/aapl_sample.json`.
- For live API data, get an Alpha Vantage API key and set `AV_API_KEY` in your local environment.
