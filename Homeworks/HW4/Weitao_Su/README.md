# HW4 - Weitao Su

Open [hw4.ipynb](hw4.ipynb) for the model, trading rules, results, findings, and checks. Saved results are included. INBX's unquoted contingent right is valued at $0.

From the course folder:

```powershell
uv sync --locked
uv run python Homeworks/HW4/Weitao_Su/run_project.py
uv run python -m unittest discover -s Homeworks/HW4/Weitao_Su -p test_accounting.py -v
```

- `hw4_data.py`, `backtester.py`, `run_project.py`: data, trading, and model code.
- `test_accounting.py`: accounting and timing tests.
- `corporate_actions.csv`: verified acquisition terms and source links.
- `results/`: one summary file and the chart. Detailed run logs go in the ignored `outputs/` folder.
