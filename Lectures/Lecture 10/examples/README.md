# Example research report

This report comes from the default Pydantic AI demo using `data/prices.csv`. The agent builds a plan, calls the research tools, saves working memory, and returns a typed note for each ticker.

Open `report.html` for the presentation. `report.md` contains the same research in text form, `summary.json` contains the numerical results and recorded usage, and `evidence.json` contains the full evidence payloads.

To reproduce it from the Lecture 10 folder:

```sh
uv run python research.py init runs/example
uv run python research.py run runs/example
```
