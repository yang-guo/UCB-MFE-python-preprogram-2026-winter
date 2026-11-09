# November 5: Building a small stock-research harness

The [lecture notebook](1_stock_research_harness.ipynb) introduces a harness built with Pydantic AI. It connects a model to the SQL and pandas work from Lectures 2 and 4, adds a follow-up conversation, and saves a report comparing three stocks.

## Setup

From the repository root:

```sh
uv sync --locked
```

Open the notebook and select the repository's `.venv` kernel. Restart the kernel after installing packages. The notebook runs from the repository root and also handles editors that initially start the kernel in the notebook folder.

The examples use the bCourses database at `Lectures/Lecture 4/data/data.db`. They load model credentials from a local `.env` file at the repository root. The default model is `anthropic:claude-haiku-4-5-20251001`, which uses `ANTHROPIC_API_KEY`.

The OpenAI alternative is `openai:gpt-4.1-mini`, using `OPENAI_API_KEY`. The model can be changed in Section 3 or selected through `RESEARCH_MODEL` in the root `.env`. Only the selected provider's key is needed. Both providers are included in the root environment. Changes to the model take effect when the agent-creation cell runs again; changes to `.env` take effect after restarting the kernel. Agent runs make live API calls to the selected provider.

## Files

| File | Purpose |
|---|---|
| [1_stock_research_harness.ipynb](1_stock_research_harness.ipynb) | Explanations, diagrams, and runnable examples |
| [classroom_support.py](classroom_support.py) | Displays agent messages as a readable table |
| `images/` | The three diagrams used in the notebook |
| [in depth/](in%20depth/readme.md) | The earlier detailed slides, available as optional reference material |

The notebook contains the database query, return calculation, agent, tool registration, conversation history, research loop, and report-writing code. The `my_question` cell in Section 7 accepts questions in ordinary language; its message table shows which tools the model requested.

Sections 5–6 generate `Lectures/Lecture 10/runs/intro/stock_report.md`. Changing the watchlist or dates and rerunning those sections replaces the report with the new results. The `runs/` folder is created as needed and excluded from Git.
