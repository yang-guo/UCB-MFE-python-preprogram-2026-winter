# November 5: A stock-research harness

Yang · 9am–noon Pacific · final lecture, Special Topics.

We want to investigate a small equity portfolio, keep the evidence we collect, and finish a report even when the work spans several runs. Lectures 2 and 4 supply the calculations. In this class, we turn them into tools and build a research application with [Pydantic AI Harness](https://pydantic.dev/docs/ai/harness/).

Teach from [the lecture notebook](1_stock_research_harness.ipynb), which combines the full explanations with executable examples. The [presenter run notes](presenter_notes.md) give the three-hour flow, cell labels, prediction questions, expected results, transitions, and fallback options. The Python package runs the same research application outside Jupyter. The [lesson text](lesson.md) and [slide deck](slides/UCB_MFE_Lecture_10_Stock_Research_Harness.pptx) remain available as references.

## Setup

From the repository root:

```sh
cd "Lectures/Lecture 10"
uv sync --locked --group notebook
uv run --group notebook jupyter lab
```

This lecture has its own Python environment and lockfile. It uses Python 3.12 or newer, Pydantic AI Harness **0.65.0**, and Pydantic AI **2.65.0**. Run the following commands from this folder. `uv run` selects the lecture's environment on macOS, Windows, and Linux.

## Notebook route

Open `1_stock_research_harness.ipynb` and select Lecture 10's `.venv` as the kernel. Run from top to bottom. The core examples use the actual Pydantic agent with predictable model responses and need no credentials. Each execution creates fresh project directories under `runs/`.

The notebook contains 32 executable cells. It traces tool calls, validates arguments, compares retrieval policies, handles dated corrections and concurrent memory edits, resumes research, and builds the final report. The benchmark extension also registers a working tool. Course-database and provider-backed runs are separate opt-in branches at the end.

To execute the notebook from a fresh kernel and save a checked copy:

```sh
uv run --group notebook python check_notebook.py
```

The result goes to `runs/notebook-check.ipynb`. Use `--output 1_stock_research_harness.ipynb` to refresh the lecture's saved outputs. This uses [NBClient](https://nbclient.readthedocs.io/en/latest/client.html) and stops on unexpected cell errors. For the live session, follow the labeled cells in the [presenter notes](presenter_notes.md). They identify which reference examples to leave for practice if discussion runs long.

## First run from a terminal

```sh
uv run python 01_data_and_tools.py
uv run python 02_agent_and_tools.py
uv run python research.py init runs/first
uv run python research.py run runs/first --max-tasks 1
uv run python 04_inspect_project.py runs/first
uv run python research.py run runs/first
```

Open `runs/first/report.html`. The project also writes `report.md`, `summary.json`, and `evidence.json`.

The first invocation leaves two tickers pending and exits with code **2**, meaning the report is incomplete. The next invocation resumes them. A complete project exits with code **0**. Invalid setup exits with code **1**.

The default `demo` engine supplies predictable model responses through Pydantic AI's `FunctionModel`. The real agent executes the tools, updates its plan and memory, saves checkpoints, and validates the final note. It runs without credentials. To see recovery within one investigation, run:

```sh
uv run python 03_resume_research.py runs/recovery
uv run python 04_inspect_project.py runs/recovery
```

The demonstration interrupts AAPL's research just after calculating its metrics. A fresh worker loads the saved conversation and finishes the note. The calculation runs once across the two attempts.

## Experiments on the fundamentals

```sh
uv run python 05_tool_call_lifecycle.py
uv run python 06_memory_and_retrieval.py
uv run python 07_retries_and_effects.py
uv run python 08_evaluate_retrieval.py
uv run python 08_evaluate_retrieval.py --rewrite
```

Predict each result before running it. The first trace distinguishes model requests, function entries, and successful calls. The memory experiment exposes retrieval mistakes and a stale concurrent edit. The retry experiment shows why a timeout can produce duplicate writes. The evaluation script measures a known retrieval miss and its repair.

The expanded lesson covers tool granularity and validation, side effects and caching, memory write policies, publication dates and corrections, retrieval and chunking, context allocation, checkpoint boundaries, retry budgets, and claim support. The three-hour teaching route gives tool and memory design 90 minutes, with recovery and evaluation built on those foundations.

## Use a model provider

The environment includes the Anthropic provider. Set `ANTHROPIC_API_KEY` in your terminal and choose an available model ID. For example, on macOS or Linux, replace the two placeholders before running:

```sh
export ANTHROPIC_API_KEY="your-api-key"
export RESEARCH_MODEL="anthropic:your-model-id"
uv run python research.py init runs/model --engine model --model "$RESEARCH_MODEL"
uv run python research.py run runs/model --max-tasks 1
uv run python research.py run runs/model --seconds 1800
```

In PowerShell, use `$env:ANTHROPIC_API_KEY = "your-api-key"` and `$env:RESEARCH_MODEL = "anthropic:your-model-id"`, then pass `--model $env:RESEARCH_MODEL`. The [provider guide](https://pydantic.dev/docs/ai/models/anthropic/) describes model configuration. Other providers can be added through the corresponding Pydantic AI extras.

The provider-backed run uses the same tools, capabilities, stores, and report checks as the demo. The model chooses its own tool sequence and writes the research notes. `04_inspect_project.py` shows the saved plan, working memory, and checkpoints. Each attempt also has a `.log` file.

## What we build

The agent combines four harness capabilities with four research tools and a typed output:

| Component | Role in the project |
|---|---|
| `Planning` | Keep an editable research plan for each ticker |
| `Memory` | Save working notes and follow-up questions across sessions |
| `StepPersistence` | Save agent messages and tool events for continuation |
| `SlidingWindowCompaction` | Trim older conversation context when it grows |
| Research tools | Calculate metrics, read or search dated sources, recall exact evidence |
| `ResearchNote` | Return a body and citation IDs for validation and storage |
| Python supervisor | Track the ticker queue, deadlines, attempts, and reports |

Plans live in `plans.sqlite`, working notes in `notebook.sqlite`, and agent checkpoints in `steps.sqlite`. Our application stores exact evidence, accepted notes, task status, and usage receipts in `memory.sqlite`.

The supervisor resumes a failed attempt from its latest settled checkpoint. A tool interrupted before its checkpoint may run again. Evidence writes reuse their IDs, and accepted notes replace the ticker's prior note. The lesson walks through these commit boundaries.

## Use the course database or another source corpus

```sh
uv run python research.py init runs/course --dataset course --as-of 2024-09-30
uv run python research.py run runs/course
```

This reads the `ohlc` table in `../Lecture 4/data/data.db`. Download the database from bCourses if needed. The adapter uses `ticker`, `ts`, `close`, and `adj_close`, with holdings from `data/holdings.csv`.

The default sample uses generated prices for AAPL, MSFT, and JPM, with 86 weekday observations each. See the [data guide](data/README.md). To add filing excerpts or other research documents, pass `--sources path/to/sources.json`. Each record has this structure:

```json
{
  "ticker": "AAPL",
  "title": "A descriptive document title",
  "available_on": "2024-02-02",
  "source_type": "issuer filing excerpt",
  "locator": "Original document URL and section or page",
  "text": "A verified excerpt, with its units and context."
}
```

Store the records in a JSON list, using publication dates for `available_on`. Each excerpt can contain at most 8,000 characters. `read_source` returns the five most recent eligible records. `search_sources` filters by ticker and publication cutoff before ranking by query-term overlap, then returns whole citable records within a 12,000-character budget. Its result includes matched and omitted counts. The retrieval experiments examine the limits of this lexical baseline.

`--data`, `--holdings`, `--start`, `--as-of`, `--tickers`, and `--question` select the other inputs. The sample adapter expects CSV columns `ticker,date,close,adj_close`. The course adapter expects the SQLite schema above. Start a new project directory when inputs or code change. Saved projects use absolute source paths, so recreate the project when moving to another computer.

## Code map

| File | Responsibility |
|---|---|
| `1_stock_research_harness.ipynb` | Full lecture with prose, executable experiments, and report generation |
| `presenter_notes.md` | Three-hour notebook flow, expected results, prompts, and recovery notes |
| `check_notebook.py` | Execute every cell in a fresh kernel and save the output |
| `01_data_and_tools.py` | Return arithmetic and cash-aware portfolio aggregation |
| `02_agent_and_tools.py` | One typed Python tool and one structured agent output |
| `03_resume_research.py` | Interrupt research after a calculation, then resume |
| `04_inspect_project.py` | Inspect evidence, plans, memory, checkpoints, and usage |
| `05_tool_call_lifecycle.py` | Trace schema rejection, semantic repair, and successful execution |
| `06_memory_and_retrieval.py` | Compare retrieval order, context budgets, and concurrent memory edits |
| `07_retries_and_effects.py` | Retry a write after losing its acknowledgement |
| `08_evaluate_retrieval.py` | Measure retrieval recall and inspect a synonym miss |
| `research.py` | Create, run, inspect, and report a project |
| `research_harness/agent.py` | Compose capabilities, register tools, validate output, load history |
| `research_harness/demo_model.py` | Predictable model responses for teaching and tests |
| `research_harness/data.py` | Read-only data access, risk calculations, portfolio exposure |
| `research_harness/tools.py` | Evidence storage, source timing, citation checks |
| `research_harness/retrieval.py` | Scope filters, lexical ranking, whole-record response budgets |
| `research_harness/memory.py` | Application tables, events, and process lock |
| `research_harness/worker.py` | Python worker with an enforceable deadline |
| `research_harness/runner.py` | Task queue, recovery, retry policy |
| `research_harness/report.py` | Tables, accepted research notes, status, evidence appendix |
| `research_harness/instructions.txt` | Research question handling and writing instructions |

## Checks and budgets

```sh
uv run python -m unittest discover -s tests -v
```

The tests exercise the installed Pydantic agent loop, output repair, persistent capability stores, recovery in a fresh worker, execution limits, and the financial calculations.

Defaults are two attempts per ticker, twelve model requests and twenty successful agent tool calls per attempt, and a 120-second worker deadline. The successful-call limit includes planning and memory tools. Schema failures still consume model requests. The application separately reserves dispatched research actions, including actions that fail after dispatch. `run --seconds` limits one supervisor invocation. Attempt counts persist across invocations. The step store records model and tool events, and the application saves usage when an attempt returns or raises an exception. A forcibly terminated worker may lack that final usage receipt.

Generated projects stay in the ignored `runs/` directory. The [example report](examples/report.html) shows the completed default demo.
