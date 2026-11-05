# Slide-based teaching guide: November 5

For the current notebook delivery, use [presenter_notes.md](presenter_notes.md). This guide retains the slide-based route for the reference deck.

The class lasts **180 minutes, including two ten-minute breaks**. The main work is understanding tool execution and memory design. Students then use those ideas to explain recovery and evaluate a completed stock-research report.

The deck has 51 slides for the live route and six reference slides. The [lesson](lesson.md) develops the arguments in more detail and provides continuation exercises. Avoid reading the lesson aloud. Use the slides to pose a concrete question, have students predict the behavior, and inspect the corresponding Python trace.

## Route

| Time (PT) | Minutes | Slides | Work |
|---|---:|---|---|
| 9:00–9:15 | 15 | 1–5 | Research question, prior analysis, agent loop, responsibility boundaries |
| 9:15–9:55 | 40 | 6–17 | Tool contracts, validation, results, effects, concurrency, caching |
| 9:55–10:05 | 10 | 18 | Break |
| 10:05–10:55 | 50 | 19–33 | Memory lifecycle, dates, retrieval, context, concurrent edits |
| 10:55–11:05 | 10 | 34 | Break |
| 11:05–11:35 | 30 | 35–43 | Checkpoints, uncertain writes, budgets, plans, saved project state |
| 11:35–11:50 | 15 | 44–48 | Acceptance, reports, retrieval evaluation, module design |
| 11:50–noon | 10 | 49–51 | Paired design exercise, discussion, complete workflow |

**160 instructional minutes + 20 minutes of breaks = 180 minutes.** Slides 52–57 support questions about schemas, cache identity, memory records, context cost, and evaluation. They do not add to the live schedule.

## Outcomes

Students should be able to trace a tool call from a model response through validation and Python execution, explain the difference between a saved note and authoritative evidence, and design retrieval that respects ticker and date constraints. They should also be able to identify what a checkpoint can restore, explain why retries can duplicate effects, and evaluate a research report beyond whether it has the expected fields.

## Preflight

Run `uv sync --locked` in this folder, then `uv run python -m unittest discover -s tests -v`. Run scripts 01, 02, 05, 06, 07, and 08. Run script 08 again with `--rewrite`. Script 06 uses a temporary memory database and script 07 uses an in-memory database, so both can be repeated without cleanup.

Create fresh projects for the complete run, partial run, and recovery demonstration. Use new names after editing core code or inputs because projects record their hashes.

```sh
uv run python research.py init runs/preflight
uv run python research.py run runs/preflight --max-tasks 1
uv run python 04_inspect_project.py runs/preflight
uv run python research.py run runs/preflight
uv run python 03_resume_research.py runs/preflight-recovery
```

The partial run exits 2 because two tickers remain pending. The following run completes them. Repeating a completed project starts no new work.

If using a provider in class, set `ANTHROPIC_API_KEY`, select an available model ID, and initialize a separate project with `--engine model --model provider:model-id`. Run one ticker and inspect its evidence, note, and usage. Keep that local result for discussion. The offline demo exercises the same runtime with predictable model decisions, which is useful for the failure experiments.

For the course database, initialize a separate project with `--dataset course --as-of 2024-09-30`. The database must exist at `../Lecture 4/data/data.db`. Open its report before class and compare the observed dates with the requested window.

Have the deck, numbered scripts, `agent.py`, `tools.py`, `retrieval.py`, `runner.py`, and the example report open. Keep terminal text large enough to read. The lesson's 18 sections are reference material for the design discussion.

## Opening: 15 minutes

Spend about five minutes connecting the project to Lectures 2 and 4. Ask for the total return of prices 100, 110, 99. It is −1%. Ask why portfolio weights include cash and why actual shares use raw close while return calculations use adjusted prices. Run script 01 only as far as needed to reestablish those conventions.

Use the next ten minutes to trace the agent loop. A model proposes a function name and arguments. Python validates and executes them. A tool result joins the next request. Ask where a database write enters this sequence and whether writing a file automatically changes the next request's context. The answer establishes why retrieval and context construction belong in the harness.

Open `agent.py` briefly. Point out the agent, dependencies, capability list, output type, and supervisor. Ask who can mark the ticker complete. The application's validation and task state determine acceptance.

## Tools: 40 minutes

### Contract and granularity: 10 minutes

Run script 02, then inspect the real `search_sources` signature. Ask which parameters the model should choose. The query and result count are useful model choices. The ticker and cutoff come from the assigned project.

Compare three interfaces: unrestricted SQL, one function per arithmetic operation, and a coherent risk-analysis tool. Have students name the additional permissions and failure cases introduced by the SQL interface, and the extra requests introduced by tiny arithmetic tools. There is no universal correct tool size. The contract should match a decision the model can usefully make.

### Validation trace: 12 minutes

Run `05_tool_call_lifecycle.py`. Before executing it, show the three requested windows: 1, 12, and 3. Ask students to predict the number of function entries and actual calculations.

The schema rejects 1 before entering the function. The function receives 12 but raises `ModelRetry` because only 3 prices exist. The final call calculates −1%. There are **four model requests, two function entries, one calculation, and one successful tool call**. Match the response to each call ID.

Explain the accounting consequence. In the pinned SDK, the tool-call usage counter counts successful calls. Request limits still matter when calls repeatedly fail validation. The application dispatcher also reserves research actions before executing them, so its count answers a different question.

Ask which failures a model can repair. A malformed date or citation can receive precise feedback. A database outage needs a runtime policy. An empty search result and a failed service should not share the same result representation.

### Results, effects, and caching: 18 minutes

Show a result containing only `0.3029`, then the risk evidence with dates, units, convention, and row count. Ask what a source chunk loses if we separate a value from the “USD millions” table header.

Introduce whole-record output budgets using `pack_evidence`. Defer the full retrieval experiment until the memory block. Students should understand that omission is a result property the caller needs to see.

Use the write-then-timeout sequence on the slides. Ask whether retrying is safe before explaining stable operation keys. Save script 07 for the recovery block, when students can compare an uncertain write with a settled checkpoint.

Finish by distinguishing tool-call IDs, operation keys, and cache keys. Ask which inputs change a risk calculation's cache identity. Include window, price adjustment, input hash, and calculation version. Discuss why concurrent reads can overlap while shared note edits need coordination.

## Memory: 50 minutes

### Responsibilities and write policy: 15 minutes

Start with examples rather than database products. Ask students where to put an exact return, an open question, the current plan, and a task's attempt count. Map those records to the project's four SQLite stores using script 04 on the partial project.

Have students classify “volatility was 30.29%,” “management expects growth,” and “refinancing may become expensive.” They are an observation, an attributed claim, and a hypothesis. A working summary should preserve those distinctions.

Use the debt correction example: 100 published in February, corrected to 120 in May. An April question sees the original. A June question sees the correction with its history. Ask why deleting every superseded record would break historical research. Distinguish economic period, publication time, and ingestion time.

### Retrieval experiment: 15 minutes

Open the five excerpts in `data/retrieval_lab_sources.json`. Run script 06 through the retrieval portion. The highest two global matches belong to the future and to another ticker. Filtering those two produces nothing. Filtering before ranking returns the debt and cash excerpts.

Now compare the 600-character and 850-character budgets. The first returns one complete record and the second returns both. Count the metadata as part of the serialized response. Ask what narrower query could recover an omitted item.

Search for “rollover obligations.” The lexical baseline returns nothing. Discuss a domain vocabulary, query rewriting, full-text search, and embeddings as different possible changes. Require students to describe an evaluation before choosing one. A vector index still needs scope and date filters, and exact numbers still need their units.

### Context and concurrent edits: 20 minutes

Allocate a 24,000-token context on the board. Include instructions, tool definitions, history, evidence, answer reserve, and headroom. Ask what happens when a new 6,000-token excerpt arrives. Which material should remain exact? What can be summarized or retrieved later?

Inspect `SlidingWindowCompaction`. Explain that this configured capability trims history rather than generating a prose summary. The evidence store retains exact records, but the model needs a retrieval path to bring them back. A short working summary should retain the question, evidence IDs, unresolved issues, and next action.

Finish script 06. Both writers begin from the same memory version. A succeeds, B receives a version conflict, and B rereads before merging. Ask what happens if the two additions contradict each other. Concurrency control detects a lost-update risk; it does not resolve a research disagreement.

Close by considering namespace and retention. Project and ticker scope come from Python. Accepted report citations need retrievable evidence even if we later remove scratch notes or old transcripts.

## Recovery and longer work: 30 minutes

### Settled versus uncertain work: 12 minutes

Run `03_resume_research.py runs/class-recovery`. The first worker stops after a settled calculation. The second worker restores its result, so the calculation runs once. Find `resumed_from` in the event log and the new attempt ID.

Run script 07. The naive writer commits, loses its acknowledgement, and writes twice on retry. The idempotent writer stores its effect and receipt in one transaction and writes once. Ask why using a new operation key on every retry would defeat this design.

Compare the two traces. A checkpoint tells us which conversation state settled. The write receipt tells us whether a particular effect committed. For remote effects, a local SQLite transaction cannot make the remote service's commit atomic.

### Budgets and progress: 10 minutes

Inspect the persistent attempt count, `UsageLimits`, dispatcher reservations, and subprocess deadline. Ask which one bounds repeated schema failures, a hanging tool, repeated plan rewrites, and a task relaunched tomorrow.

Discuss multiplicative retries: two task attempts times three SDK retries times three tool HTTP attempts can allow eighteen underlying attempts. Each retry layer should have a clear responsibility. A timeout does not establish that an external write failed.

Define progress in research terms: new evidence, a resolved question, or an accepted result. A longer run should be able to stop with a clear unresolved issue instead of spending its full budget restating a plan.

### Plans and state inspection: 8 minutes

Show the model's plan beside the application's task table. A completed plan does not override invalid output. Continue the partial project and inspect it again. AAPL retains one attempt while the remaining tickers complete.

Discuss the saved input and code hashes. A new calculation definition warrants a new project or an explicit migration policy. Reusing messages from the old definition can produce inconsistent reasoning even if the Python import still succeeds.

## Reports and evaluation: 15 minutes

Spend five minutes following a report sentence to its citation. Check that the referenced evidence exists and then separately ask whether it supports the sentence. Use the historical-volatility/next-quarter-earnings example. Explain why an incomplete ticker remains visible and why portfolio weights need the whole holdings denominator.

Spend five minutes running script 08 with and without `--rewrite`. Mean recall is 0.75 and 1.0 on the four examples. Ask what this result establishes. It demonstrates repair of a known miss, so we need held-out questions to evaluate generalization. Include wrong-ticker and future-dated results as hard failures.

Use the remaining five minutes to design the leverage module on slide 48. Students should supply units, periods, publication dates, the zero/negative denominator policy, evidence fields, and completion checks before discussing another library.

## Paired exercise and close: 10 minutes

Give pairs five minutes on Exercise A or C from the lesson. Ask for a small record schema or a sequence of writes, not a general discussion. Use three minutes for two pairs to explain their designs and two minutes to locate the full application and report outputs.

Exercise discussion:

- **A:** Preserve both debt versions. Query eligibility and supersession relative to the research date. Publication time differs from the balance-sheet period.
- **B:** Evaluate original and held-out queries separately. Duplicate excerpts can increase apparent match count while reducing coverage of distinct evidence.
- **C:** A changing key creates two notes. The same key with different arguments must fail because it no longer identifies one consistent operation.
- **D:** Multiple allocations can work. Require an answer reserve and an explicit account of what information becomes less accessible.
- **E:** The stock returns 2%, the benchmark 1%, and the difference is 1 percentage point. Require compatible calendars and endpoints, with both source hashes in the cache identity.
- **F:** Schema and reference checks may pass. Historical price volatility does not establish future earnings growth. Revise the claim to describe observed risk or collect business evidence.
- **G:** Use repeated provider-backed runs under the same conditions. The fixed demo measures runtime behavior, not the quality benefit of planning.

If discussion runs long, leave cache design and the context-cost derivation in the reference slides. Preserve the tool validation trace, retrieval ordering experiment, concurrent memory edit, settled checkpoint, uncertain write, and citation-support distinction.
