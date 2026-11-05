# Presenter run notes: building a stock-research harness

Teach from [1_stock_research_harness.ipynb](1_stock_research_harness.ipynb). The notebook carries the explanations and working code. Your job during the live session is to make students predict a behavior, observe the result, and explain the design decision behind it.

The central question is: **what does the application need to know before it can continue the research or accept the answer?** Return to that question when discussing tool arguments, memory, checkpoints, and report validation. It connects the components to the stock-research task instead of turning the class into a tour of APIs.

Every code cell starts with a stable label such as `T5 validation trace` or `M8 stale memory write`. Use notebook search to jump to a label. The timing below refers to those labels rather than execution counts, which change when you rerun cells.

## The three-hour flow

| Time (PT) | Minutes | Notebook | Live work |
|---|---:|---|---|
| 9:00–9:15 | 15 | Setup, §§1–2 | S0–S2, T1–T2: prior analysis and the model/tool loop |
| 9:15–9:55 | 40 | §§3–6 | T3–T6: scope, schemas, execution, validation, result design |
| 9:55–10:05 | 10 | First break | Leave the prediction about persistent state on screen |
| 10:05–10:55 | 50 | §§7–12 | M1–M6, M8–M9: stores, dates, retrieval, context, concurrent edits |
| 10:55–11:05 | 10 | Second break | Prepare the comparison between settled and uncertain work |
| 11:05–11:35 | 30 | §§13–15 | R1–R5: recovery, effect receipts, budgets, project completion |
| 11:35–11:50 | 15 | §§16–18 | P1–P3: report, claim support, retrieval evaluation, extension design |
| 11:50–noon | 10 | §20 | Paired exercise, discussion, locate the saved project |

**160 instructional minutes + 20 minutes of breaks = 180 minutes.** T7, M7, and X1–X5 are reference or extension cells. Their explanations remain in the notebook, and Run All executes the local examples. The course-database and provider branches stay off unless explicitly enabled.

The notebook contains more prose than you should read aloud. Students can use it before or after class. During class, summarize the next concept in one or two sentences, then work through its example. Keep the detailed policy discussions available for questions.

## Before class

From `Lectures/Lecture 10`, run:

```sh
uv sync --locked --group notebook
uv run --group notebook python check_notebook.py
uv run --group notebook jupyter lab
```

`check_notebook.py` executes all cells in a fresh kernel and saves the checked copy at `runs/notebook-check.ipynb`. It stops on an unexpected error. The intentional failures in the teaching examples are caught and explained in their cells. To refresh the primary notebook's saved output, pass `--output 1_stock_research_harness.ipynb`.

In JupyterLab, open the lecture notebook and check that S0 reports Pydantic AI Harness 0.65.0 and Pydantic AI 2.65.0. If using VS Code, explicitly select Lecture 10's `.venv`. A kernel from another lecture may import a different package version or lack the harness entirely.

Keep a second tab with the executed notebook as a fallback. Also open `research_harness/agent.py`, `tools.py`, `retrieval.py`, and `runner.py`. You will visit those files briefly when connecting the experiments to the application. Open `examples/report.html` so students can see the intended output before the full project completes.

Decide whether you want a provider-backed demonstration. The default live route works without it. If using one, configure the API key in the environment that launches Jupyter, choose an available model ID, and preflight X5 before class. Never display the key or the contents of `.env`. Retain that project's report and traces for comparison. The model may choose a different sequence from the deterministic example, so discuss the resulting evidence and acceptance checks rather than expecting identical wording.

For X4, place the bCourses database at `../Lecture 4/data/data.db`. The notebook does not need it for the main route.

### Preparing the notebook view

Use a font size that makes a code line readable from the back of the room. Keep the table of contents open and collapse long outputs after discussing them. T5's trace and P1's full report are the main places where output can occupy several screens.

Execute S0 once when starting the session. It creates a fresh `runs/notebook-...` directory. Rerunning S0 changes the root for later experiments, so avoid doing so casually halfway through the class. Restart and Run All is safe because the setup and individual projects choose new names. To rerun a multi-cell experiment, start at its first cell: T4 before T5, M8 before M9, and R2 before R3.

The notebook uses `await agent.run(...)` and awaited memory operations. Keep those calls as written. Calling `asyncio.run()` or `agent.run_sync()` inside the notebook's running event loop is a common source of confusion. The process-boundary demonstrations deliberately use `run_script`, which launches this kernel's Python in a separate process.

## 9:00–9:15: connect the research question to an agent loop

Open with the report and ask: “What would you need before you were comfortable with these weights and this risk comparison?” Students should mention prices, holdings, cash, dates, risk definitions, and evidence. Add the operational question: “If the process stops after AAPL, how do we know what remains?”

Run S0. Spend about three minutes on S1. Ask for the two daily returns and the whole-period return before execution. The answers are +10%, −10%, and −1%. Maximum drawdown is −10% because it compares the final price with the intermediate peak. This restores a calculation students already understand from Lecture 4.

Run S2 and point to the cash weight. The stock weights plus cash equal one. Explain that dropping a holding from an incomplete research report would change the denominator in the same way as dropping cash. We will return to that rule when the report is built.

Run T1 and T2. Ask students to point to the line that **proposes** a tool call and the line that **calculates** the result. Follow `return-1` through the trace. Python executes the function; the model sees its returned observation in the following request.

Call out that the scripted model makes the decisions predictable while the real Pydantic runtime performs dispatch and validation. This lets us observe the mechanism. It does not tell us whether a live model will choose a good investigation.

Close the opening with: “We now have a model that can ask for a calculation. The next question is how much freedom that interface should give it.”

## 9:15–9:55: tool design and execution

### First ten minutes: scope and the contract

Run T3. Before showing the schema, ask which fields should belong to the model. Students should identify the query as a useful choice. The assigned ticker and research date already come from the application. Show that the generated schema contains `query`, while `RunContext` supplies the other values.

You can change the dependency ticker from AAPL to MSFT and rerun T3 to make the boundary visible. Restore AAPL afterward if you want the displayed result to match the prose. Later experiments use their own fixed scope, so this edit does not alter them.

Discuss the three tool sizes in §3: unrestricted SQL, a tool for every arithmetic step, and a coherent risk calculation. Ask what extra controls the SQL tool needs and what extra model requests the tiny arithmetic tools create. The point is to choose a useful research action and define its contract.

Open the full `search_sources` function in `agent.py` briefly. The production version adds bounded query length, result limits, persistent evidence, and active-task checks through the dispatcher. Return to the notebook before the implementation detail consumes the discussion.

### Next twelve minutes: prediction and validation trace

Read T4 selectively. Point to the schema range and then to `entered.append`. Ask why the requested window of 1 cannot reach that line. Point to the semantic check and ask why a window of 12 can reach it but cannot calculate a return.

Before running T5, collect predictions for:

- Model requests: **4**, including the final answer.
- Python function entries: **2**, for windows 12 and 3.
- Completed calculations: **1**.
- Successful SDK tool calls: **1**.

The prose already contains the explanation, so keep the viewport on the prediction and code until students answer. Run T5 and follow the three call IDs. Show the difference between schema feedback and the explicit `ModelRetry` message.

Emphasize why the counters matter. A successful-tool-call limit does not stop unlimited invalid proposals on its own. Request limits bound those proposals. The application dispatcher also reserves research actions before execution, which measures something different from the SDK counter.

Ask which failure the model can repair. A valid smaller window or a real citation can resolve a request error. Rephrasing the question cannot restore an unavailable database. An empty search result should not hide a service outage.

### Final eighteen minutes: results, effects, and concurrency

Run T6 and first draw attention to the bare value. Ask what it could mean without its label. Then inspect the definition, observed dates, and row count. Connect raw close to valuing actual shares and adjusted close to returns. Ask what happens when an extracted filing value loses its “USD millions” header.

Use the result-envelope example to explain bounded responses. A whole-record budget preserves identifiers and context. The omission count gives the caller a reason to narrow the query. The next block will demonstrate this with actual excerpts.

Discuss the write-then-timeout sequence without running R2 yet. Ask whether the timeout tells us the note failed to save. Leave the answer as a recovery requirement: the caller needs a stable identity and a way to discover the prior effect.

Differentiate a tool-call ID, an operation key, and a cache key. Use T7 only if students need a concrete cache example. Its stable JSON hash changes with the date window but stays the same when dictionary field order changes.

End with parallel execution. Independent reads may overlap. Shared note writes need conflict handling, and dependent calculations must wait for their inputs. Async code permits overlap; it does not decide whether the operations are safe to overlap.

## 10:05–10:55: memory and retrieval

### First fifteen minutes: what survives and why

Start with four examples: an exact return, an open refinancing question, a planned next step, and the current attempt count. Ask students where each should live. Then run M1 and M2 to show the actual stores.

M1 leaves AAPL complete and the other tickers pending. The incomplete CLI exit code is **2**, which `run_script` explicitly accepts. M2 shows the evidence IDs, plan, working note, and checkpoint. Ask whether a memory sentence saying “AAPL is done” could replace the accepted task record. It cannot.

Discuss the write policy before storage products. An observation, an attributed management claim, and a hypothesis need different treatment. A summary that removes those distinctions can turn an unresolved idea into an apparent fact on the next request.

Run M3. Ask what an April investigation can know about the debt figure and what changes in June. The values are **100** and **120 USD million**. The economic period remains December 31. Publication controls eligibility, while ingestion describes our system's history. Globally deleting superseded records would erase the April view.

### Next fifteen minutes: retrieval failures students can see

Run M4. Read the corpus table first. Ask students to identify the future AAPL document and the wrong-ticker document. The global top two are `aapl-future` and `msft-debt`. Filtering only those records leaves nothing. Filtering before ranking returns `aapl-debt` and `aapl-cash`.

Have a student explain why simply increasing the global top-k is not the general fix. The scope restriction belongs before selection. Similarity does not establish that a record is eligible.

Run M5. The 600-character budget returns one complete record. The 850-character budget returns both. The serialized count includes metadata. Ask what the model should do when `omitted` is nonzero. Narrowing the query is one possible response, although the appropriate response depends on what evidence is missing.

The synonym query returns zero matches. Ask whether that establishes that the company has no debt. Discuss lexical search, query rewriting, and embeddings as possible retrieval policies. Keep exact values and metadata available regardless of the search method. Ask how students would evaluate a proposed improvement before revealing P3 later.

### Final twenty minutes: context and shared memory

Run M6 and allocate the 24,000-token design target. Ask pairs to make room for a 6,000-token excerpt. Require a tradeoff tied to the question, including what information becomes harder to retrieve. Preserve an answer reserve and headroom.

Explain the configured sliding window. It trims conversation history and preserves valid tool-call/result relationships. It does not generate a summary. A working summary should retain the question, supported conclusions with evidence IDs, unresolved issues, and next action. Exact records remain in the evidence store, but they influence a later request only if retrieved.

M7 provides the optional cost derivation: eight requests use 74,000 versus 46,000 input tokens under the two growth assumptions. These are token counts, not current provider prices. Leave the derivation for reading if time is tight.

Run M8 and stop before M9. Both writers started with the same version. A's write advances it, and B's stale write fails. Ask what B must do next. Run M9 to reread and merge the compatible additions.

Ask a harder follow-up: “What if A and B disagree about whether debt rose?” The store prevents a silent lost update; it does not resolve the financial disagreement. Dates, definitions, and provenance still matter.

Finish with scope and retention. The application supplies the namespace. A shared service also needs authenticated user or organization scope. Published report citations need their evidence to remain retrievable after scratch notes or transcripts expire.

## 11:05–11:35: longer projects and recovery

### First twelve minutes: compare two interruption points

Run R1. Predict **two attempts and one calculation**. The new worker restores a settled result. Show the `resumed_from` value in the event table. Then ask what would happen if evidence committed before the tool result settled: the call might repeat, while its stable evidence ID would reuse the record.

Run R2. Two calls produce **two notes** because the commit happened before each timeout. Ask students to identify where the effect became durable.

Read R3's transaction. The note and operation receipt commit together. Retrying with the same key and arguments returns the original note ID, leaving **one note**. Reusing the key with different arguments raises `ValueError`.

Keep the distinction explicit: the checkpoint preserves conversation progress, while the receipt identifies a committed effect. A local transaction cannot make an external service's write atomic with our receipt. A durable workflow engine still needs an effect policy at that boundary.

### Next ten minutes: budgets and progress

Run R4. The agent successfully pings twice, then the request budget stops the third request. Each ping is valid, but it adds no research evidence. Ask how a longer project could recognize that lack of progress before spending the entire budget.

Walk through the application budgets: persisted attempts, model requests, successful tool calls, dispatched research actions, and the worker deadline. Ask which stops repeated schema failures, an unresponsive function, and repeated relaunches tomorrow.

Use the multiplication example to discuss retry ownership: two task attempts times three SDK attempts times three tool HTTP attempts could allow eighteen underlying attempts. Backoff belongs with transient service failures and must fit the overall deadline. Repeated retries do not repair a deterministic data error.

### Final eight minutes: acceptance and continuation

Return to the distinction between the plan and task state. A model can mark its plan complete before its final answer passes validation. The application's acceptance contract still applies.

Run R5 to complete the portfolio project. Run it again as the cell does and compare task tables. All tickers are complete with one attempt, and completed work does not repeat. Point out the frozen input and code hashes. After changing a tool's meaning, a new project avoids mixing old messages with new calculations.

## 11:35–11:50: the report and its evaluation

Run P1 and inspect the risk table and one accepted note. Follow its citation into the evidence appendix. Identify actual coverage and the portfolio denominator. Collapse the long report output after the discussion.

Run P2. The citation checks pass for an unsupported future-earnings claim. Ask what those checks actually established: the IDs exist, belong to JPM, and include the required evidence kinds. They did not establish that historical volatility predicts next quarter's earnings. The example does not save that note into the project.

Run P3. Mean recall@2 rises from **0.75 to 1.0** after one explicit rewrite. Ask what this result proves. We repaired a known miss on the same small set. We need held-out questions, wrong-date and wrong-ticker cases, duplicates, and conflicts to evaluate the policy more broadly.

Use the last few minutes to design an additional module. Section 18 discusses leverage. X1 and X2 provide a complete benchmark example students can run afterward: stock return **2%**, benchmark return **1%**, difference **1 percentage point**, and an explicit failure for mismatched calendars. The model requests the comparison while Python supplies the price series. A production integration would also persist its evidence and extend the report's acceptance rules.

## 11:50–noon: paired design and close

Give pairs five minutes on Exercise A or C in §20. Ask for a record schema or a sequence of writes, rather than a general recommendation. Use three minutes for two pairs to explain their choices and one failure their design prevents.

For A, expect both versions, a reporting-period key, publication dates, and a correction relationship applied relative to the question date. For C, expect a stable intended-operation key and rejection when the same key has different arguments.

Use the final two minutes to locate the project directory printed in S0. Show the report, evidence, logs, and SQLite stores. The Python application can continue outside the notebook, and the notebook can inspect what it saved.

## If time is short

Keep T5, M4–M5, M8–M9, R1–R3, and P2. Together they expose schema versus semantic validation, retrieval eligibility, response budgets, concurrent edits, settled checkpoints, uncertain writes, and claim support.

Move T7's cache implementation, M7's token-cost derivation, and X1–X5 to independent practice. Summarize the storage-method comparison instead of discussing each product category. Use the saved P1 report rather than rerunning an entire project if the local environment is slow. Preserve the distinction between a valid citation and a supported claim even if the final extension discussion is brief.

## Recovery during a live demonstration

| Symptom | What to do |
|---|---|
| Import error or wrong package version | Select Lecture 10's `.venv` and restart at S0 |
| “Event loop already running” after an edit | Restore top-level `await`; remove an added `asyncio.run()` or `run_sync()` |
| A dependent variable is missing | Restart that experiment at its first labeled cell, or run prior cells in order |
| M1 returns an incomplete report | Expected: two tickers remain pending; continue at R5 later |
| Input or code hash changed | Create a new project by rerunning M1 or R1 with its fresh path |
| R1's project fails to recover | Show the saved executed notebook and inspect the worker log after class |
| Provider demonstration fails or stalls | Keep its task status visible and continue with the deterministic trace |
| Long output hides the next explanation | Collapse the output after inspecting the relevant rows |

If a result differs from the expected trace, make the difference explicit and inspect the relevant boundary. A failed demonstration can still teach the design if students can identify which assumption changed.
