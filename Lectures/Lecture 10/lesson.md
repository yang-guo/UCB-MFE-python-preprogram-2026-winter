# Building a stock-research harness

In Lectures 2 and 4, we wrote Python that could turn holdings and prices into a useful stock report. We knew which calculations to run, which inputs they needed, and what the output should look like. This class adds a model to that workflow. The model can choose what to investigate next, call our functions, inspect the results, and write an interpretation.

That flexibility creates some new questions. What can the model ask a tool to do? What should it remember between requests? Which source should it retrieve when several documents describe the same company? What happens if a process stops halfway through a write? How do we decide that the research is finished?

We will work through those questions using Pydantic AI Harness. By the end, we will have a Python application that researches several holdings, keeps evidence and progress on disk, resumes interrupted work, and produces a report. The numbered scripts let us examine individual mechanisms before putting them together.

## 1. What happens inside an agent loop?

A model request contains the context available at that moment: instructions, conversation messages, tool definitions, and any retrieved material we have included. The response may contain an answer, a tool call, or several tool calls. A tool call is a request to execute a function with particular arguments.

For example, the model might produce this request:

```json
{
  "tool_name": "search_sources",
  "args": {"query": "debt refinancing", "limit": 2},
  "tool_call_id": "call_17"
}
```

The application validates the arguments, runs the function, and sends the result back with the same call ID. That association matters when a response contains several calls. The next model request can use the returned evidence to choose another action or produce a final answer.

| Stage | What happens | A failure we need to handle |
|---|---|---|
| Construct context | Select instructions, history, schemas, and evidence | Relevant information never reaches the model |
| Request a response | Call the model provider | Timeout or rate limit |
| Validate a call | Check the function name and argument schema | Missing field or invalid range |
| Execute Python | Check scope, access data, calculate or write | Missing data or a partially completed effect |
| Return an observation | Serialize a result with its call ID | Output is too large or loses its units |
| Continue or finish | Make another request or validate the final output | Repeated work or an unsupported conclusion |

We can describe the loop as a sequence of contexts, actions, and observations. At step `t`, the model sees `context_t` and proposes `action_t`. The runtime executes the action and obtains `observation_t`. Our context policy decides which parts enter `context_(t+1)`.

This last step is easy to overlook. Saving a file does not make its contents available to every future model request. Something must retrieve the file and include its contents. Persistent memory changes future inputs to the model; it does not update the model's trained parameters.

Run `02_agent_and_tools.py` to see the smallest example. Its `FunctionModel` produces predictable responses, but Pydantic AI still performs the tool dispatch and output validation. This lets us study the runtime without variability in the model's decisions. Later, `--engine model` uses a provider-backed model with the same application code. The [Pydantic AI tool documentation](https://pydantic.dev/docs/ai/tools/) describes how Python functions become tool definitions.

## 2. Where the harness ends and our application begins

Pydantic AI runs the model/tool loop. Its harness capabilities provide planning, working memory, conversation persistence, and context management. Our application supplies the financial calculations, source corpus, task queue, acceptance checks, and report.

In `research_harness/agent.py`, we compose the agent with a list of capabilities. Here is a shortened version of that configuration:

```python
capabilities = [
    Planning(store=plan_store, tools=["write_plan"]),
    Memory(memory_store, namespace=ticker, max_tokens=1200),
    SlidingWindowCompaction(max_tokens=24000, keep_tokens=12000),
    StepPersistence(store=step_store, run_id=attempt_id),
]
```

Each capability has a particular job. A saved plan helps the model organize an investigation. Working memory retains useful notes. A checkpoint helps the runtime restore a conversation. None of these, by itself, establishes that a ticker has a valid research result.

The supervisor owns that decision. It assigns a ticker, reserves an attempt, starts a worker, and checks for an accepted note. The model may mark every item in its plan complete, but the task remains incomplete if its final citations fail validation. Keeping those responsibilities separate gives us a clear answer to “who decides that the work is done?” See the [harness overview](https://pydantic.dev/docs/ai/harness/) for the capability interfaces.

## 3. Designing a tool contract

A useful tool interface expresses an action the model can choose while leaving application constraints in Python. Our search tool accepts a query and a small result limit:

```python
@agent.tool
def search_sources(
    ctx: RunContext[ResearchDeps],
    query: Annotated[str, Field(min_length=1, max_length=200)],
    limit: Annotated[int, Field(ge=1, le=5)] = 3,
) -> dict:
    d = ctx.deps
    return dispatch(d.run, d.ticker, d.attempt, "search_sources",
                    {"query": query, "limit": limit})
```

The model controls the search terms. Python supplies the project, assigned ticker, and attempt through `ResearchDeps`. The project's configuration supplies the research date. We do not need the model to repeat these constraints correctly on every call. `RunContext` carries application dependencies and does not appear as a model-supplied parameter in the tool schema. See [dependencies](https://pydantic.dev/docs/ai/dependencies/).

Before implementing a tool, write its contract in ordinary language:

| Decision | Example for `search_sources` |
|---|---|
| Purpose | Find excerpts relevant to a particular research question |
| Model arguments | Query of 1–200 characters, result limit of 1–5 |
| Application scope | Current project, assigned ticker, fixed publication cutoff |
| Preconditions | Input hashes still match the saved project |
| Result | Whole evidence records, citation IDs, matched and omitted counts |
| Side effect | Save returned evidence and a retrieval event |
| Retry behavior | Identical evidence reuses its content-derived ID |
| Completion meaning | A search result supplies evidence but does not complete research |

Tool granularity is a design choice. A general `run_sql(sql)` tool gives the model substantial flexibility, but requires database permissions, query limits, and a careful definition of allowed data. A separate tool for every arithmetic operation makes the model spend requests orchestrating calculations we already know how to perform. Here, `analyze_prices` performs a coherent calculation and returns its conventions and coverage. The model chooses when to request that analysis and how to interpret it.

Descriptions matter as well as types. “Search documents” gives little guidance. “Search eligible excerpts for the assigned ticker using lexical overlap” tells the model both the scope and the search behavior. Describe empty results, truncation, and units explicitly. Those details affect what the model should do next.

## 4. Validation happens at several boundaries

A type annotation handles only part of a tool's contract. We need to distinguish four checks:

1. **Schema validation:** Are the required arguments present? Do their types and ranges satisfy the schema?
2. **Semantic validation:** Is this request meaningful and permitted for this project? Is there enough data for the requested window?
3. **Result validation:** Are the values finite? Are the dates, units, and identifiers intact?
4. **Completion validation:** Does the final research note cite the required evidence for the assigned ticker?

Pydantic can coerce some inputs to the requested type unless we choose strict validation. A value reaching a Python function has passed the configured schema, but we still need the semantic checks inside the function. A schema is also not an authorization system. The application must enforce data scope independently of what the prompt asks the model to do.

Run `05_tool_call_lifecycle.py`. The model requests the same calculation three times:

| Requested window | Schema | Function entered? | Result |
|---|---|---|---|
| 1 observation | Fails the minimum of 2 | No | Validation feedback |
| 12 observations | Passes | Yes | `ModelRetry`: only 3 observations exist |
| 3 observations | Passes | Yes | Total return of −1% |

The trace shows the original call IDs beside the responses. The calculation runs once, although the Python tool function is entered twice. The model makes four requests: three tool requests followed by a structured answer.

There is a useful accounting detail here. In the pinned SDK, `RunUsage.tool_calls` counts **successful** tool calls. This experiment therefore records four requests and one successful tool call. A tool-call limit alone would not bound repeated invalid requests. We also set a request limit, and our application ledger reserves dispatched research actions before executing them.

Use `ModelRetry` when the model can repair its request or output from the feedback. An unsupported window or unknown citation is a good example. A database outage usually calls for a runtime retry or a failed task, since changing the wording of the query will not restore the database. Returning an empty list for both “no matches” and “service unavailable” would hide a distinction the research process needs.

## 5. Designing the result the model will read

Consider a tool returning `0.3029`. We cannot tell whether this is a return, a volatility, or a portfolio weight. Even “volatility: 0.3029” leaves open the frequency, price adjustment, date window, and estimator.

Our risk evidence includes the assigned ticker, requested and observed dates, row count, and calculation conventions. Annualized volatility uses the sample standard deviation of daily log returns multiplied by the square root of 252. Total return uses adjusted endpoint prices. Portfolio valuation uses raw close because it values the shares we actually hold. These choices reproduce the analysis from Lectures 2 and 4.

Source excerpts need comparable care. If a filing table says “USD millions,” that header belongs with the extracted values. A chunk containing “Debt: 100” without the header can create an error of a millionfold. A useful excerpt carries the company, reporting period, publication date, document locator, and enough surrounding text to interpret the number.

Large results create another problem. Sending every document on every call increases latency and consumes context. Cutting a serialized JSON string halfway through a record can remove a citation ID, unit, or closing delimiter. Our `pack_evidence` function includes whole records within a character budget and returns `matched`, `omitted`, and `limit` alongside the records. If a record does not fit, it remains in the corpus and contributes to the omitted count.

The caller can then narrow the query or use `read_source`. An empty search response means that this retrieval policy found no eligible matches. It does not establish that the company has no debt or that a fact does not exist elsewhere.

## 6. Side effects, retries, and parallel calls

Reading a local immutable file is straightforward to retry. Appending a note, submitting a job, or updating an external service introduces a commit boundary. If a tool times out, the caller may not know whether the effect happened.

Run `07_retries_and_effects.py`. Its first writer inserts a note, commits the transaction, and then raises a timeout to represent a lost acknowledgement. Retrying creates a second note. The failure occurred after the effect, so retrying the function repeated the effect.

The second writer uses a stable operation key, such as `project-17:AAPL:debt-question:v1`. It saves the note and a receipt for that key in the same transaction. When the caller retries with the same key and arguments, the function returns the existing receipt. Reusing the key with different arguments raises an error.

This is **idempotency**: repeating a particular operation has the same intended effect as executing it once. A model-generated tool-call ID is useful for connecting a request and response, but a new attempt can generate a new call ID. The operation key should identify the business action we want to deduplicate across those attempts.

The transaction in this example covers both the effect and its receipt because both live in one database. A local receipt cannot make a remote API call atomic with our local commit. For a remote write, use the service's idempotency support when available, or reconcile its state before repeating an uncertain operation. A durable workflow engine still needs a policy for this boundary.

Our application uses several different write behaviors. Evidence uses a canonical content hash and `INSERT OR IGNORE`, so an identical observation reuses an ID. An accepted note replaces the prior accepted note for that ticker. A working-memory edit can conflict with another edit, and blindly appending the same sentence twice would still create duplication. Each write deserves its own retry policy.

Parallel tool execution also changes the design. Independent reads can often overlap. A ratio calculation must wait for its inputs, and concurrent writes to the same note can overwrite one another. `async` permits overlapping work but does not make shared state safe. The course supervisor runs ticker tasks serially and holds a project lock. If we later parallelize it, we must revisit task claiming, provider rate limits, and shared report generation.

A cache solves a different problem: avoiding repeated work when inputs are unchanged. A risk-calculation cache key should include ticker, window, price-adjustment convention, source hash, and calculation version. A key containing only the ticker can return last month's result for this month's question. A tool-call ID, an idempotency key, and a cache key each identify something different.

## 7. What do we mean by memory?

“Add memory” is too broad to implement. Start by asking what the application must retain and how it will use it.

| Kind of memory | Example | How we use it |
|---|---|---|
| Current context | Recent tool results and instructions | Supply the next model request |
| Working memory | “Check whether cash covers near-term maturities” | Resume an investigation without rereading everything |
| Evidence | A dated source excerpt or exact risk calculation | Support claims and reproduce calculations |
| Plan | “Read debt schedule, compare cash, write note” | Organize the model's next steps |
| Workflow state | AAPL running, attempt 2, JPM pending | Decide what Python should execute next |
| Checkpoint | Settled conversation messages | Restore an interrupted agent run |
| Procedures | Tool definitions and research instructions | Define how research should proceed |

These are logical responsibilities. They do not require seven databases. Our application puts evidence, accepted notes, task status, and events in `memory.sqlite`. Pydantic AI Harness stores plans in `plans.sqlite`, working notes in `notebook.sqlite`, and conversation checkpoints in `steps.sqlite`.

Run `04_inspect_project.py` on a project with one completed ticker. Find an exact metric, an open research question, a plan item, and an accepted task. Ask which of those would be sufficient to reconstruct a report. A sentence in working memory saying “AAPL is done” cannot replace the task's acceptance checks.

A memory system needs a full lifecycle: decide what to write, attach metadata, store it, retrieve it, include it in context, and eventually update or retire it. Storage is only one step. A perfectly durable record that we never retrieve has no influence on the model's next answer.

## 8. What should enter memory?

We should distinguish observations, sourced claims, and hypotheses. “The calculation returned 30.29% annualized volatility over this window” is an observation. “Management expects a stronger second half” is a claim attributable to a source. “Refinancing may become more expensive” is a hypothesis to investigate. Saving all three as unqualified facts encourages later requests to treat speculation as established evidence.

A richer evidence record might contain:

```python
{
    "record_id": "...",
    "project_id": "...",
    "entity": "AAPL",
    "kind": "reported_debt",
    "value": 100,
    "unit": "USD million",
    "period_end": "2023-12-31",
    "available_at": "2024-02-02",
    "ingested_at": "2024-02-03",
    "source_hash": "...",
    "locator": "document URL and page",
    "status": "reported",
    "supersedes": None,
}
```

This is a design example for a financial-statement extension. The current evidence schema stores `kind`, ticker, a content-derived ID, and a payload containing the relevant metadata. Our projects freeze their inputs and code, which keeps the first implementation manageable.

The dates answer different questions. `period_end` tells us when the measured economic period ended. `available_at` tells us when an investor could have known the information. `ingested_at` tells us when our system acquired it. A December 31 balance-sheet value published February 2 cannot enter a January research project just because the economic period ended in December.

Updates need a policy too. Suppose a February document reports debt of 100, and a May correction reports 120. An April investigation should retain the original observation. A June investigation should see the correction and its relationship to the original. Globally marking the old record “superseded” and filtering it out of every query would erase the historical view. We need to apply version and availability rules relative to the question's date.

Expiration depends on the information. A current quote may become stale quickly. A historical filing excerpt remains useful when its publication date and version are intact. A derived ratio becomes invalid when an input or calculation changes, even if only a minute has passed. A single time-to-live for all memory would miss these differences.

## 9. Choosing a memory store and retrieval method

A vector database is one possible component of memory. It does not decide which facts belong in memory, when they become available, or whether they support a conclusion.

| Approach | Useful for | Decisions still required |
|---|---|---|
| Files | Small working notes and procedures | Naming, concurrent edits, scope, versioning |
| SQL tables | Exact values, task state, dated evidence | Schema, transactions, indexes, migrations |
| Full-text search | Names, phrases, filing terminology | Tokenization, ranking, metadata filters |
| Vector search | Queries that use different words from the source | Embedding model, chunking, filters, evaluation |
| Hybrid retrieval | Combining lexical and semantic matches | Score combination, reranking, duplicate removal |

We use SQLite because students can inspect the state directly and the corpus is small. [SQLite FTS5](https://www.sqlite.org/fts5.html) provides a possible next step for full-text search. An embedding index becomes useful when literal wording is a poor guide to relevance, but similarity still does not establish the correct ticker, reporting period, or units.

Keep exact numbers and dates in structured fields even if their surrounding text enters a vector index. Retain the original source and a stable locator so a retrieved chunk can support a citation. If the embedding model or chunking policy changes, record that index version. Otherwise, a change in retrieval behavior becomes difficult to explain.

Chunking affects what retrieval can recover. Very small chunks may separate a value from its header or qualification. Very large chunks can consume the result budget with unrelated content. Overlap helps preserve context at boundaries, but can also return near-duplicate passages that crowd out a different source. Evaluate the retrieved evidence at the level the research question needs.

## 10. Retrieval is a pipeline

Our source retrieval first applies ticker and publication-date constraints, then ranks the eligible records, selects records that fit the response budget, and saves the returned evidence. The order matters.

Run `06_memory_and_retrieval.py`. Its small corpus includes a future AAPL document and a document about MSFT. Both overlap strongly with the query “debt refinancing liquidity.” If we rank the entire corpus, take two records, and only then filter for AAPL as of April 30, we return nothing. The two slots have already gone to ineligible records. Filtering first leaves three eligible records, of which the debt and cash excerpts match the query.

The lexical baseline scores the number of distinct query terms found in the title and text. It uses deterministic tie-breaking so we can reproduce the result. This is intentionally simple enough to inspect line by line. Its weaknesses are visible: “rollover obligations” gets no matches even though the debt excerpt discusses refinancing.

Possible changes include a small domain vocabulary, stemming, query rewriting, or semantic retrieval. Each change should have a measurable purpose. If we tune the search on this one query and then report success on the same query, we have measured a repair to that example. We still need new queries to evaluate whether the improvement generalizes.

Response size changes the result too. The experiment fits one whole evidence record into 600 characters and both matches into 850. These budgets include the envelope and identifiers. The application search tool uses 12,000 characters and up to five records. Character counts are easy to enforce in this teaching example; actual token usage depends on the tokenizer and content.

A useful retrieval policy also considers coverage and disagreement. Five nearly identical excerpts may be less useful than a debt schedule, a cash disclosure, and a conflicting management statement. We should retain the provenance of conflicting claims and ask whether they refer to different periods or definitions. Picking the highest search score does not resolve the disagreement.

## 11. Context is a limited working area

Every model request has a context budget. Instructions, tool schemas, history, evidence, and the eventual answer compete for it. Here is an illustrative allocation for a 24,000-token design target:

| Component | Tokens |
|---|---:|
| Instructions | 2,000 |
| Tool definitions | 2,000 |
| Recent conversation | 6,000 |
| Retrieved evidence | 10,000 |
| Answer reserve | 2,000 |
| Headroom | 2,000 |

These are planning numbers, not measured usage from our demo. The provider's context capacity and output limits determine what is possible. We should inspect actual usage and leave room for the answer, rather than filling every available token with retrieved text.

There are several ways to manage a growing conversation. A sliding window retains recent exchanges but can drop the original reason for a decision. A summary can preserve that reason, but may omit an exception or alter a number. Retrieval brings back selected evidence, but can miss a relevant record. In practice, we often combine a short working summary with retrieval of exact records.

Our `SlidingWindowCompaction` trims older messages and preserves valid tool-call/result relationships. It does not generate an LLM summary. The configuration sets a 24,000-token trigger and a 12,000-token retention target, subject to the capability's other retention rules. Exact evidence remains in the application store and can be recalled. See [context compaction](https://pydantic.dev/docs/ai/harness/compaction/).

A useful working summary records the current question and scope, conclusions with evidence IDs, unresolved issues, and the next action. Keep exact financial values in retrievable evidence as well. If a summary says “leverage improved,” the underlying records should still tell us which leverage definition and dates support that statement.

Long histories also affect cost. Suppose every request repeats a fixed context of `C` tokens and each completed round adds `R` tokens. Across `n` requests, input tokens are approximately `n*C + R*n*(n-1)/2`, before compaction or provider caching. With `C=4000`, `R=1500`, and `n=8`, that is 74,000 input tokens. Reducing each addition to 500 tokens gives 46,000. These are token counts, not dollar prices. A provider cache may reduce billing for repeated content, but it does not improve the relevance of that content.

Retrieved documents and model-written memory are also inputs that can contain instructions. A source excerpt saying “ignore the research date” should remain quoted source content. It should not acquire the authority of the application's instructions. Keep project constraints in Python, identify source material clearly, and give tools only the access needed for their task.

## 12. Concurrent edits, scope, and retention

The second half of `06_memory_and_retrieval.py` uses the actual `SqliteMemoryStore`. Two researchers read the same version of a note. Researcher A adds a cash comparison. Researcher B adds a refinancing question using the older version.

The store rejects B's stale write because its `expected_version` no longer matches. B then rereads the current note and deliberately merges the additional question. This is optimistic concurrency control: the write succeeds only if the record has not changed since we read it. The [memory capability](https://pydantic.dev/docs/ai/harness/memory/) exposes versioned memory operations.

Detecting the conflict prevents a lost update. It does not decide how to merge contradictory conclusions. If one researcher says debt increased and another says it decreased, concatenating their text preserves the conflict but does not resolve it. We must inspect periods, definitions, and evidence.

Scope is equally important. Our memory lives inside a project directory and uses a ticker namespace. A shared service would also need an authenticated user or organization scope supplied by the application. Allowing the model to choose an arbitrary namespace would weaken that boundary.

Retention should reflect the role of each record. We may discard an old scratch note after preserving its useful questions. A published report's citations need their supporting records to remain available, or a documented archive that can resolve them. Conversation transcripts, caches, and accepted evidence do not necessarily need the same retention period.

## 13. What a checkpoint can restore

A checkpoint records a settled point in the agent conversation. On restart, our application looks for the latest usable snapshot from an earlier attempt and supplies its messages to a new run. The new attempt gets a new run ID while retaining the ticker's conversation identity.

The checkpoint does not contain every part of the application. Task status lives in the supervisor's database, working notes have their own store, and an external service has its own state. We need to understand the boundaries between these commits.

| Interruption point | What recovery can do |
|---|---|
| Before claiming a task | Leave it pending |
| After a calculation and settled tool result | Restore that result and continue |
| After evidence commit but before a settled tool result | Repeat the call and reuse the evidence ID |
| After a remote write with no acknowledgement | Reconcile the effect or reuse a remote idempotency key |
| After note commit but before task completion | Restart the attempt and validate a replacement note |
| After accepted completion | Skip the ticker on later invocations |

Run `03_resume_research.py runs/recovery`. It interrupts the demo after a calculation, starts a fresh worker, and verifies that the settled calculation is reused. Then compare that result with the lost-acknowledgement experiment in script 07. They interrupt at different boundaries, so they require different mechanisms.

Saving steps gives us a basis for resumption. It does not guarantee that every operation executes exactly once. Durable execution frameworks can automate restart scheduling and replay, but writes still need appropriate idempotency or reconciliation. See [step persistence](https://pydantic.dev/docs/ai/harness/step-persistence/).

We also freeze the project configuration and hashes of its inputs, code, and dependency lock. Resuming old messages against changed calculation code can otherwise mix two meanings of the same tool. After changing inputs or code, create a new project. A larger application could introduce explicit state and tool-schema migrations, but that requires a compatibility policy.

## 14. Budgets, failure categories, and progress

The defaults are two attempts per ticker, twelve model requests per attempt, twenty successful agent tool calls per attempt, and a 120-second worker deadline. Planning and memory calls count toward the successful-call limit. The application separately reserves dispatched research actions, including ones that fail after dispatch. `run --seconds` limits a supervisor invocation, and attempt counts persist across invocations.

These limits control different failure modes. Request limits bound a model that repeatedly proposes invalid arguments. Tool limits constrain successful activity. A worker deadline bounds a process that does not return. A retry limit bounds repeated attempts across restarts. A fresh process should not reset the whole project's budget.

| Failure | Useful response |
|---|---|
| Repairable argument or citation | Return specific validation feedback |
| Missing required price data | Record the failure and preserve the other tickers |
| Temporary provider failure | Retry within a bounded policy |
| Rate limit | Respect the service's retry timing and total deadline |
| Uncertain external write | Check the prior effect before repeating it |
| Repeated calls with no new evidence | Stop, revise the question, or report the unresolved issue |

Retries can multiply. Two task attempts, three SDK attempts per request, and three HTTP attempts inside a tool can permit eighteen underlying HTTP attempts. Decide which layer owns each retry and observe the total cost. Add backoff and jitter for transient service failures, but do not use repeated retries to conceal a deterministic data error.

Our worker uses a subprocess deadline so the supervisor can terminate work that outlasts its budget. Cancellation cannot undo an external operation that already committed. A forcibly terminated process may also miss its final usage receipt, although earlier committed events remain available.

Progress deserves a definition beyond “the model is still making calls.” New evidence, resolution of an open question, and acceptance of a valid result are useful signals. Rewriting the same plan repeatedly consumes resources without necessarily advancing the investigation. A longer research application could track repeated tool arguments and unchanged evidence sets, then request a revised plan or stop with an explanation.

## 15. Plans and research decomposition

A plan is useful when the model must keep track of several related questions. For a refinancing investigation, the plan might ask for the debt schedule, compare near-term obligations with cash, inspect qualifications in the disclosure, and draft a conclusion with citations.

A plan should include a recognizable completion condition. “Research AAPL” is difficult to evaluate. “Produce a dated risk calculation and a source-supported interpretation, with both cited in the final note” is closer to a contract we can check.

The application queue and the model's plan operate at different levels. The queue assigns tickers and attempts. The plan organizes work within one ticker. Splitting work across more agents would add coordination questions: who owns shared evidence, how conflicting conclusions are resolved, which tasks depend on others, and who accepts the final report. Add that complexity when the task structure justifies it.

We can test whether planning helps by comparing repeated provider-backed runs with and without the capability. Compare supported conclusions, missed evidence, latency, and cost. The deterministic demo follows a fixed sequence, so it cannot establish that planning improves research quality.

## 16. Acceptance and report generation

The model returns a typed `ResearchNote` containing a body and citation IDs. Pydantic checks the structure. Our validator checks that cited records exist, belong to the assigned ticker, and include both calculated metrics and source evidence. Invalid references return to the model for repair. Python saves an accepted note only after these checks. See [structured outputs](https://pydantic.dev/docs/ai/output/).

Citation validity is only one layer of review. A real price-volatility record does not support the sentence “earnings will grow next quarter.” The reference can exist and still fail to support the claim. A future extension could represent individual claims with their supporting IDs, making claim-level review easier than reviewing one large paragraph.

The report builds numerical tables from saved calculations and adds accepted interpretations. It includes actual data coverage, task status, and an evidence appendix. A failed ticker remains visible. Portfolio weights require prices for every holding because the denominator includes all equity market values plus cash. Silently dropping a failed holding would change the meaning of every reported weight.

A report that compares tickers should also make their observation windows visible. If one stock has a shorter history, apparent risk differences may partly reflect different periods. The current report exposes coverage so the reader can identify that issue. A stricter comparison module could require a common window before calculating comparative statistics.

## 17. Evaluating the harness in layers

An end-to-end report can look plausible even when retrieval missed the key source. Evaluate the components at the level where failures arise.

| Layer | Example check |
|---|---|
| Calculation | Prices 100, 110, 99 produce −1% total return |
| Tool interface | A schema failure never enters the calculation |
| Retrieval | Correct ticker and date, with the required supporting excerpt |
| Memory | A stale edit cannot silently overwrite a newer finding |
| Recovery | An uncertain retry does not duplicate the intended write |
| Output | Citation IDs resolve to the assigned ticker's evidence |
| Claim support | Each substantive claim follows from its cited evidence |
| Whole project | Complete coverage, useful report, acceptable latency and cost |

Run `08_evaluate_retrieval.py`, then run it with `--rewrite`. The script has four questions with known supporting excerpts. Recall@2 is the fraction of expected relevant records found among the top two returned results. Precision here is the fraction of returned records that are relevant, using zero when nothing returns.

The lexical baseline has mean recall of 0.75. Rewriting “rollover obligations” as “debt refinancing” raises it to 1.0 on these four examples. We have repaired one known miss. To evaluate the policy, add held-out questions, ambiguous terms, duplicate excerpts, conflicting statements, and documents from the wrong date or ticker. A metadata violation should fail the evaluation even if the returned passage sounds highly relevant.

Compare configurations under the same corpus, questions, budgets, and acceptance rules. For a provider-backed model, use repeated runs and retain trajectories so we can inspect variation. Track token usage and elapsed time alongside quality. A retrieval change that improves recall but floods the model with irrelevant text may worsen the final answer or cost more than the improvement warrants.

## 18. Adding a research module

Suppose we want a leverage tool. First define the question precisely: debt divided by which equity measure, for which period, using what units? Decide what to return when equity is zero or negative. Determine which publication date makes the inputs eligible.

Then work through the same boundaries we have studied:

1. Define model arguments and application-owned scope.
2. Validate source fields and calculate the ratio in Python.
3. Save the inputs, units, dates, calculation version, and provenance with the result.
4. Specify retry behavior and any cache key.
5. Add acceptance checks and a report section that can explain the result.
6. Evaluate numerical cases, unavailable data, and claims that overstate what the ratio shows.

A notebook can remain behind a tool when it already contains useful analysis. A Papermill adapter would need a parameter schema, an isolated output directory, a deadline, and a structured result extracted from the executed notebook. The model-facing contract should describe the research operation rather than notebook execution details.

## Experiments and exercises

The class uses the scripts in the order that helps explain the mechanism:

| Script | Question to predict before running |
|---|---|
| `01_data_and_tools.py` | What are the total return and cash-aware portfolio weights? |
| `02_agent_and_tools.py` | Which part chooses a tool and which part calculates the number? |
| `05_tool_call_lifecycle.py` | How many requests, function entries, and calculations occur? |
| `06_memory_and_retrieval.py` | What changes when filtering precedes ranking? |
| `03_resume_research.py` | Which completed work can the next worker reuse? |
| `07_retries_and_effects.py` | Did a timeout happen before or after the write committed? |
| `08_evaluate_retrieval.py` | Which query misses its supporting source? |
| `04_inspect_project.py` | Which store owns each part of the completed investigation? |

**A. A correction in memory.** Design records for debt of 100 published in February and a correction to 120 published in May. Show what an April query and a June query should return. Include units, economic period, publication time, and the relationship between the versions.

**B. Retrieval under pressure.** Add a synonym query and a near-duplicate excerpt to the retrieval lab. Change one retrieval policy and evaluate both the original questions and two new questions. Explain whether the change improved coverage, precision, or only the example you tuned it on.

**C. An uncertain write.** Change the operation key in script 07 on every retry. Predict the number of notes. Then restore a stable key and try different arguments with the same key. Explain why the function should reject that reuse.

**D. A context budget.** Allocate 24,000 tokens across instructions, tool schemas, history, evidence, answer reserve, and headroom. A new source needs 6,000 tokens. Decide what to remove or retrieve later, and explain which research question that choice might make harder to answer.

**E. A benchmark tool.** A stock moves from 100 to 102 while its benchmark moves from 200 to 202 over the same window. Compute the return difference, then define the tool's dates, calendar checks, result fields, and cache key. Add a case where the calendars differ.

**F. Citation versus support.** A note cites JPM's historical volatility and concludes that JPM's earnings will rise next quarter. Identify which automated checks could pass and what additional evidence the claim would need. Rewrite the note so its conclusion follows from the available evidence.

**G. A planning comparison.** Compare several provider-backed runs with and without planning while keeping the question and budgets fixed. Examine accepted claims, evidence coverage, cost, and latency. Explain what the deterministic demo can and cannot tell us about that comparison.
