"""Compose a research agent from typed tools and Pydantic AI Harness capabilities."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import to_jsonable_python
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness import Memory, Planning, SlidingWindowCompaction, StepPersistence
from pydantic_ai_harness.memory import SqliteMemoryStore
from pydantic_ai_harness.planning import SqlitePlanStore
from pydantic_ai_harness.step_persistence import SqliteStepStore, continue_run

from .memory import connect, event
from .tools import dispatch, validate_note

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ResearchDeps:
    """Application state supplied by Python, outside the model's tool arguments."""

    run: Path
    ticker: str
    attempt: int


class ResearchNote(BaseModel):
    model_config = ConfigDict(extra="forbid")

    body: str = Field(min_length=40, max_length=3000)
    citations: list[str] = Field(min_length=2, max_length=20)


def build_agent(deps: ResearchDeps, model: Model | str) -> Agent[ResearchDeps, ResearchNote]:
    """Each capability has a separate job and an explicit persistent store."""
    agent = Agent(
        model,
        name="stock_researcher",
        deps_type=ResearchDeps,
        output_type=ResearchNote,
        instructions=(ROOT / "research_harness/instructions.txt").read_text(),
        retries=2,
        model_settings={"parallel_tool_calls": False, "max_tokens": 2000},
        capabilities=[
            Planning(
                store=SqlitePlanStore(str(deps.run / "plans.sqlite"), session=deps.ticker),
                tools=["write_plan"],
            ),
            Memory(
                SqliteMemoryStore(database=deps.run / "notebook.sqlite"),
                namespace=deps.ticker,
                max_tokens=1200,
                max_memory_size=8000,
                injection_errors="raise",
            ),
            SlidingWindowCompaction(max_tokens=24000, keep_tokens=12000),
            StepPersistence(
                store=SqliteStepStore(database=deps.run / "steps.sqlite", max_snapshots_per_run=30),
                run_id=f"{deps.ticker}-attempt-{deps.attempt}",
            ),
        ],
    )

    @agent.tool
    def analyze_prices(ctx: RunContext[ResearchDeps]) -> dict:
        """Compute historical returns, volatility, and drawdown for the assigned ticker."""
        d = ctx.deps
        return dispatch(d.run, d.ticker, d.attempt, "analyze_prices", {})

    @agent.tool
    def read_source(ctx: RunContext[ResearchDeps]) -> list[dict]:
        """Retrieve up to five source excerpts available by the project's cutoff."""
        d = ctx.deps
        return dispatch(d.run, d.ticker, d.attempt, "read_source", {})

    @agent.tool
    def search_sources(
        ctx: RunContext[ResearchDeps],
        query: Annotated[str, Field(min_length=1, max_length=200)],
        limit: Annotated[int, Field(ge=1, le=5)] = 3,
    ) -> dict:
        """Search dated source excerpts by words in their title or text.

        Use a specific topic such as debt maturities. Results include evidence
        IDs and an omitted count. An empty result means no lexical match within
        the project's eligible corpus. read_source can inspect recent excerpts.
        """
        d = ctx.deps
        return dispatch(d.run, d.ticker, d.attempt, "search_sources", {"query": query, "limit": limit})

    @agent.tool
    def recall_memory(ctx: RunContext[ResearchDeps]) -> list[dict]:
        """Retrieve exact evidence already saved for this ticker in this project."""
        d = ctx.deps
        return dispatch(d.run, d.ticker, d.attempt, "recall_memory", {})

    @agent.output_validator
    def check_citations(ctx: RunContext[ResearchDeps], note: ResearchNote) -> ResearchNote:
        # Pydantic checks the shape. Our validator checks the research meaning.
        try:
            with connect(ctx.deps.run) as db:
                validate_note(db, ctx.deps.ticker, note.body, note.citations)
        except ValueError as error:
            raise ModelRetry(str(error)) from error
        return note

    return agent


async def previous_history(deps: ResearchDeps) -> tuple[list, str | None]:
    """Load the newest settled checkpoint, skipping attempts with none."""
    store = SqliteStepStore(database=deps.run / "steps.sqlite")
    for attempt in range(deps.attempt - 1, 0, -1):
        run_id = f"{deps.ticker}-attempt-{attempt}"
        try:
            history = await continue_run(store, run_id=run_id)
        except LookupError:
            continue
        return history, run_id
    return [], None


async def run_research(deps: ResearchDeps, config: dict, model: Model | str | None = None):
    """Run one attempt, resume its conversation, and commit a checked note."""
    if model is None:
        if config["engine"] == "demo":
            from .demo_model import demo_model
            model = demo_model()
        else:
            model = config["model"]
    agent = build_agent(deps, model)
    history, resumed_from = await previous_history(deps)
    with connect(deps.run) as db:
        event(db, "agent_started", {"ticker": deps.ticker, "attempt": deps.attempt,
                                    "resumed_from": resumed_from, "messages": len(history)})
    prompt = (
        f"Research {deps.ticker}. Question: {config['question']}\n"
        f"Window: {config['start']} through {config['as_of']}.\n"
        "Continue from the saved evidence and plan. Return the completed ResearchNote."
    )
    usage = RunUsage()
    try:
        result = await agent.run(
            prompt, deps=deps, message_history=history,
            run_id=f"{deps.ticker}-attempt-{deps.attempt}", conversation_id=deps.ticker,
            usage=usage,
            usage_limits=UsageLimits(request_limit=config["max_requests"],
                                     tool_calls_limit=config["max_tool_calls"]),
        )
        dispatch(deps.run, deps.ticker, deps.attempt, "save_note", result.output.model_dump())
        return result
    finally:
        # The step store also records model and tool events as they happen.
        with connect(deps.run) as db:
            event(db, "agent_usage", {"ticker": deps.ticker, "attempt": deps.attempt,
                                      **to_jsonable_python(asdict(usage))})
