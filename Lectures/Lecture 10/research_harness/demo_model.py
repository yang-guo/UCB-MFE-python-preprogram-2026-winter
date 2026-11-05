"""A deterministic model for the classroom. It uses the real agent and tool loop."""

from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel


def respond(messages, info):
    """Choose the next call from completed tool results, including resumed history."""
    returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    results = {part.tool_name: part.content for part in returns}

    def call(name, arguments):
        return ModelResponse(parts=[ToolCallPart(name, arguments)])

    steps = ["Collect price and source evidence", "Write findings and a follow-up question"]
    if "write_plan" not in results:
        return call("write_plan", {"items": [{"content": s, "status": "pending"} for s in steps]})
    for name in ["recall_memory", "analyze_prices", "read_source"]:
        if name not in results:
            return call(name, {})
    metric = results["analyze_prices"]
    sources = results["read_source"]
    if not sources:
        raise ValueError("No source was available by the research cutoff")
    m, ticker = metric["payload"], metric["ticker"]
    follow_up = f"Next for {ticker}: compare the latest eligible filing's business risks with the price history."
    # A recovery may have saved memory just before the process stopped. The
    # capability injects that memory, so the demo avoids appending it twice.
    if "write_memory" not in results and follow_up not in str(messages):
        return call("write_memory", {"content": follow_up})
    if sum(part.tool_name == "write_plan" for part in returns) < 2:
        return call("write_plan", {"items": [{"content": s, "status": "completed"} for s in steps]})
    body = (f"{ticker} returned {m['total_return']:.2%} from {m['first_date']} to "
            f"{m['last_date']}. Its largest decline from a previous peak was "
            f"{abs(m['max_drawdown']):.2%}. {follow_up}")
    return call(info.output_tools[0].name, {"body": body, "citations": [metric["id"], sources[0]["id"]]})


def demo_model():
    return FunctionModel(respond, model_name="classroom-demo")
