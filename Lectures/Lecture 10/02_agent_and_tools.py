"""A first Pydantic AI agent: one Python tool and a typed result.

The FunctionModel supplies predictable responses. The Agent still validates the
requests, runs the tool, and constructs a ReturnSummary from the final response.
"""

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel


class ReturnSummary(BaseModel):
    total_return: float
    explanation: str


def classroom_model(messages, info):
    results = [part for message in messages for part in message.parts
               if isinstance(part, ToolReturnPart)]
    if not results:
        return ModelResponse(parts=[ToolCallPart("total_return", {"first": 100, "last": 99})])
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {
        "total_return": results[-1].content,
        "explanation": "The final price is 1% below the starting price.",
    })])


agent = Agent(FunctionModel(classroom_model), output_type=ReturnSummary,
              instructions="Use the tool to calculate the return, then explain it.")


@agent.tool_plain
def total_return(first: float, last: float) -> float:
    """Calculate the whole-period return from positive endpoint prices."""
    if first <= 0 or last <= 0:
        raise ValueError("Prices must be positive")
    return last / first - 1


if __name__ == "__main__":
    result = agent.run_sync("Prices went from 100 to 110 to 99. What was the total return?")
    print(result.output.model_dump_json(indent=2))
    print(f"Model requests: {result.usage.requests}. Tool calls: {result.usage.tool_calls}.")
