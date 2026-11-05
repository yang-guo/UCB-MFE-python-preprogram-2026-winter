"""Trace schema validation, a business-rule failure, and a successful tool call."""

import json
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import UsageLimits


class Calculation(BaseModel):
    total_return: float
    observations: int


def experiment():
    prices = [100.0, 110.0, 99.0]
    entered = []
    calculated = []
    schemas = []

    def model(messages, info):
        if not schemas:
            schemas.append(info.function_tools[0].parameters_json_schema)
        returns = [p for message in messages for p in message.parts if isinstance(p, ToolReturnPart)]
        requests = [(1, "schema-error"), (12, "coverage-error"), (3, "valid-request")]
        if len(returns) < len(requests):
            window, call_id = requests[len(returns)]
            return ModelResponse(parts=[ToolCallPart("calculate_return", {"window": window}, tool_call_id=call_id)])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, returns[-1].content)])

    agent = Agent(FunctionModel(model), output_type=Calculation, retries=3)

    @agent.tool_plain
    def calculate_return(window: Annotated[int, Field(ge=2, le=252)]) -> dict:
        """Calculate total return over the final window price observations."""
        entered.append(window)
        if window > len(prices):
            raise ModelRetry(f"Only {len(prices)} prices are available. Use a window from 2 to {len(prices)}.")
        selected = prices[-window:]
        value = {"total_return": selected[-1] / selected[0] - 1, "observations": len(selected)}
        calculated.append(value)
        return value

    result = agent.run_sync("Calculate the return over the available history.",
                            usage_limits=UsageLimits(request_limit=5, tool_calls_limit=3))
    return schemas[0], result, entered, calculated


if __name__ == "__main__":
    schema, result, entered, calculated = experiment()
    print("TOOL ARGUMENT SCHEMA\n" + json.dumps(schema, indent=2))
    print("\nMESSAGE TRACE")
    for message in result.all_messages():
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                print("CALL", part.tool_call_id, part.tool_name, part.args)
            elif isinstance(part, ToolReturnPart):
                print("RETURN", part.tool_call_id, str(part.content)[:250])
    print("\nFunction entered with windows:", entered)
    print("Calculations completed:", len(calculated))
    print("Final output:", result.output.model_dump())
    print("Recorded usage:", result.usage)
    assert entered == [12, 3] and len(calculated) == 1
