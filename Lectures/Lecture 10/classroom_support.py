"""Display Pydantic AI messages as a readable table for the lecture."""

import json

import pandas as pd
from pydantic_ai.messages import TextPart, ToolCallPart, ToolReturnPart, UserPromptPart


def show_messages(messages):
    """Turn Pydantic AI messages into a table; this does not call the model."""
    rows = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, UserPromptPart):
                rows.append({'who': 'You', 'action': 'Ask', 'content': str(part.content)})
            elif isinstance(part, ToolCallPart):
                rows.append({'who': 'Model', 'action': f'Request {part.tool_name}',
                             'content': json.dumps(part.args_as_dict())})
            elif isinstance(part, ToolReturnPart):
                rows.append({'who': 'Python', 'action': f'Return {part.tool_name}',
                             'content': json.dumps(part.content)})
            elif isinstance(part, TextPart):
                rows.append({'who': 'Model', 'action': 'Answer', 'content': part.content})
    return pd.DataFrame(rows, columns=['who', 'action', 'content'])
