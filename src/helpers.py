from datetime import datetime
from typing import List

from langchain_core.messages import BaseMessage, filter_messages


def get_today_str() -> str:
    """A simple utility function to get the current date in a human-readable string format."""
    # We format the date as "Day Mon Day, Year" (e.g., "Mon Dec 25, 2025").
    return datetime.now().strftime("%a %b %-d, %Y")


def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """A helper function to extract the string content from ToolMessage objects in the supervisor's message history."""
    # This filters the message history for messages of type 'tool' and returns their content.
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]