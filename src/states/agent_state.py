from langgraph.graph import MessagesState
from typing import Annotated, List, Optional, Sequence, TypedDict
from langgraph.graph.message import add_messages
import operator
from langchain_core.messages import BaseMessage


# The states for the top-level, user-facing graph.
class AgentInputState(MessagesState):
    """The initial input state, which only contains the user's messages."""
    pass


class AgentState(MessagesState):
    """The main state for the full multi-agent system, which accumulates all final artifacts."""
    research_brief: Optional[str]
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    raw_notes: Annotated[List[str], operator.add] = []

    notes: Annotated[List[str], operator.add] = [] # The final, curated notes for the writer.
    draft_report: str
    final_report: str
    