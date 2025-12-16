from langgraph.graph import MessagesState
from typing import Annotated, List, Optional, Sequence, TypedDict
import operator
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


# The state for the "worker" research agent sub-graph.
class ResearcherState(TypedDict):
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    research_topic: str

    # The final, cleaned-up output of a research run.
    compressed_research: str

    # The temporary buffer of raw search results for this specific worker.
    raw_notes: Annotated[List[str], operator.add]


# A specialized state defining the output of the research agent sub-graph.
class ResearcherOutputState(TypedDict):
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
