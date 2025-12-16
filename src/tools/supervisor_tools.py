from langchain_core.tools import tool
from pydantic import BaseModel, Field


@tool
class ConductResearch(BaseModel):
    """
    A tool for delegating a specific research task to a specialized sub-agent. 
    The Supervisor uses this to fan out work.
    """
    # The detailed research topic for the sub-agent. This needs to be a complete, standalone instruction.
    research_topic: str = Field(
        description="The topic to research. Should be a single, self-contained topic described in high detail.",
    )


@tool
class ResearchComplete(BaseModel):
    """
    A tool for the Supervisor to signal that the research process is complete 
    and the final report can be generated.
    """
    # This tool takes no arguments. Its invocation is the signal.
    pass


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """
    Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.
    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps.
    Returns:
        Confirmation that reflection was recorded for decision-making.
    """
    return f"Reflection recorded: {reflection}"