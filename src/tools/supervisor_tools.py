from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Annotated

from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from src.prompts import report_generation_with_draft_insight_prompt
from src.helpers import get_today_str
from src.models.hf_models import get_hf_model


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
        reflection (str): Your detailed reflection on research progress, findings, gaps, and next steps.

    Returns:
        str: Confirmation that reflection was recorded for decision-making.
    """
    return f"Reflection recorded: {reflection}"



@tool(parse_docstring=True)
def refine_draft_report(
    research_brief: Annotated[str, InjectedToolArg], 
    findings: Annotated[str, InjectedToolArg], 
    draft_report: Annotated[str, InjectedToolArg]
) -> str:
    """Synthesizes all research findings into a comprehensive draft report.

    Args:
        research_brief: The user's research request (injected by tool runner).
        findings: Collected research findings for the user request (injected).
        draft_report: Current draft report to be refined (injected).

    Returns:
        The refined draft report content.
    """

    # We format the detailed prompt with the brief, the current draft, and the new findings.
    draft_report_prompt = report_generation_with_draft_insight_prompt.format(
        research_brief=research_brief,
        findings=findings,
        draft_report=draft_report,
        date=get_today_str()
    )
    # Writer model
    writer_model = get_hf_model(model_name="moonshotai/Kimi-K2-Instruct")
    # We invoke our powerful 'writer_model' to generate the new, "denoised" draft.
    draft_report_response = writer_model.invoke([HumanMessage(content=draft_report_prompt)])
    return draft_report_response.content
