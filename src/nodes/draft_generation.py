from pydantic import BaseModel, Field
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from src.states.agent_state import AgentState
from src.models.hf_models import get_hf_model
from src.prompts import draft_report_generation_prompt
from src.helpers import get_today_str


class DraftReport(BaseModel):
    """A Pydantic schema for the structured output of the initial draft generation agent."""

    # This single field will contain the complete, unresearched initial draft report.
    draft_report: str = Field(
        description="A draft report that will be used as the starting point for the diffusion process.",
    )


# Our creative model
creative_model = get_hf_model(model_name="moonshotai/Kimi-K2-Instruct")

def write_draft_report(state: AgentState) -> dict:
    """
    This node takes the research brief and generates an initial, unresearched draft.
    This serves as the "noisy" starting point for our diffusion process.
    """
    # 1. We use our 'creative_model' for this task and bind it to the 'DraftReport' schema.
    # structured_output_model = creative_model.with_structured_output(
    #     DraftReport,
    #     method="json_mode"
    # )
    output_parser = JsonOutputParser(
        name="DraftReportParser",
        pydantic_object=DraftReport
    )
    # structured_output_model = creative_model.with_output_parser(output_parser)
    research_brief = state.get("research_brief", "")

    # 2. We format the prompt for the drafter, injecting the research brief, current date, and format instructions.
    draft_report_prompt_formatted = draft_report_generation_prompt.format(
        research_brief=research_brief,
        date=get_today_str(),
        format_instructions=output_parser.get_format_instructions()
    )
    print(f"Draft Report Prompt: {draft_report_prompt_formatted[:500]}...")

    # 3. We invoke the LLM to generate the initial "noisy" draft.
    # response = structured_output_model.invoke([HumanMessage(content=draft_report_prompt_formatted)])
    # response = DraftReport.model_validate(response)
    response = creative_model.invoke([HumanMessage(content=draft_report_prompt_formatted)])
    parsed_response = output_parser.parse(response.content)
    print(f"Draft Report Response: {parsed_response}")

    # 4. This node returns a simple dictionary update. This is the final output of the 'scope_research' sub-graph.
    #    The 'supervisor_messages' field is populated to "hand off" the initial state to the main Supervisor loop.
    return {
        "research_brief": research_brief,
        "draft_report": parsed_response["draft_report"],
        "supervisor_messages": ["Here is the draft report: " + parsed_response["draft_report"], research_brief]
    }