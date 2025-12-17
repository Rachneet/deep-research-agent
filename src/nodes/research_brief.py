from pydantic import BaseModel, Field
from langgraph.types import Command
from langchain_core.messages import HumanMessage, get_buffer_string
from typing import Literal

from src.states.agent_state import AgentState
from src.models.hf_models import get_hf_model
from src.prompts import transform_messages_into_research_topic_human_msg_prompt
from src.helpers import get_today_str


class ResearchQuestion(BaseModel):
    """A Pydantic schema for the structured output of the research brief generation agent."""

    # This single field will contain the complete, detailed research brief.
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


def write_research_brief(state: AgentState) -> Command[Literal["write_draft_report"]]:
    """
    This node transforms the confirmed conversation history into a single, comprehensive research brief.
    """
    # 1. We bind our 'ResearchQuestion' Pydantic schema to the model to ensure structured output.
    model = get_hf_model()
    structured_output_model = model.with_structured_output(ResearchQuestion)


    # 2. We invoke the LLM with our specialized 'transform_messages...' prompt and the conversation history.
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_human_msg_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    # 3. We return a Command to update the state with the new research_brief
    #    and direct the graph to proceed to the 'write_draft_report' node.
    return Command(
            goto="write_draft_report", 
            update={"research_brief": response.research_brief}
        )