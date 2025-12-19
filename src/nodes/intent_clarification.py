from pydantic import BaseModel, Field
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.output_parsers import JsonOutputParser
from typing import Literal

from src.states.agent_state import AgentState
from src.models.hf_models import get_hf_model
from src.prompts import clarify_with_user_instructions
from src.helpers import get_today_str

from langgraph.graph import END


class ClarifyWithUser(BaseModel):
    """A Pydantic schema for the output of the user clarification decision agent."""

    # This boolean is the core of the decision-making process.
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    # The question to be asked if clarification is needed.
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    # A confirmation message to send to the user if no clarification is needed.
    verification: str = Field(
        description="A verification message to confirm that research will begin.",
    )


def clarify_with_user(state: AgentState) -> Command[Literal["write_research_brief", END]]:
    """
    This node acts as a gatekeeper. It determines if the user's request has enough detail to proceed.
    If not, it HALTS the graph and asks a clarifying question. If yes, it proceeds to the next step.
    """

    # 1. We first format the entire conversation history into a single string for the LLM.
    messages_text = get_buffer_string(state["messages"])
    current_date = get_today_str()

    # 2. We bind our 'ClarifyWithUser' Pydantic schema to the model. This forces a structured JSON output.
    model = get_hf_model(model_name="moonshotai/Kimi-K2-Instruct")
    # output_parser = JsonOutputParser(pydantic_object=ClarifyWithUser)
    structured_output_model = model.with_structured_output(ClarifyWithUser)

    # 3. We invoke the LLM with our detailed 'clarify_with_user_instructions' prompt.
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=messages_text, 
            date=current_date
        ))
    ])

    # 4. This is the core logic. We check the 'need_clarification' boolean from the LLM's response.
    if response.need_clarification:

        # If clarification is needed, we return a special 'Command' object.
        # 'goto=END' tells LangGraph to STOP execution here.
        # 'update={...}' adds the AI's clarifying question to the message history before stopping.
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:

        # If no clarification is needed, we return another Command.
        # 'goto="write_research_brief"' directs the graph to proceed to the next node.
        # We add the AI's verification message to the history for a clean conversational flow.
        return Command(
            goto="write_research_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )