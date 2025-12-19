from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage, filter_messages
from pydantic import BaseModel, Field
from typing import Literal, List

from src.states.researcher_state import ResearcherState
from src.models.hf_models import get_hf_model
from src.prompts import (
    research_agent_prompt, 
    summarize_webpage_prompt, 
    compress_research_system_prompt, 
    compress_research_human_message
)
from src.helpers import get_today_str   
from src.tools.researcher_tools import tavily_search


# We set up our tool-enabled model for the researcher agent.
model_with_tools = get_hf_model(model_name="moonshotai/Kimi-K2-Instruct").bind_tools([tavily_search])

def llm_call(state: ResearcherState):
    """The 'brain' of the researcher: analyzes the current state and decides on the next action (call a tool or finish)."""

    # This node invokes our tool-bound model with the specific research_agent_prompt and the current message history for this sub-task.
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt.format(date=get_today_str()))] + state["researcher_messages"]
            )
        ]
    }


def tool_node(state: ResearcherState):
    """The 'hands' of the researcher: executes all tool calls from the previous LLM response."""

    # We get the most recent message from the state, which should contain the tool calls.
    tool_calls = state["researcher_messages"][-1].tool_calls
    tools_by_name = {tool.name: tool for tool in model_with_tools.tools}

    # We execute all the planned tool calls.
    observations = []
    for tool_call in tool_calls:

        # We look up the correct tool function by its name.
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    # We format the results of the tool calls into 'ToolMessage' objects.
    # This is the standard way to return tool results to the LLM in LangGraph.
    tool_outputs = [
        ToolMessage(
            content=str(observation),
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    # We return the tool outputs to be added to the message history.
    return {"researcher_messages": tool_outputs}


def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """A conditional edge that determines whether to continue the ReAct loop or finish the research sub-task."""
    messages = state["researcher_messages"]
    last_message = messages[-1]

    # If the last message from the LLM contains tool calls, we continue the loop.
    if last_message.tool_calls:
        return "tool_node"

    # If there are no tool calls, the agent has decided its research is complete, and we proceed to the compression step.
    return "compress_research"  


def compress_research(state: ResearcherState) -> dict:
    """The final node in the research sub-graph: it compresses all findings from the ReAct loop into a clean, cited summary."""
    # 1. We format the system and human messages for our compression model.
    system_message = compress_research_system_prompt.format(date=get_today_str())

    # The full message history of the ReAct loop is passed as context.
    messages = [
        SystemMessage(content=system_message)] + \
            state.get("researcher_messages", []) + \
                [HumanMessage(content=compress_research_human_message.format(research_topic=state['research_topic']))]
    
    # 2. We invoke our powerful 'compress_model'.
    # Compress model
    compress_model = get_hf_model(model_name="moonshotai/Kimi-K2-Instruct", max_tokens=32000)
    response = compress_model.invoke(messages)

    # 3. We also extract the raw, unprocessed notes from the tool and AI messages.
    #    This is for archival purposes and can be used by the Supervisor for deeper analysis if needed.
    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"], 
            include_types=["tool", "ai"] # We only include the 'action' and 'observation' parts of the loop.
        )
    ]

    # 4. This node returns the final, clean outputs that will be passed out of the sub-graph.
    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }