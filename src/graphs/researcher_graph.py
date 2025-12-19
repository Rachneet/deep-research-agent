"""
This graph is the worker engine that our Supervisor will delegate tasks to.
It is a self-contained ReAct loop that takes a single research topic and 
produces a compressed summary of its findings.
"""


from langgraph.graph import StateGraph, START, END
from src.states.researcher_state import ResearcherState, ResearcherOutputState
from src.nodes.researcher_node import llm_call, tool_node, should_continue, compress_research


# We initialize a new StateGraph, specifying the input state and, importantly, the output schema for this sub-graph.
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# We add our three nodes: the thinker (llm_call), the actor (tool_node), and the final compressor.
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

# The entry point for this sub-graph is always the 'llm_call' (the brain).
agent_builder.add_edge(START, "llm_call")

# After the brain thinks, our conditional edge, 'should_continue', decides what to do next.
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node", # If tools are needed, we go to the tool node.
        "compress_research": "compress_research", # If research is done, we proceed to the compression node.
    },
)

# After the tool node acts, the flow loops back to the brain to process the results of the action.
agent_builder.add_edge("tool_node", "llm_call")

# The compression node is the final step; its output is the output of the entire sub-graph, so we connect it to END.
agent_builder.add_edge("compress_research", END)

# We compile the graph into a runnable object.
researcher_agent = agent_builder.compile()