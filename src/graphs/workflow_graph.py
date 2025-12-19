"""
Building the entire graph.

It encapsulates the entire user interaction and task definition process.

If a user request is ambiguous, this sub-graph will run, ask a clarifying question, and then stop.
If the request is clear, it will run through all three nodes, producing the research_brief and 
the initial draft_report needed by the main Supervisor loop.
"""

from langgraph.graph import StateGraph, START, END

from src.states.agent_state import AgentInputState, AgentState
from src.nodes.intent_clarification import clarify_with_user
from src.nodes.research_brief import write_research_brief
from src.nodes.draft_generation import write_draft_report


# We initialize a new StateGraph, specifying the main AgentState and the initial AgentInputState.
scope_builder = StateGraph(AgentState, input_schema=AgentInputState)

# We add the three nodes that make up our scoping process.
scope_builder.add_node("clarify_with_user", clarify_with_user)
scope_builder.add_node("write_research_brief", write_research_brief)
scope_builder.add_node("write_draft_report", write_draft_report)

# The entry point is the 'clarify_with_user' node.
scope_builder.add_edge(START, "clarify_with_user")

# We already built the dynamic routing inside the 'clarify_with_user' node.
# It will either END the graph or proceed to 'write_research_brief'.
scope_builder.add_edge("write_research_brief", "write_draft_report")

# The 'write_draft_report' node is the final step of this sub-graph.
scope_builder.add_edge("write_draft_report", END)

# We compile this into a runnable sub-graph.
scope_research = scope_builder.compile()