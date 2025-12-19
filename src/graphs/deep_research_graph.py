from langgraph.graph import StateGraph, START, END
from src.states.agent_state import AgentInputState, AgentState
from src.nodes.intent_clarification import clarify_with_user
from src.nodes.research_brief import write_research_brief
from src.nodes.draft_generation import write_draft_report
from src.graphs.supervisor_graph import supervisor_agent
from src.nodes.final_report_generation import final_report_generation


# We initialize our master StateGraph, using the main AgentState and AgentInputState.
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# We add our pre-compiled sub-graphs and our individual nodes to the master graph.
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("write_draft_report", write_draft_report)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent) # Here we add our complex sub-graph as a single node.
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# Now, we define the high-level control flow for the entire system.
# The entry point is the 'clarify_with_user' node.
deep_researcher_builder.add_edge(START, "clarify_with_user")

# The scoping process is a linear sequence.
deep_researcher_builder.add_edge("write_research_brief", "write_draft_report")

# After the initial draft is created, we hand off control to the main Supervisor loop.
deep_researcher_builder.add_edge("write_draft_report", "supervisor_subgraph")

# Once the Supervisor loop completes (by calling ResearchComplete), its output is passed to the final writer.
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")

# After the final report is generated, the graph terminates.
deep_researcher_builder.add_edge("final_report_generation", END)

# We compile the full, end-to-end workflow into our final 'agent' object.
deep_research_agent = deep_researcher_builder.compile()
print("Advanced Systems Loaded: Red Team, Context Pruner, and Evaluator are online.")


### Graph visualization and saving
# save the graph for visualization
import os
from pathlib import Path

# save the graph for visualization
# Use absolute path to ensure consistency regardless of where script runs from
project_root = Path(__file__).parent.parent.parent  # Navigate from src/graphs/deep_research_graph.py to project root
figures_dir = project_root / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)  # Create figures/ if it doesn't exist

save_path = str(figures_dir / "deep_research_graph.png")

try:
    deep_research_agent.get_graph().draw_png(output_file_path=save_path)
    print(f"✓ Graph saved to {save_path}")
except Exception as e:
    print(f"✗ Failed to save graph: {e}")
