"""
Compiling the Denoising Loop

Now we assemble the most complex part of our system which is the Supervisor main diffusion loop. 

This graph will contain the Supervisor brain, its hands (the tools node), 
and the parallel self-correction agents (the Red Team and the Context Pruner).
"""


from langgraph.graph import StateGraph, START, END

from src.states.supervisor_state import SupervisorState
from src.nodes.supervisor_node import supervisor, supervisor_tools
from src.nodes.red_team_node import red_team_node
from src.nodes.context_pruning_node import context_pruning_node


# We initialize the StateGraph for our Supervisor.
supervisor_builder = StateGraph(SupervisorState)

# We add the four key nodes of the main loop.
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_node("red_team", red_team_node)
supervisor_builder.add_node("context_pruner", context_pruning_node)

# The entry point of the loop is the 'supervisor' brain.
supervisor_builder.add_edge(START, "supervisor")

# After the brain thinks and plans, it proceeds to the 'supervisor_tools' node to act.
supervisor_builder.add_edge("supervisor", "supervisor_tools")

# The 'supervisor_tools' node is our parallel fan-out point. Its return Command
# dynamically directs the graph to run 'red_team' and 'context_pruner' simultaneously.
# Therefore, we only need to define the edges for the fan-in, which bring the control flow back.
# After the Red Team runs, control returns to the 'supervisor' for the next iteration.
supervisor_builder.add_edge("red_team", "supervisor")

# After the Context Pruner runs, control also returns to the 'supervisor'.
supervisor_builder.add_edge("context_pruner", "supervisor")

# We compile this into our main, cyclical 'denoising engine'.
supervisor_agent = supervisor_builder.compile()