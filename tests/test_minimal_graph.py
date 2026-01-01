"""
Test nodes within minimal graph configurations.

This allows you to test specific node sequences without running the entire graph.
"""

from langgraph.graph import StateGraph, START, END
from src.states.agent_state import AgentState, AgentInputState
from src.nodes.intent_clarification import clarify_with_user
from src.nodes.research_brief import write_research_brief
# from src.nodes.draft_generation import write_draft_report
# from src.nodes.final_report_generation import final_report_generation


def test_clarification_flow():
    """Test just the clarification node in a minimal graph."""
    print("=" * 50)
    print("Testing: Clarification Flow")
    print("=" * 50)

    # Build minimal graph
    builder = StateGraph(AgentState, input_schema=AgentInputState)
    builder.add_node("clarify_with_user", clarify_with_user)
    builder.add_edge(START, "clarify_with_user")
    # Note: We don't add edges from clarify_with_user since it uses Command routing

    agent = builder.compile()

    # Test with a clear query
    result = agent.invoke({
        "messages": [{"role": "user", "content": "I want to research the impact of AI on healthcare in 2024"}]
    })

    print(f"\nFinal state messages: {len(result['messages'])} messages")
    for i, msg in enumerate(result['messages']):
        print(f"\nMessage {i + 1} ({type(msg).__name__}):")
        print(f"  {msg.content[:200]}...")

    return result


def test_clarification_to_brief_flow():
    """Test clarification → research brief sequence."""
    print("\n" + "=" * 50)
    print("Testing: Clarification → Research Brief Flow")
    print("=" * 50)

    # Build minimal graph with two nodes
    builder = StateGraph(AgentState, input_schema=AgentInputState)
    builder.add_node("clarify_with_user", clarify_with_user)
    builder.add_node("write_research_brief", write_research_brief)
    builder.add_edge(START, "clarify_with_user")
    builder.add_edge("write_research_brief", END)

    agent = builder.compile()

    # Test
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Research quantum computing applications"}]
    })

    print(f"\nFinal state messages: {len(result['messages'])} messages")
    if 'research_brief' in result:
        print(f"\nResearch Brief Generated: {result['research_brief'][:200]}...")

    return result


if __name__ == "__main__":
    # Run tests
    test_clarification_flow()
    # test_clarification_to_brief_flow()  # Uncomment when ready
