"""
Test individual nodes in isolation.

This script allows you to test nodes directly without running the full graph.
"""

from langchain_core.messages import HumanMessage, AIMessage
from src.states.agent_state import AgentState
from src.nodes.intent_clarification import clarify_with_user
# from src.nodes.research_brief import write_research_brief
# from src.nodes.draft_generation import write_draft_report
# from src.nodes.final_report_generation import final_report_generation


def test_clarify_with_user():
    """Test the clarify_with_user node."""
    print("=" * 50)
    print("Testing: clarify_with_user node")
    print("=" * 50)

    # Create a test state
    test_state = AgentState(
        messages=[HumanMessage(content="I want to research AI safety")]
    )

    # Call the node
    result = clarify_with_user(test_state)

    print(f"\nResult type: {type(result)}")
    print(f"Goto: {result.goto}")
    print(f"Update: {result.update}")

    if result.update and "messages" in result.update:
        print(f"\nAI Response: {result.update['messages'][0].content}")

    return result


def test_clarify_with_user_vague():
    """Test clarify_with_user with a vague query."""
    print("\n" + "=" * 50)
    print("Testing: clarify_with_user node (vague query)")
    print("=" * 50)

    test_state = AgentState(
        messages=[HumanMessage(content="Tell me about AI")]
    )

    result = clarify_with_user(test_state)

    print(f"\nResult type: {type(result)}")
    print(f"Goto: {result.goto}")

    if result.update and "messages" in result.update:
        print(f"\nAI Response: {result.update['messages'][0].content}")

    return result


# Add more node tests here...
# def test_write_research_brief():
#     """Test the write_research_brief node."""
#     pass


if __name__ == "__main__":
    # Run individual node tests
    test_clarify_with_user()
    # test_clarify_with_user_vague()

    # Add more tests as needed
    # test_write_research_brief()
