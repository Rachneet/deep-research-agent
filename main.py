from langchain_core.messages import HumanMessage
from src.graphs.deep_research_graph import deep_research_agent
import asyncio


async def main():
    print("Hello from deep-research-agent!")

    # This is our complex, multi-faceted research query.
    complex_query = """
    Conduct a deep analysis of the 'Splinternet' phenomenon's impact on global semiconductor supply chains by 2028. 
    Specifically contrast TSMC's diversification strategy against Intel's IDM 2.0 model under 2024-2025 US export controls, 
    and predict the resulting shift in insurance liability models for cross-border wafer shipments.
    """

    # We invoke the fully compiled agent with our complex query.
    # The 'thread_id' ensures that our conversation history is maintained correctly in LangSmith.

    config = {"configurable": {"thread_id": "demo_complex_1"}}
    # NOTE: The following execution assumes valid API keys are set and will take several minutes to run.
    # The output shown below is a formatted representation of a real execution trace.
    result = await deep_research_agent.ainvoke(
        {"messages": [HumanMessage(content=complex_query)]}, 
        config=config
    )
    print("=== Final Output ===")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())