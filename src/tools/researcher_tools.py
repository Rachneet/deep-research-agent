from tavily import TavilyClient
from typing import List, Literal, Annotated

from langchain_core.tools import tool, InjectedToolArg
from src.helpers import process_search_results, format_search_output, deduplicate_search_results


# We initialize the Tavily client.
tavily_client = TavilyClient()
MAX_CONTEXT_LENGTH = 250000

def tavily_search_multiple(
    search_queries: List[str], 
    max_results: int = 3, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
    ) -> List[dict]:
    """A helper function to perform a search using the Tavily API for a list of queries."""

    print(f"--- [TOOL] Executing Tavily search for queries: {search_queries} ---")
    search_docs = []

    # We execute the searches for each query.
    for query in search_queries:
        result = tavily_client.search(
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic
        )
        search_docs.append(result)
    return search_docs


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    ) -> str:
    """
    A tool that fetches results from the Tavily search API and performs content summarization.


    Args:
        query: A single, specific search query to execute.
        max_results: The maximum number of results to return.
        topic: The topic to filter results by ('general', 'news', 'finance').
    Returns:
        A formatted string of the deduplicated and summarized search results.
    """

    # 1. Execute the search.
    search_results = tavily_search_multiple([query], max_results=max_results, topic=topic, include_raw_content=True)

    # 2. Deduplicate the results.
    unique_results = deduplicate_search_results(search_results)

    # 3. Process and summarize the content.
    summarized_results = process_search_results(unique_results)

    # 4. Format the final output.
    return format_search_output(summarized_results)