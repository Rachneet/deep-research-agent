from tavily import TavilyClient
from typing import List, Literal, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, InjectedToolArg

from src.helpers import get_today_str
from src.prompts import summarize_webpage_prompt
from src.models.hf_models import get_hf_model

from src.config import settings


# We initialize the Tavily client.
tavily_client = TavilyClient(api_key=settings.tavily_api_key)
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

def deduplicate_search_results(search_results: List[dict]) -> dict:
    """Deduplicates a list of search results based on the URL."""
    unique_results = {}

    for response in search_results:
        for result in response['results']:
            url = result['url']
            
            # We use the URL as a key in a dictionary to automatically handle duplicates.
            if url not in unique_results:
                unique_results[url] = result
    return unique_results


def format_search_output(summarized_results: dict) -> str:
    """Formats the final, summarized search results into a clean string for the agent."""
    if not summarized_results:
        return "No valid search results found."
    
    formatted_output = "Search results: \n\n"
    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"
    return formatted_output


def process_search_results(unique_results: dict) -> dict:
    """Processes a dictionary of unique search results by summarizing their raw content."""
    summarized_results = {}
    for url, result in unique_results.items():
        # If raw_content is available, we summarize it.
        if result.get("raw_content"):
            content = summarize_webpage_content(result['raw_content'][:MAX_CONTEXT_LENGTH])
        else:
            # Otherwise, we just use the short snippet provided by the search API.
            content = result['content']
        summarized_results[url] = {'title': result['title'], 'content': content}
    return summarized_results


class Summary(BaseModel):
    """A Pydantic schema for the structured output of the webpage summarization model."""
 
   # The main, concise summary of the webpage.
    summary: str = Field(description="Concise summary of the webpage content")

    # A list of direct quotes or key excerpts to preserve important verbatim information.
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")


def summarize_webpage_content(webpage_content: str) -> str:
    """Summarizes a single piece of webpage content using our configured summarization model."""
    try:
        # We bind our 'Summary' Pydantic schema to the summarization model.
        # Summarization model
        summarization_model = get_hf_model(model_name="moonshotai/Kimi-K2-Instruct")
        structured_model = summarization_model.with_structured_output(Summary)


        # We invoke the LLM with our detailed summarization prompt.
        summary_result = structured_model.invoke([
            HumanMessage(content=summarize_webpage_prompt.format(
                webpage_content=webpage_content, 
                date=get_today_str()
            ))
        ])

        # We format the structured output into a clean, human-readable string.
        formatted_summary = (
            f"<summary>\n{summary_result.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary_result.key_excerpts}\n</key_excerpts>"
        )
        return formatted_summary
    except Exception as e:

        # If summarization fails, we fall back to a simple truncation of the raw content.
        print(f"Failed to summarize webpage: {str(e)}")
        return webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    ) -> str:
    """
    A tool that fetches results from the Tavily search API and performs content summarization.

    Args:
        query (str): A single, specific search query to execute.
        max_results (int): The maximum number of results to return.
        topic (Literal["general", "news", "finance"]): The topic to filter results by ('general', 'news', 'finance').
    
    Returns:
        str: A formatted string of the deduplicated and summarized search results.
    """

    # 1. Execute the search.
    search_results = tavily_search_multiple([query], max_results=max_results, topic=topic, include_raw_content=True)

    # 2. Deduplicate the results.
    unique_results = deduplicate_search_results(search_results)

    # 3. Process and summarize the content.
    summarized_results = process_search_results(unique_results)

    # 4. Format the final output.
    return format_search_output(summarized_results)