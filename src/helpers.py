from datetime import datetime
from typing import List

from src.nodes.research_agent import MAX_CONTEXT_LENGTH, summarize_webpage_content

def get_today_str() -> str:
    """A simple utility function to get the current date in a human-readable string format."""
    # We format the date as "Day Mon Day, Year" (e.g., "Mon Dec 25, 2025").
    return datetime.now().strftime("%a %b %-d, %Y")


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