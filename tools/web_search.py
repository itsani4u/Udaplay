from langchain.tools import tool
from langchain_community.utilities import TavilySearchAPIWrapper
from src.config import Config
from src.logger import logger

tavily_search = TavilySearchAPIWrapper(tavily_api_key=Config.TAVILY_API_KEY)

@tool
def search_web_gaming_data(query: str) -> str:
    """Searches the internet for real-time video game news, current developments, and missing game info."""
    logger.info(f"Tool Executing: search_web_gaming_data with query: '{query}'")
    try:
        results = tavily_search.results(query, max_results=3)
        formatted = [f"Source: {r['url']}\nContent: {r['content']}" for r in results]
        return "\n\n---\n\n".join(formatted)
    except Exception as e:
        logger.error(f"Web Search error: {str(e)}")
        return f"Web search failed due to error: {str(e)}"
