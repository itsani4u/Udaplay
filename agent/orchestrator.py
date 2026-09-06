from tools.internal_search import query_internal_games_db
from tools.web_search import search_web_gaming_data
from langchain_google_genai import ChatGoogleGenerativeAI
from src.logger import logger

class UdaPlayOrchestrator:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    def run(self, query: str) -> str:
        # Invoke the LangChain tool directly
        internal_result = query_internal_games_db.invoke(query)

        # Check if vectorstore returned valid documents
        if internal_result not in ["NO_RESULTS_FOUND", "NO_CONFIDENT_RESULTS_FOUND"]:
            logger.info(f"Sufficient internal data found for query: '{query}'")
            prompt = f"Answer the user query based ONLY on this context:\n\n{internal_result}\n\nQuery: {query}"
            response = self.llm.invoke(prompt)
            return f"{response.content}\n\n*Source: Internal Database (FAISS)*"

        # Fallback to DuckDuckGo when internal database has no confident match
        logger.info(f"No confident internal match for '{query}'. Invoking web search tool...")
        web_result = search_web_gaming_data.invoke(query) if hasattr(search_web_gaming_data, 'invoke') else search_web_gaming_data(query)

        if web_result and "Web search error" not in web_result:
            prompt = f"Answer the user query using the following web search context:\n\n{web_result}\n\nQuery: {query}"
            response = self.llm.invoke(prompt)
            return f"{response.content}\n\n*Source: Web Search (DuckDuckGo)*"

        return "I apologize, but I could not find relevant information in either the internal database or via web search."
