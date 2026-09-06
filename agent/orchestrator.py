from tools.internal_search import query_internal_games_db
from tools.web_search import search_web_gaming_data
from langchain_google_genai import ChatGoogleGenerativeAI

class UdaPlayOrchestrator:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    def run(self, query: str) -> str:
        # 1. Attempt retrieval from local vector database
        context = query_internal_games_db(query)

        # 2. Check if context contains valid data
        if context and not self._is_refusal_or_empty(context):
            prompt = f"""
            Answer the user query based ONLY on the following internal context:
            Context: {context}
            Query: {query}
            """
            response = self.llm.invoke(prompt)
            return f"{response.content}\n\n*Source: Internal Database (FAISS)*"

        # 3. Fallback: Trigger DuckDuckGo Search
        print(f"[Orchestrator] Missing/Low internal context for '{query}'. Switching to DuckDuckGo...")
        web_results = search_web_gaming_data(query)

        if web_results and "Web search error" not in web_results:
            prompt = f"""
            Answer the user query using the following web search context:
            Web Search Results: {web_results}
            Query: {query}
            """
            response = self.llm.invoke(prompt)
            return f"{response.content}\n\n*Source: Web Search (DuckDuckGo)*"

        return "I could not find information on this topic in either the internal database or via web search."

    def _is_refusal_or_empty(self, context: str) -> bool:
        """Helper to catch empty strings or common missing-data statements."""
        if not context or not context.strip():
            return True
        refusal_phrases = [
            "not available",
            "cannot answer",
            "no information",
            "provided context does not contain"
        ]
        return any(phrase in context.lower() for phrase in refusal_phrases)
