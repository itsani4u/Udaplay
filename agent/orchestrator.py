from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import Config
from tools.internal_search import query_internal_games_db
from tools.web_search import search_web_gaming_data
from src.logger import logger

class UdaPlayOrchestrator:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0.2
        )

    def route_and_execute(self, user_query: str) -> dict:
        logger.info(f"Agent Pipeline Triggered for Input: '{user_query}'")
        
        db_result = query_internal_games_db.invoke({"query": user_query})
        
        if db_result in ["NO_RESULTS_FOUND", "NO_CONFIDENT_RESULTS_FOUND"]:
            logger.info("Internal database query returned low confidence/no data. Executing Fallback -> Tavily Web Search.")
            web_result = search_web_gaming_data.invoke({"query": user_query})
            retrieved_context = web_result
            source_type = "Web Search (Tavily)"
        else:
            logger.info("High confidence internal match retrieved.")
            retrieved_context = db_result
            source_type = "Internal Database (FAISS)"

        prompt = f"""
You are UdaPlay, an expert AI gaming analyst assistant. Answer the user's query clearly using only the provided context.

User Query: {user_query}
Context ({source_type}):
{retrieved_context}

Instructions:
1. Provide a well-structured response.
2. Explicitly cite your source type at the bottom: "Source: {source_type}".
"""
        response = self.llm.invoke(prompt)
        
        return {
            "query": user_query,
            "context": retrieved_context,
            "source_type": source_type,
            "final_response": response.content
        }
