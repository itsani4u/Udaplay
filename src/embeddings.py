from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import Config

def get_gemini_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=Config.GEMINI_API_KEY
    )
