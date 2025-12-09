import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "portfolio-rag")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") # For LLM response generation (Legacy)
    BYTEZ_API_KEY = os.getenv("BYTEZ_API_KEY") # Bytez provider (Legacy)
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") # OpenRouter (Current)

settings = Settings()
