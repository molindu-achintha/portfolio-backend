import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "portfolio-rag")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY") # For LLM response generation

settings = Settings()
