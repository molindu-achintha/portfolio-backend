import logging
import time
from huggingface_hub import InferenceClient
from app.core.config import settings

logger = logging.getLogger(__name__)

# BGE Small - Best Free Embedding Model (Same dimensions as current index: 384)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

def get_embedding(text: str):
    """
    Generate embedding using Hugging Face InferenceClient (BGE model).
    """
    if not settings.HUGGINGFACE_API_KEY:
        raise Exception("HUGGINGFACE_API_KEY is not set in .env")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            client = InferenceClient(model=EMBED_MODEL, token=settings.HUGGINGFACE_API_KEY)
            
            # Feature extraction returns numpy array
            embedding = client.feature_extraction(text)
            
            # Ensure list format
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            
            # Handle batch format [[...]]
            if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                return embedding[0]
                
            return embedding

        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "loading" in error_str.lower():
                time.sleep(5)
                continue
            if "429" in error_str:
                time.sleep(2 ** attempt)
                continue
            
            logger.error(f"HF Embedding Error: {e}")
            raise e
            
    raise Exception("Embedding generation failed after retries.")
