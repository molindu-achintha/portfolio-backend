from pinecone import Pinecone
from pinecone import ServerlessSpec 
import time
from app.core.config import settings

pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index(settings.PINECONE_INDEX_NAME)

def _ensure_index_exists():
    # BGE model produces 384-dim embeddings
    REQUIRED_DIMENSION = 384
    
    existing_indexes = [i.name for i in pc.list_indexes()]
    if settings.PINECONE_INDEX_NAME in existing_indexes:
        # Check if dimension matches
        details = pc.describe_index(settings.PINECONE_INDEX_NAME)
        if details.dimension != REQUIRED_DIMENSION:
            print(f"Index dimension mismatch (Found: {details.dimension}, Expected: {REQUIRED_DIMENSION}). Deleting...")
            pc.delete_index(settings.PINECONE_INDEX_NAME)
            time.sleep(5) # Wait for deletion
            existing_indexes.remove(settings.PINECONE_INDEX_NAME)

    if settings.PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating index '{settings.PINECONE_INDEX_NAME}' ({REQUIRED_DIMENSION} dim, cosine)...")
        try:
            pc.create_index(
                name=settings.PINECONE_INDEX_NAME,
                dimension=REQUIRED_DIMENSION, # BGE-small-en-v1.5 dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready
            while not pc.describe_index(settings.PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
            print(f"Index '{settings.PINECONE_INDEX_NAME}' created successfully!")
        except Exception as e:
            print(f"Failed to create index automatically: {e}")
            raise e

def query_vectors(vector: list, top_k: int = 3):
    return index.query(
        vector=vector,
        top_k=top_k,
        include_metadata=True
    )

def upsert_vectors(vectors):
    _ensure_index_exists()
    # Re-initialize index after potential recreation
    index = pc.Index(settings.PINECONE_INDEX_NAME) 
    index.upsert(vectors=vectors)

def delete_all_vectors():
    try:
        _ensure_index_exists()
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        index.delete(delete_all=True)
        print(f"Index '{settings.PINECONE_INDEX_NAME}' cleared successfully.")
    except Exception as e:
        print(f"Error clearing index: {e}")
