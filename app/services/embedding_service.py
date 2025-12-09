"""
Embedding service - now uses OpenCLIP for multimodal embeddings.
This file maintains backward compatibility with existing imports.
"""
from app.services.clip_service import get_text_embedding as get_embedding
from app.services.clip_service import get_image_embedding

__all__ = ['get_embedding', 'get_image_embedding']
