from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import uvicorn
import shutil
import os

from app.core.config import settings
from app.services import embedding_service, llm_service, vector_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Portfolio RAG Backend (Groq + Pinecone)")

# CORS (Frontend Access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    model_provider: str = "groq" # defaults to groq

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Backend is running with Groq + Pinecone"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        query = request.message
        provider = request.model_provider
        logger.info(f"Received query: {query} (Provider: {provider})")

        # Embed Query (Using Generic Service - BGE)
        try:
            embedding = embedding_service.get_embedding(query)
        except Exception as e:
             logger.error(f"Embedding failed: {e}")
             raise HTTPException(status_code=500, detail=f"Embedding Error: {str(e)}")
        
        # Retrieve Context
        # Increase top_k to ensure we get all projects if requested
        search_results = vector_db.query_vectors(embedding, top_k=100)
        
        context_text = ""
        unique_images = {} 

        # Pinecone returns a dictionary-like object, matches are in 'matches' key
        for match in search_results.get('matches', []):
            score = match['score']
            metadata = match.get('metadata', {})
            text = metadata.get('text', '')
            logger.info(f"  Match: {match['id']} (score: {score:.3f}) - Metadata Keys: {list(metadata.keys())}")
            
            # Threshold to filter irrelevant context
            if score > 0.15:
                context_text += text + "\n---\n"
                
                # Collect Image URLs
                if 'image_url' in metadata:
                    img_url = metadata['image_url']
                    title = metadata.get('title', 'Visual')
                    unique_images[img_url] = title

        logger.info(f"Retrieved context length: {len(context_text)}")
        logger.info(f"Found {len(unique_images)} unique images: {list(unique_images.keys())}")

        # Generate Answer (Using Groq / Llama 3)
        # We default to Groq for speed and quality
        try:
             response = llm_service.generate_response(query, context_text)
             
             # Append images to response if they exist and are not already mentioned
             if unique_images:
                 started_visuals_section = False
                 
                 for img_url, title in unique_images.items():
                     # Only append if the URL is NOT ANYWHERE in the response text
                     if img_url not in response:
                         if not started_visuals_section:
                             response += "\n\n**Visuals:**\n"
                             started_visuals_section = True
                         response += f"![{title}]({img_url})\n"
                     else:
                         logger.info(f"Skipping duplicate image in response: {img_url}")

        except Exception as e:
             logger.error(f"Generation failed: {e}")
             raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

        return {
            "response": response,
            "provider": "groq",
            "context_used": len(context_text) > 0,
            "images": list(unique_images) 
        }

    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def health():
    return {"status": "ok"}
