from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import uvicorn
import shutil
import os
import re

from app.core.config import settings
from app.services import embedding_service, llm_service, vector_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Portfolio RAG Backend (Bytez + Pinecone)")

# CORS (Frontend Access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    model_provider: str = "bytez"
    history: List[ChatMessage] = []

# Keywords for profile image
PROFILE_IMAGE_KEYWORDS = [
    'who are you', 'about you', 'your profile', 'your image', 'your photo', 
    'picture of you', 'look like', 'yourself', 'introduce'
]

# Keywords that trigger showing media (images/video)
MEDIA_KEYWORDS = [
    'show', 'image', 'picture', 'photo', 'video', 'demo', 'visual', 'see', 'look'
]

def should_show_profile_image(query: str) -> bool:
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in PROFILE_IMAGE_KEYWORDS)

def should_show_media(query: str) -> bool:
    """Check if user is explicitly asking for visuals/media."""
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in MEDIA_KEYWORDS)

def clean_suggestion(suggestion: str) -> str:
    s = suggestion.replace('**', '').replace('*', '').replace('"', '')
    s = re.sub(r'^\d+\.\s*', '', s)
    s = re.sub(r'^-\s*', '', s)
    return s.strip()

def normalize_text(text: str) -> str:
    if not text: return ""
    return re.sub(r'[^a-z0-9]', '', text.lower())

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Backend is running with Bytez + Pinecone"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        query = request.message
        logger.info(f"Received query: {query}")

        show_profile_img = should_show_profile_image(query)
        wants_media = should_show_media(query)  # User explicitly wants visuals

        # Embed Query
        try:
            embedding = embedding_service.get_embedding(query)
        except Exception as e:
             logger.error(f"Embedding failed: {e}")
             raise HTTPException(status_code=500, detail=f"Embedding Error: {str(e)}")
        
        # Retrieve Context
        search_results = vector_db.query_vectors(embedding, top_k=100)
        
        context_text = ""
        image_candidates = {}
        video_candidates = {}
        profile_images = []

        for match in search_results.get('matches', []):
            score = match['score']
            metadata = match.get('metadata', {})
            text = metadata.get('text', '')
            match_type = metadata.get('type', '')
            
            if score > 0.25:
                context_text += text + "\n---\n"
                
                title = metadata.get('title', '')
                
                if 'image_url' in metadata:
                    img_url = metadata['image_url']
                    if match_type == 'project':
                        image_candidates[img_url] = title
                    elif match_type == 'profile' and show_profile_img:
                        profile_images.append(img_url)
                
                if 'video_url' in metadata:
                    vid_url = metadata['video_url']
                    video_candidates[vid_url] = title

        # Generate Answer
        try:
            full_response = llm_service.generate_response(query, context_text, "")
             
            response_text = full_response
            suggestions = []
             
            if "<<SUGGESTIONS>>" in full_response:
                parts = full_response.split("<<SUGGESTIONS>>")
                response_text = parts[0].strip()
                suggestions_raw = parts[1].strip().split("\n")
                suggestions = [clean_suggestion(s) for s in suggestions_raw if s.strip()]
             
            # STRICT Media Filtering
            # Only show media if:
            # 1. User explicitly asked for visuals (wants_media=True) AND
            # 2. The project title is mentioned in the QUERY (not just response)
            final_images = set(profile_images)
            final_videos = set()
             
            query_norm = normalize_text(query)
             
            # Only add media if user explicitly asked for it
            if wants_media:
                for url, raw_title in image_candidates.items():
                    t_norm = normalize_text(raw_title)
                    # Match if title keywords are in query
                    if len(t_norm) > 3 and t_norm in query_norm:
                        final_images.add(url)
            
                for url, raw_title in video_candidates.items():
                    t_norm = normalize_text(raw_title)
                    if len(t_norm) > 3 and t_norm in query_norm:
                        final_videos.add(url)

            # Append to Text
            started_visuals = False
             
            if final_images:
                for img_url in final_images:
                    if img_url not in response_text:
                        if not started_visuals:
                            response_text += "\n\n**Visuals:**\n"
                            started_visuals = True
                        alt = "Visual"
                        for u, t in image_candidates.items():
                            if u == img_url: alt = t; break
                        response_text += f"![{alt}]({img_url})\n"
             
            if final_videos:
                if not started_visuals:
                    response_text += "\n\n**Visuals:**\n"
                for vid_url in final_videos:
                    t = "Video"
                    for u, title in video_candidates.items():
                        if u == vid_url: t = title; break
                    response_text += f"\n🎥 **Watch Video:** [{t}]({vid_url})\n"

        except Exception as e:
             logger.error(f"Generation failed: {e}")
             raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

        return {
            "response": response_text,
            "provider": "bytez",
            "context_used": len(context_text) > 0,
            "images": list(final_images), 
            "videos": list(final_videos),
            "suggestions": suggestions
        }

    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))
